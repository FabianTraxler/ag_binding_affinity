from typing import Any, Dict, List, Tuple, Optional, Callable

from scipy.stats import chi2

import wandb
import numpy as np
import math
import pandas as pd
import os
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
import logging
import torch
from scipy import stats
from torch.optim import Adam
from torch.utils.data import DataLoader as DL_torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from functools import partial
# import pdb

from ..dataset import AffinityDataset
from ..dataset.advanced_data_utils import complexes_from_dms_datasets, get_bucket_dataloader
from ..model import AffinityGNN, TwinWrapper
from ..train.wandb_config import configure
from ..utils.config import get_data_paths
from ..utils.visualize import plot_correlation
logger = logging.getLogger(__name__)  # setup module logger


def forward_step(model: AffinityGNN, data: Dict, device: torch.device) -> Tuple[Dict, Dict]:
    """ Perform one forward step with the data and model provided

    Offers functionality to train models with different types of data so that the correct information is extracted from
    the data object

    Args:
        model: torch module used to make predictions
        data: data object (multiple types possible)
        device: device to use for forward step

    Returns:
        Tuple: Model output and Labels
    """

    if data["relative"]:  # relative data available
        twin_model = TwinWrapper(model)
        if model.scaled_output:
            temperature = 0.2
        else:
            temperature = 2
        output = twin_model(data, rel_temperature=temperature)
    else:
        data["input"]["graph"] = data["input"]["graph"].to(device)
        if "deeprefine_graph" in data["input"]:
            data["input"]["deeprefine_graph"] = data["input"]["deeprefine_graph"].to(device)

        output = model(data["input"])
    output["relative"] = data["relative"]

    label = get_label(data, device)

    return output, label


def get_label(data: Dict, device: torch.device) -> Dict:
    # We compute all possible labels, so that we can combine different loss functions
    label = {
        "relative": data["relative"]
    }
    for output_type in ["E", "-log(Kd)"]:
        if data["relative"]:
            label_1 = (data["input"][0]["graph"][output_type] > data["input"][1]["graph"][output_type])
            label_2 = (data["input"][1]["graph"][output_type] > data["input"][0]["graph"][output_type])
            label[f"{output_type}_prob"] = torch.stack((label_1.float(), label_2.float()), dim=-1)
            label[f"{output_type}_stronger_label"] = label_2.long()  # Index of the stronger binding complex
            label[f"{output_type}"] = data["input"][0]["graph"][output_type].float().to(device).view(-1,1)
            label[f"{output_type}2"] = data["input"][1]["graph"][output_type].float().to(device).view(-1,1)
            label[f"{output_type}_difference"] = label[f"{output_type}"] - label[f"{output_type}2"]
        else:
            # We add an additional dimension to match the output (Batchsize, N-Channel=1)
            label[output_type] = data["input"]["graph"][output_type].float().to(device).view(-1,1)

    # TODO Try to return NLL values of data if available!
    return label

def get_loss(loss_functions: str, label: Dict, output: Dict) -> torch.Tensor:
    # Different loss functions are seperated with a +
    # Optionally, they can contain a weight seperated with a -
    # E.g. args.loss_function = L2-1+L1-0.1+relative_l1-2+relative_l2-0.1+relative_ce
    loss_types = [x.split("-") for x in loss_functions.split("+")]
    loss_types = [(x[0], float(x[1])) if len(x) == 2 else (x[0], 1.) for x in loss_types]

    losses = []
    # Using Mean reduction might weight losses unequally (assume e.g. one batch (size 64) contain 1 E value and 63 -log(Kd) values
    # Therefore, we should try sum reduction
    loss_functions: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "L1": partial(torch.nn.functional.l1_loss, reduction='sum'),
        "L2": partial(torch.nn.functional.mse_loss, reduction='sum'),
        "NLL": lambda output, label, var: torch.sum(0.5 * (torch.log(torch.clamp(var,1e-6)) + (output - label)**2 / torch.clamp(2 * var,1e-6))),
        "RL2": lambda output, label: torch.sqrt(torch.nn.functional.mse_loss(output, label, reduction='sum') + 1e-10), # We add 1e-10 to avoid nan gradients when mse=0
        "relative_L1": partial(torch.nn.functional.l1_loss, reduction='sum'),
        "relative_L2": partial(torch.nn.functional.mse_loss, reduction='sum'),
        "relative_RL2": lambda output, label: torch.sqrt(torch.nn.functional.mse_loss(output, label, reduction='sum') + 1e-10),
        "relative_ce": partial(torch.nn.functional.nll_loss, reduction='sum'),
        "relative_cdf": lambda output, label: torch.nn.functional.nll_loss(output, label, reduction="sum"),
        "cosinesim": lambda output, label: -1 * torch.nn.functional.cosine_similarity(output, label, dim=0, eps=1e-6).mean(),
    }

    for (criterion, weight) in loss_types:
        # As we use sum reduction but don't want to scale our loss to large, we divide by the batch size
        weight = weight / output["-log(Kd)"].shape[0]
        loss_fn = loss_functions[criterion]
        for output_type in ["E", "-log(Kd)"]:

            if criterion in ["L1", "L2", "RL2"]:
                valid_indices = ~torch.isnan(label[output_type])
                if valid_indices.sum() > 0:
                    losses.append(weight * loss_fn(output[output_type][valid_indices],
                                               label[output_type][valid_indices]))
                # if output["relative"]:  # previously, we also used the second data point for absolute loss
                #     valid_indices = ~torch.isnan(label[f"{output_type}2"])
                #     if valid_indices.sum() > 0:
                #         losses.append(weight * loss_fn(output[f"{output_type}2"][valid_indices],
                #                                    label[f"{output_type}2"][valid_indices]))
            if criterion == "cosinesim":
                valid_indices = ~torch.isnan(label[output_type])
                if valid_indices.sum() > 1:
                    losses.append(weight * loss_fn(output[output_type][valid_indices],
                                                   label[output_type][valid_indices]))

            elif criterion == "NLL":
                # This loss optimizes the predicted Likelihood jointly
                valid_indices = ~torch.isnan(label[output_type])
                if valid_indices.sum() > 0:
                    losses.append(weight * loss_fn(output[output_type][valid_indices],
                                                   label[output_type][valid_indices],
                                                   output["uncertainty"][valid_indices]))
            elif output["relative"] and criterion.startswith("relative"):
                if criterion in ["relative_L1", "relative_L2", "relative_RL2"]:
                    output_key = f"{output_type}_difference"
                    label_key = f"{output_type}_difference"
                elif criterion == "relative_ce":
                    output_key = f"{output_type}_logit"
                    label_key = f"{output_type}_stronger_label"
                elif criterion == "relative_cdf":
                    output_key = f"{output_type}_logit_cdf"
                    label_key = f"{output_type}_stronger_label"
                valid_indices = ~torch.isnan(label[label_key])
                if valid_indices.sum() > 0:
                    losses.append(weight * loss_fn(output[output_key][valid_indices],
                                                   label[label_key][valid_indices]))

    if any([torch.isnan(l) for l in losses]):
        logging.error("Somehow a nan in loss")
    assert len(losses) > 0, f"No valid lossfunction was given with:{loss_functions} and relative data {output['relative']}"

    return sum(losses)



def train_epoch(model: AffinityGNN, train_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device = torch.device("cpu"), tqdm_output: bool = True) -> Tuple[AffinityGNN, float]:
    """ Train model for one epoch given the train and validation dataloaders

    Args:
        model: torch module used to train
        train_dataloader: Dataloader with training examples
        val_dataloaders: Dataloaders with validation examples
        optimizer: Optimizer object used to perform update step of model parameter
        device: Device to use for training

    Returns:
        Tuple: updated model and dict with results
    """
    total_loss_train = 0.0
    model.train()

    for data in tqdm(train_dataloader, disable=not tqdm_output):
        optimizer.zero_grad()

        output, label = forward_step(model, data, device)
        loss = get_loss(data["loss_criterion"], label, output)
        total_loss_train += loss.item()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)

        loss.backward()
        optimizer.step()

    return model, total_loss_train


def validate_epochs(model: AffinityGNN, val_dataloaders: List[DataLoader], device: torch.device, args: Namespace):
    # Validate the results on all provided data loaders
    model.eval()
    results = {}
    for val_dataloader in val_dataloaders:
        total_loss_val = 0
        all_predictions = []
        all_binary_predictions = []
        all_labels = []
        all_binary_labels = []
        all_pdbs = []

        for data in tqdm(val_dataloader, disable=not args.tqdm_output):
            output, label = forward_step(model, data, device)
            loss = get_loss(data["loss_criterion"], label, output)
            total_loss_val += loss.item()

            if label["E"].isnan().any():
                if label["-log(Kd)"].isnan().any():
                    logger.error(f"Both E and -log(Kd) are NaN. Skipping batch (len {len(label['E'])})")
                    continue
                else:
                    output_type = "-log(Kd)"
            else:
                output_type = "E"

            all_predictions.append(output[f"{output_type}"].flatten().detach().cpu().numpy())
            all_labels.append(label[f"{output_type}"].flatten().detach().cpu().numpy())
            if data["relative"]:
                pdb_ids_1 = [filepath.split("/")[-1].split(".")[0] for filepath in data["input"][0]["filepath"]]
                pdb_ids_2 = [filepath.split("/")[-1].split(".")[0] for filepath in data["input"][1]["filepath"]]
                all_pdbs.extend(list(zip(pdb_ids_1, pdb_ids_2)))
                all_binary_labels.append(label[f"{output_type}_stronger_label"].detach().cpu().numpy())
                all_binary_predictions.append(torch.argmax(output[f"{output_type}_prob"].detach().cpu(), dim=1).numpy())
            else:
                all_pdbs.extend([filepath.split("/")[-1].split(".")[0] for filepath in data["input"]["filepath"]])

        all_predictions = np.concatenate(all_predictions) if len(all_predictions) > 0 else np.array([])
        all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])

        if args.scale_values:
            all_labels = all_labels * (args.scale_max - args.scale_min) + args.scale_min
            all_predictions = all_predictions * (args.scale_max - args.scale_min) + args.scale_min

        # sometimes all_pdbs contains tuples of strings (when using RELATIVE LOSS), so this is code to simply flatten the list
        # TODO this looks very ugly and can probably be done nicer!
        all_pdbs_tmp = []
        for pdb in all_pdbs:
            if isinstance(pdb, tuple) or isinstance(pdb, list):
                all_pdbs_tmp.extend(list(pdb))
            else:
                all_pdbs_tmp.append(pdb)
        all_pdbs = all_pdbs_tmp
        # this is needed because all_pdbs is a list of strings, which is not directly usable by np.concatenate
        # need to convert to np.array
        # TODO maybe this is never used? Also we change the length of all_pdbs, make it longer than others
        if len(all_pdbs) > 0 and not isinstance(all_pdbs[0], np.ndarray):
            all_pdbs = [np.array([pdb]) for pdb in all_pdbs]
        all_pdbs = np.concatenate(np.array(all_pdbs)) if len(all_pdbs) > 0 else np.array([])

        all_binary_predictions = np.concatenate(all_binary_predictions) if len(
            all_binary_predictions) > 0 else np.array([])
        all_binary_labels = np.concatenate(all_binary_labels) if len(all_binary_labels) > 0 else np.array([])

        if len(all_binary_labels) > 0:
            acc = accuracy_score(all_binary_labels, all_binary_predictions)
        else:
            acc = np.nan

        pearson_corr = stats.pearsonr(all_labels, all_predictions)
        spearman_corr = stats.spearmanr(all_labels, all_predictions)
        rmse = math.sqrt(np.square(np.subtract(all_labels, all_predictions)).mean())


        # We could have the same dateset with different losses in one validation, therefore, we need to add the loss?!
        # However, on wandb it would be nice to see the results together when training with different losses...

        #results[f"{val_dataloader.dataset.full_dataset_name}#{val_dataloader.dataset.loss_criterion}"] = {
        results[f"{val_dataloader.dataset.full_dataset_name}"] = {
            "val_loss": total_loss_val,
            "pearson_correlation": pearson_corr[0],
            "pearson_correlation_p": pearson_corr[1],
            "spearman_correlation": spearman_corr[0],
            "spearman_correlation_p": spearman_corr[1],
            "all_labels": all_labels,
            "all_predictions": all_predictions,
            "all_binary_labels": all_binary_labels, # TODO Never used?
            "all_binary_predictions": all_binary_predictions, # TODO Never used?
            "all_pdbs": all_pdbs, # TODO Never used?
            "val_accuracy": acc,
            "rmse": rmse
        }



    return results


def get_optimizer(args: Namespace, model: torch.nn.Module):
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay_factor,
            verbose=args.verbose)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_decay_factor,
            patience=args.patience or 10, verbose=args.verbose)
    elif args.lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_decay_factor,
            patience=args.patience or 10, verbose=args.verbose)
        # Stop as soon as LR is reduced by one step -> constant LR with early stopping
        args.stop_at_learning_rate = args.learning_rate
    else:
        raise ValueError(f"LR Scheduler {args.lr_scheduler} not supported.")

    return optimizer, scheduler


def load_model(num_node_features: int, num_edge_features: int, dataset_names: List[str], args: Namespace,
               device: torch.device = torch.device("cpu")) -> AffinityGNN:
    """ Load a specific model type and initialize it randomly

    Args:
        num_node_features: Dimension of features of nodes in the graph
        num_edge_features: Dimension of features of edges in the graph
        dataset_names: List of dataset names used for training
        args: CLI arguments
        device: Device the model will be loaded on

    Returns:
        nn.Module: model on specified device
    """
    if args.pretrained_model in args.config["MODELS"] and args.load_pretrained_weights:
        pretrained_model_path = args.config["MODELS"][args.pretrained_model]["model_path"]
    else:
        pretrained_model_path = None

    model = AffinityGNN(num_node_features, num_edge_features, args,
                        num_nodes=args.max_num_nodes,
                        pretrained_model=args.pretrained_model, pretrained_model_path=pretrained_model_path,
                        gnn_type=args.gnn_type, num_gat_heads=args.attention_heads,
                        layer_type=args.layer_type, num_gnn_layers=args.num_gnn_layers,
                        channel_halving=args.channel_halving, channel_doubling=args.channel_doubling,
                        node_type=args.node_type,
                        aggregation_method=args.aggregation_method,
                        nonlinearity=args.nonlinearity,
                        num_fc_layers=args.num_fc_layers, fc_size_halving=args.fc_size_halving,
                        device=device,
                        scaled_output=args.scale_values,  # seems to work worse than if the model learns it on its own
                        dataset_names=complexes_from_dms_datasets(dataset_names, args))

    return model


def bucket_learning(model: AffinityGNN, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset],
                    args: Namespace) -> Tuple[Dict, AffinityGNN, Any]:
    """ Train a model using the bucket (multitask) learning approach

    Provides the utility of generating batches for different data types

    Args:
        model: torch module used to train
        train_datasets: List of datasets with trainings examples
        val_datasets: list of datasets with validation examples
        args: CLI arguments used to define training procedure
        bucket_size: Integer specifying the maximal size of the dataset buckets

    Returns:
        Tuple: Results as dict, trained model
    """
    plot_path = os.path.join(args.config["plot_path"],
                             f"{args.wandb_name}/bucket_learning/val_set_{args.validation_set}")
    Path(plot_path).mkdir(exist_ok=True, parents=True)

    dataset2optimize = args.target_dataset.split("#")[0]

    wandb_inst, wdb_config, use_wandb, run = configure(args, model.dataset_specific_layer)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer, scheduler = get_optimizer(args, model)

    # initialize training values
    best_loss = np.inf
    best_pearson_corr = -np.inf
    best_spearman_corr = -np.inf
    best_rmse = np.inf
    patience = args.patience
    results = {
        "epoch_loss": [],
        "epoch_corr": [],
        "epoch_spearman_corr": [],
        "target_epoch_loss": [],
        "target_epoch_corr": [],
        "target_epoch_spearman_corr": [],
        "target_epoch_rmse": [],
    }

    best_model = deepcopy(model)
    n_datasets = len(model.dataset_specific_layer.linear.bias)
    dataset_specific_column_names = ["Epoch"] + [f"Weight {i}" for i in range(n_datasets)] + [f"Bias {i}" for i in
                                                                                              range(n_datasets)]
    dataset_specific_layer_df = pd.DataFrame(columns=dataset_specific_column_names)
    #table = wandb_inst.Table(columns=dataset_specific_column_names)

    for epoch in range(args.max_epochs):

        if epoch == args.warm_up_epochs:
            model.unfreeze()

        # create new buckets for each epoch (implements shuffling)
        # This ensures that different part of the dataset are used when geometric mean is set
        # Also shuffles the batches as otherwise each bucket dataloader returns the same combination of samples in one batch
        train_dataloader, val_dataloaders = get_bucket_dataloader(args, train_datasets, val_datasets)

        model, total_loss_train = train_epoch(model, train_dataloader, optimizer, device,
                                              args.tqdm_output)

        # Save modelspecific weights
        dataset_specific_layer_df.loc[epoch] = (epoch, *model.dataset_specific_layer.linear.weight[:, 0].tolist(), *model.dataset_specific_layer.linear.bias.tolist())
        #table.add_data(epoch, *model.dataset_specific_layer.linear.weight[:, 0].tolist(), *model.dataset_specific_layer.linear.bias.tolist())

        wandb_log = {"dataset_specific_layer": wandb_inst.Table(dataframe=dataset_specific_layer_df)}

        ####   Validation  ######
        val_results = validate_epochs(model, val_dataloaders, device, args)

        val_result = {"total_val_loss": 0,
                      "pearson_correlation": 0,
                      "spearman_correlation": 0,
                      }
        for dataset_name, dataset_results in val_results.items():
            logger.info(
                f'Epochs: {epoch + 1}  | Dataset: {dataset_name} | Val-Loss: {dataset_results["val_loss"]: .3f} | '
                f'Val-corr: {dataset_results["pearson_correlation"]: .4f} | p-value r=0: {dataset_results["pearson_correlation_p"]: .4f} | '
                f'Val-spearman-rho: {dataset_results["spearman_correlation"]: .4f} | p-value sp=0: {dataset_results["spearman_correlation_p"]: .4f} | '
                f'Val-RMSE: {dataset_results["rmse"]} | '
                f'Val-Acc: {dataset_results["val_accuracy"]: .4f}')

            if np.isnan(dataset_results["pearson_correlation"]):
                logger.error(f"Pearson correlation is NaN for dataset {dataset_name}.")
                # return results, best_model, wandb_inst

            wandb_log[f"{dataset_name}_val_loss"] = dataset_results["val_loss"]
            wandb_log[f"{dataset_name}_val_corr"] = dataset_results["pearson_correlation"]
            wandb_log[f"{dataset_name}_val_spearman_corr"] = dataset_results["spearman_correlation"]
            wandb_log[f"{dataset_name}_val_rmse"] = dataset_results["rmse"]
            wandb_log[f"{dataset_name}_val_acc"] = dataset_results["val_accuracy"]
            val_result["total_val_loss"] += dataset_results["val_loss"]
            val_result["pearson_correlation"] += dataset_results["pearson_correlation"]
            val_result["spearman_correlation"] += dataset_results["spearman_correlation"]

        #We need to ensure that we use the correct amount of samples here
        wandb_log["val_loss"] = val_result["total_val_loss"] / np.sum([len(v) for v in val_dataloaders])
        wandb_log["train_loss"] = total_loss_train / len(train_dataloader)
        wandb_log["pearson_correlation"] = val_result["pearson_correlation"]
        wandb_log["spearman_correlation"] = val_result["spearman_correlation"]
        wandb_log["learning_rate"] = optimizer.param_groups[0]['lr']
        wandb_log["rmse"] = val_results[dataset2optimize]["rmse"]

        results["epoch_loss"].append(val_result["total_val_loss"] / len(val_dataloaders))
        results["epoch_corr"].append(val_result["pearson_correlation"] / len(val_dataloaders))
        results["epoch_spearman_corr"].append(val_result["spearman_correlation"] / len(val_dataloaders))

        if dataset2optimize in val_results.keys():
            results["target_epoch_loss"].append(val_results[dataset2optimize]["val_loss"])
            results["target_epoch_corr"].append(val_results[dataset2optimize]["pearson_correlation"])
            results["target_epoch_spearman_corr"].append(val_results[dataset2optimize]["spearman_correlation"])
            results["target_epoch_rmse"].append(val_results[dataset2optimize]["rmse"])
        else:
            raise ValueError(f"Somehow the dataset2optimize {dataset2optimize} is not contained in the validation datasets!")

        if args.lr_scheduler == "exponential":
            scheduler.step()
        else:
            scheduler.step(metrics=results['target_epoch_loss'][-1])
            if args.patience is not None:
                patience = args.patience - scheduler.num_bad_epochs


        if val_results[dataset2optimize]["rmse"] < best_rmse:  # Mihail analyzed that RMSE works well. It also is the metric we report so it fits
            patience = args.patience
            best_loss = val_results[dataset2optimize]["val_loss"]
            best_pearson_corr = val_results[dataset2optimize]["pearson_correlation"]
            best_spearman_corr = val_results[dataset2optimize]["spearman_correlation"]

            best_rmse = val_results[dataset2optimize]["rmse"]
            for dataset_name, dataset_results in val_results.items(): # correlation plot for each dataset
                plot_correlation(x=dataset_results["all_labels"],
                                 y=dataset_results["all_predictions"],
                                 path=os.path.join(plot_path, f"{dataset_name}Epoch{epoch}.png"))

            best_data = [[x, y]
                         for x, y
                         in zip(val_results[dataset2optimize]["all_predictions"],
                                val_results[dataset2optimize]["all_labels"])]
            best_table = wandb_inst.Table(data=best_data, columns=["predicted", "true"])
            wandb_inst.log({"scatter_plot": wandb_inst.plot.scatter(best_table, "predicted", "true",
                                                          title="Label vs. Predictions")})

            best_model = deepcopy(model)
        elif args.lr_scheduler == 'exponential' and patience is not None:
            patience -= 1

        logger.info(
            f'Epochs: {epoch + 1} | Total-Train-Loss: {total_loss_train / len(train_dataloader) : .3f}'
            f' | Total-Val-Loss: {val_result["total_val_loss"] / len(val_dataloaders) : .3f} | Patience: {patience} ')

        if epoch %5 == 0:
            benchmark_results = run_and_log_benchmarks(model, args)
            wandb_log.update(benchmark_results)
        wandb_inst.log(wandb_log, commit=True)
        stop_training = epoch > args.warm_up_epochs

        for param_group in optimizer.param_groups:
            stop_training = stop_training and (
                param_group['lr'] < args.stop_at_learning_rate)

        if args.lr_scheduler == 'exponential' and patience is not None and patience < 0 and epoch > args.warm_up_epochs:
            stop_training = True

        if stop_training:
            if use_wandb:
                run.summary[f"{dataset2optimize}_val_loss"] = best_loss
                run.summary[f"{dataset2optimize}_val_corr"] = best_pearson_corr
                run.summary[f"{dataset2optimize}_val_spearman_corr"] = best_spearman_corr
                run.summary[f"{dataset2optimize}_val_rmse"] = best_rmse
            break

        model.train()

        # only force recomputation in first epoch and then use precomputed graphs
        if args.save_graphs:
            for train_dataset in train_datasets:
                train_dataset.force_recomputation = False
            for val_dataset_ in val_datasets:
                val_dataset_.force_recomputation = False

    results["best_loss"] = best_loss
    results["best_correlation"] = best_pearson_corr
    results["best_spearman_correlation"] = best_spearman_corr
    results["best_rmse"] = best_rmse

    return results, best_model, wandb_inst

def log_gradients(model: AffinityGNN):
    from torch_geometric.nn import GATv2Conv
    from torch.nn import Linear
    gnn_gradients = []
    for gnn_layer in model.graph_conv.gnn_layers:
        summed_layer_gradient = 0
        if isinstance(gnn_layer, GATv2Conv):
            summed_layer_gradient += torch.sum(gnn_layer.att.grad).item()
            summed_layer_gradient += torch.sum(gnn_layer.lin_l.weight.grad).item()
            summed_layer_gradient += torch.sum(gnn_layer.lin_r.weight.grad).item()
        gnn_gradients.append(round(summed_layer_gradient, 6))

    fc_gradients = []
    for fc_layer in model.regression_head.fc_layers:
        summed_layer_gradient = 0
        if isinstance(fc_layer, Linear):
            summed_layer_gradient += torch.sum(fc_layer.weight.grad).item()
        fc_gradients.append(round(summed_layer_gradient, 6))

    logger.debug(f"Summed regression head gradients >>> {fc_gradients}")
    logger.debug(f"Summed GNN gradients >>> {gnn_gradients}")

    return gnn_gradients, fc_gradients


def evaluate_model(model: AffinityGNN, dataloader: DataLoader, args: Namespace, tqdm_output: bool = True,
                   device: torch.device = torch.device("cpu"), plot_path: Optional[str] = None) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
    # TODO Why do we have different validations at different places (train_epoch, bucket_learning, ...)
    total_loss_val = 0
    all_predictions = np.array([])
    all_labels = np.array([])
    all_pdbs = []
    all_uncertainties = np.array([])
    model.eval()
    for data in tqdm(dataloader, disable=not tqdm_output):
        output, label = forward_step(model, data, device)

        loss = get_loss(data["loss_criterion"], label, output)

        total_loss_val += loss.item()

        try:
            output_type = "E" if dataloader.dataset.affinity_type == "E" else "-log(Kd)"
        except AttributeError:
            output_type = "E" if dataloader.dataset.datasets[0].affinity_type == "E" else "-log(Kd)"  # hacky

        all_predictions = np.append(all_predictions, output[f"{output_type}"].flatten().detach().cpu().numpy())
        all_uncertainties = np.append(all_uncertainties, output["uncertainty"].flatten().detach().cpu().numpy())
        all_labels = np.append(all_labels, label[f"{output_type}"].detach().cpu().numpy())

        all_pdbs.extend([ filepath.split("/")[-1].split(".")[0] for filepath in data["input"]["filepath"]])
        # if len(all_labels) > 2:
            # break

    # scale prediction back to original values
    if args.scale_values:
        all_labels = all_labels * (args.scale_max - args.scale_min) + args.scale_min
        all_predictions = all_predictions * (args.scale_max - args.scale_min) + args.scale_min
        #all_uncertainties = all_uncertainties * (args.scale_max - args.scale_min)**2
    val_loss = total_loss_val / (len(all_predictions))
    try:
        pearson_corr = stats.pearsonr(all_labels, all_predictions)[0]
        spearman_corr = stats.spearmanr(all_labels, all_predictions)[0]
    except ValueError:
        logging.warning(f"nan in predictions or labels:\n{all_labels}\n{all_predictions}")
        pearson_corr = None
        spearman_corr = None

    rmse = math.sqrt(np.square(np.subtract(all_labels, all_predictions)).mean())

    # TODO pull out the plotting too
    if plot_path is not None:

        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plot_correlation(x=all_labels, y=all_predictions,
                         path=plot_path)

    res_df = pd.DataFrame({
                "pdb": all_pdbs,
                "prediction": all_predictions,
                "labels": all_labels,
                "uncertainties": all_uncertainties,
                "squared_error": np.square(np.subtract(all_labels, all_predictions))
            })
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "val_loss": val_loss,
        "rmse": rmse
    }, res_df


def get_benchmark_score(model: AffinityGNN, args: Namespace, tqdm_output: bool = True, plot_path: Optional[str] = None) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
    if not hasattr(get_benchmark_score, "dataset"):
        get_benchmark_score.dataset = AffinityDataset(args.config, False, "AntibodyBenchmark", "L2",
                                node_type=args.node_type,
                                max_nodes=args.max_num_nodes,
                                interface_distance_cutoff=args.interface_distance_cutoff,
                                interface_hull_size=args.interface_hull_size,
                                max_edge_distance=args.max_edge_distance,
                                pretrained_model=args.pretrained_model,
                                scale_values=args.scale_values,
                                scale_min=args.scale_min,
                                scale_max=args.scale_max,
                                save_graphs=args.save_graphs,
                                force_recomputation=args.force_recomputation,
                                preprocess_data=args.preprocess_graph,
                                preprocessed_to_scratch=args.preprocessed_to_scratch,
                                num_threads=args.num_workers,
                                load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path)
                                )

    dataloader = DL_torch(get_benchmark_score.dataset, num_workers=args.num_workers, batch_size=1,
                              collate_fn=AffinityDataset.collate)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return evaluate_model(model, dataloader, args=args, tqdm_output=tqdm_output, device=device,
                                     plot_path=plot_path)


def get_abag_test_score(model: AffinityGNN, args: Namespace, tqdm_output: bool = True, plot_path: Optional[str] = None,
                        validation_set: Optional[int] = None) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:

    if not hasattr(get_abag_test_score, "dataset"):
        summary_path, _ = get_data_paths(args.config, "abag_affinity")
        summary_df = pd.read_csv(summary_path, index_col=0)
        if validation_set is None:
            summary_df = summary_df[summary_df["test"]]
        elif validation_set < 0:
            summary_df = summary_df[summary_df["validation"] != ((-1*validation_set) - 1)]
        else:
            summary_df = summary_df[summary_df["validation"] == validation_set]

        test_pdbs_ids = summary_df.index.tolist()

        get_abag_test_score.dataset = AffinityDataset(args.config, False, "abag_affinity", "L2", test_pdbs_ids,
                                node_type=args.node_type,
                                max_nodes=args.max_num_nodes,
                                interface_distance_cutoff=args.interface_distance_cutoff,
                                interface_hull_size=args.interface_hull_size,
                                max_edge_distance=args.max_edge_distance,
                                pretrained_model=args.pretrained_model,
                                scale_values=args.scale_values,
                                scale_min=args.scale_min,
                                scale_max=args.scale_max,
                                save_graphs=args.save_graphs,
                                force_recomputation=args.force_recomputation,
                                preprocess_data=args.preprocess_graph,
                                preprocessed_to_scratch=args.preprocessed_to_scratch,
                                num_threads=args.num_workers,
                                load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path)
                                )

    dataloader = DL_torch(get_abag_test_score.dataset, num_workers=args.num_workers, batch_size=1,
                              collate_fn=AffinityDataset.collate)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return evaluate_model(model, dataloader, args=args, tqdm_output=tqdm_output, device=device, plot_path=plot_path)


def combine_pvalues(pvalues):
    # Take the natural log of all p-values
    ln_pvalues = np.log(pvalues)

    # Apply Fisher's method
    test_statistic = -2 * np.sum(ln_pvalues)

    # The test statistic follows a chi-square distribution with 2n degrees of freedom
    # Calculate the new p-value from this distribution
    combined_pvalue = chi2.sf(test_statistic, 2*len(pvalues))

    return combined_pvalue

def _compute_grouped_correlations(res_df):
    metrics = {}
    metrics["pearson"] = res_df.groupby("pdb").apply(lambda group: stats.pearsonr(group.labels, group.prediction)[0])
    metrics["pearson_weighted_mean"] = np.sum(res_df.groupby("pdb").apply(lambda group: stats.pearsonr(group.labels, group.prediction)[0] * len(group))) / len(res_df)
    metrics["pearson_pval"] = combine_pvalues(res_df.groupby("pdb").apply(lambda group: stats.pearsonr(group.labels, group.prediction)[1]).values)

    metrics["spearman"] = res_df.groupby("pdb").apply(lambda group: stats.spearmanr(group.labels, group.prediction)[0])
    metrics["spearman_weighted_mean"] = np.sum(res_df.groupby("pdb").apply(lambda group: stats.spearmanr(group.labels, group.prediction)[0] * len(group))) / len(res_df)
    metrics["spearman_pval"] = combine_pvalues(res_df.groupby("pdb").apply(lambda group: stats.spearmanr(group.labels, group.prediction)[1]).values)

    return metrics


def get_skempi_corr(model: AffinityGNN, args: Namespace, tqdm_output: bool = True, plot_path: str = None) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
    """
    Take the available Skempi mutations for validation

    Note that the spearman-aggregated p-value is not reliable for small sample sizes (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
    """
    if not hasattr(get_skempi_corr, "dataset"):
        get_skempi_corr.dataset = AffinityDataset(args.config, False, "SKEMPI.v2", "L2",
                                node_type=args.node_type,
                                max_nodes=args.max_num_nodes,
                                interface_distance_cutoff=args.interface_distance_cutoff,
                                interface_hull_size=args.interface_hull_size,
                                max_edge_distance=args.max_edge_distance,
                                pretrained_model=args.pretrained_model,
                                scale_values=args.scale_values,
                                scale_min=args.scale_min,
                                scale_max=args.scale_max,
                                save_graphs=args.save_graphs,
                                force_recomputation=args.force_recomputation,
                                preprocess_data=args.preprocess_graph,
                                preprocessed_to_scratch=args.preprocessed_to_scratch,
                                num_threads=args.num_workers,
                                load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path)
                                )

    dataloader = DL_torch(get_skempi_corr.dataset, num_workers=args.num_workers, batch_size=1,
                            collate_fn=AffinityDataset.collate)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    metrics, res_df = evaluate_model(model, dataloader, args=args, tqdm_output=tqdm_output, device=device,
                        plot_path=plot_path)

    # take everything after dash (-)
    res_df["mutation"] = res_df["pdb"].apply(lambda v: v.split("-")[1])
    res_df["pdb"] = res_df["pdb"].apply(lambda v: v.split("-")[0])
    # split results by PDBs and compute separate correlations

    metrics.update({f"grouped_{k}": v for k, v in _compute_grouped_correlations(res_df).items()})

    res_df["grouped_correlations_pearson"] = res_df["pdb"].apply(metrics["grouped_pearson"].get)
    res_df["grouped_spearman_correlations"] = res_df["pdb"].apply(metrics["grouped_spearman"].get)

    return metrics, res_df

    # results.append(res)

    # return np.mean([v[0] for v in results]), np.mean([v[1] for v in results])


def _synthetic_benchmark(model, args, validation_i=1):
    """

    Args:
        validation_i:  `0` is test

    """

    # Buffered dataset generation
    if not hasattr(_synthetic_benchmark, "test_set"):

        summary_path, _ = get_data_paths(args.config, "synthetic_ddg")
        summary_df = pd.read_csv(summary_path, index_col=0)

        validation_pdbs_ids = summary_df[summary_df["validation"] == validation_i].index.tolist()
        test_pdbs_ids = summary_df[summary_df["test"]].index.tolist()

        common_args = dict(
            config=args.config, is_relaxed=False, dataset_name="synthetic_ddg", loss_criterion="L2",
            node_type=args.node_type,
            max_nodes=args.max_num_nodes,
            interface_distance_cutoff=args.interface_distance_cutoff,
            interface_hull_size=args.interface_hull_size,
            max_edge_distance=args.max_edge_distance,
            pretrained_model=args.pretrained_model,
            scale_values=args.scale_values,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            save_graphs=args.save_graphs,
            force_recomputation=args.force_recomputation,
            preprocess_data=args.preprocess_graph,
            preprocessed_to_scratch=args.preprocessed_to_scratch,
            num_threads=args.num_workers,
            load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path)
        )

        test_set = AffinityDataset(**common_args, pdb_ids=test_pdbs_ids)
        validation_set = AffinityDataset(**common_args, pdb_ids=validation_pdbs_ids)

        _synthetic_benchmark.test_loader = DL_torch(test_set, num_workers=args.num_workers, batch_size=1,
                                                    collate_fn=AffinityDataset.collate)
        _synthetic_benchmark.validation_loader = DL_torch(validation_set, num_workers=args.num_workers, batch_size=1,
                                                    collate_fn=AffinityDataset.collate)

    # Run the model predictions and compute metrics
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    wandb_benchmark_log = {}
    for dl_name, dataloader in zip(["test", "validation"], [_synthetic_benchmark.test_loader, _synthetic_benchmark.validation_loader]):
        metrics, res_df = evaluate_model(model, dataloader, args=args, tqdm_output=args.tqdm_output, device=device, plot_path=None)
        wandb_benchmark_log.update({f"{dl_name}_{key}": value for key, value in metrics.items()})

        # Compute additional grouped metrics
        grouped_mean_corrs = _compute_grouped_correlations(res_df)
        wandb_benchmark_log.update({f"{dl_name}_grouped_{key}": value for key, value in grouped_mean_corrs.items()})

    return wandb_benchmark_log


def _experimental_benchmark(model, args) -> Dict[str, Any]:
    # Run benchmarks
    benchmark_metrics, benchmark_df = get_benchmark_score(model, args, tqdm_output=args.tqdm_output)

    skempi_metrics, skempi_df = get_skempi_corr(model, args, tqdm_output=args.tqdm_output)

    abag_test_plot_path = os.path.join(args.config["plot_path"], f"abag_affinity_test_cv{args.validation_set}.png")

    test_metrics, test_df = get_abag_test_score(model, args, tqdm_output=args.tqdm_output,
                                                           plot_path=abag_test_plot_path,
                                                           validation_set=args.validation_set)

    # When a negative validation set is provided, we use all but the corresponding set.
    # As we have to deal with the case validation_set==0 we first add 1 before flipping the sign
    train_set_indicator = (args.validation_set+1)*-1 if args.validation_set is not None else -1
    train_metrics, train_df = get_abag_test_score(model, args, tqdm_output=args.tqdm_output,
                                                           plot_path=abag_test_plot_path,
                                                           validation_set=train_set_indicator)

    # Log all metrics
    all_metrics = {f"{dataset}_{metric}": value for dataset, metrics in
                   {"benchmark_test": benchmark_metrics, "skempi_test": skempi_metrics, "abag_test": test_metrics, "abag_train": train_metrics}.items()
                   for metric, value in metrics.items()}

    for metric, value in all_metrics.items():
        logger.info(f"{metric} >>> {value}")

    benchmark_df["Dataset"] = "benchmark"
    skempi_df["Dataset"] = "skempi"
    test_df["Dataset"] = "abag_test"
    train_df["Dataset"] = "abag_train"
    full_df = pd.concat([test_df, benchmark_df, skempi_df, train_df])
    all_metrics["Full_Predictions"] = wandb.Table(dataframe=full_df)

    if "uncertainties" in full_df.columns:
        all_metrics.update({"abag_test_uncertainty": np.mean(test_df["uncertainties"]),
                            "abag_train_uncertainty": np.mean(train_df["uncertainties"]),
                            "benchmark_test_uncertainty": np.mean(benchmark_df["uncertainties"]),
                            "skempi_test_uncertainty": np.mean(skempi_df["uncertainties"]),
                            })

    return all_metrics

def run_and_log_benchmarks(model, args, wandb_inst=None, experimental_benchmark=False, synthetic_benchmark=False):
    """
    Run all our benchmarks on the given model and report logs.

    Datasets are cached by binding them to their respective functions (better would be to use a class)
    """

    wandb_benchmark_log = {}

    if experimental_benchmark:
        wandb_benchmark_log.update(_experimental_benchmark(model, args))
    if synthetic_benchmark:
        wandb_benchmark_log.update(_synthetic_benchmark(model, args))


    if wandb_inst is not None:
        wandb_inst.log(wandb_benchmark_log, commit=True)

    return wandb_benchmark_log
