import glob
from typing import Dict, List, Tuple, Union, Optional
from matplotlib._api import itertools
import numpy as np
import math
import pandas as pd
import random
import os
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
import logging
import torch
from scipy import stats
from torch.optim import Adam
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader as DL_torch
from torch.utils.data import Subset
import yaml
from torch_geometric.data import DataLoader
from tqdm import tqdm
from scipy.stats.mstats import gmean
from sklearn.metrics import accuracy_score
import scipy.spatial as sp
# import pdb

from ..dataset import AffinityDataset
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
        output["affinity_type"] = data["affinity_type"]

    label = get_label(data, device)

    return output, label


def get_label(data: Dict, device: torch.device) -> Dict:
    # We compute all possible labels, so that we can combine different loss functions
    label = {}
    if data["relative"]:
        label_1 = (data["input"][0]["graph"].y > data["input"][1]["graph"].y)
        label_2 = (data["input"][1]["graph"].y > data["input"][0]["graph"].y)
        label["x_prob"] = torch.stack((label_1.float(), label_2.float()), dim=-1)
        label["x_stronger"] = label_2.long()  # Index of the stronger binding complex
        label["x"] = data["input"][0]["graph"].y.to(device)
        label["x2"] = data["input"][1]["graph"].y.to(device)
        label["difference"] = label["x"] - label["x2"]
    else:
        label["x"] = data["input"]["graph"].y.to(device)

    # TODO Try to return NLL values of data if available!
    return label

def get_loss(loss_functions: str, label: Dict, output: Dict) -> torch.Tensor:
    # Different loss functions are seperated with a +
    # Optionally, they can contain a weight seperated with a -
    # E.g. args.loss_function = L2-1+L1-0.1+relative_l1-2+relative_l2-0.1+relative_ce
    loss_types = [x.split("-") for x in loss_functions.split("+")]
    loss_types = [(x[0], float(x[1])) if len(x) == 2 else (x[0], 1.) for x in loss_types]
    losses = []
    for criterion, weight in loss_types:
        if criterion == "L1":
            loss = weight * torch.nn.functional.l1_loss(output["x"], label["x"])
            if output["relative"]:
                loss += weight * torch.nn.functional.l1_loss(output["x2"], label["x2"])
        elif criterion == "L2":
            loss = weight * torch.nn.functional.mse_loss(output["x"], label["x"])
            if output["relative"]:
                loss += weight * torch.nn.functional.mse_loss(output["x2"], label["x2"])
        elif criterion == "relative_L1" and output["relative"]:
            loss = weight * torch.nn.functional.l1_loss(output["difference"], label["difference"])
        elif criterion == "relative_L2" and output["relative"]:
            loss = weight * torch.nn.functional.mse_loss(output["difference"], label["difference"])
        elif criterion == "relative_ce" and output["relative"]:
            loss = weight * torch.nn.functional.nll_loss(output["x_logit"], label["x_stronger"])
        elif criterion == "relative_cdf" and output["relative"]:
            loss = weight * torch.nn.functional.nll_loss(output["x_prob_cdf"].log(), label["x_stronger"])
        else:
            raise ValueError(
                f"Loss_Function must either in ['L1','L2','relative_L1','relative_L2','relative_ce'] but got {criterion}")
        losses.append(loss)
    assert len(losses) > 0, f"No valid lossfunction was given with:{loss_functions} and relative data {output['relative']}"
    return sum(losses)


def train_epoch(model: AffinityGNN, train_dataloader: DataLoader, val_dataloaders: List[DataLoader],
                optimizer: torch.optim, device: torch.device = torch.device("cpu"), tqdm_output: bool = True) -> Tuple[AffinityGNN, List, np.ndarray, float]:
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

        # pdb.set_trace()
        output, label = forward_step(model, data, device)
        loss = get_loss(data["loss_criterion"], label, output)
        total_loss_train += loss.item()

        loss.backward()
        optimizer.step()
        # break

    model.eval()
    results = []
    for val_dataloader in val_dataloaders:
        total_loss_val = 0
        all_predictions = []
        all_binary_predictions = []
        all_labels = []
        all_binary_labels = []
        all_pdbs = []
        for data in tqdm(val_dataloader, disable=not tqdm_output):
            output, label = forward_step(model, data, device)

            loss = get_loss(data["loss_criterion"], label, output)
            total_loss_val += loss.item()

            all_predictions.append(output["x"].flatten().detach().cpu().numpy())
            all_labels.append(label["x"].detach().cpu().numpy())
            if data["relative"]:
                pdb_ids_1 = [filepath.split("/")[-1].split(".")[0] for filepath in data["input"][0]["filepath"]]
                pdb_ids_2 = [filepath.split("/")[-1].split(".")[0] for filepath in data["input"][1]["filepath"]]
                all_pdbs.extend(list(zip(pdb_ids_1, pdb_ids_2)))
                all_binary_labels.append( label["x_stronger"].detach().cpu().numpy())
                all_binary_predictions.append(torch.argmax(output["x_prob"].detach().cpu(), dim=1).numpy())

            else:
                #We need to ensure that binary labels have the length of the validation dataset as we later slice the datasets appart.
                # TODO maybe we could store them in a dataset specific dict instead?
                all_binary_labels.append(np.zeros(label["x"].shape[0]))
                all_binary_predictions.append(np.zeros(label["x"].shape[0]))
                all_pdbs.extend([filepath.split("/")[-1].split(".")[0] for filepath in data["input"]["filepath"]])

            #all_labels = np.append(all_labels, label.numpy())
            #all_predictions = np.append(all_predictions, output["x"].flatten().detach().cpu().numpy())

        val_loss = total_loss_val / len(all_predictions)

        # if len(all_predictions) > 2:
        #     break
        all_predictions = np.concatenate(all_predictions) if len(all_predictions) > 0 else np.array([])
        all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
        # Does not work when both relative and absolute datasets are there (as we have tuple and singe strings)
        #all_pdbs = np.concatenate(all_pdbs) if len(all_pdbs) > 0 else np.array([])

        all_binary_predictions = np.concatenate(all_binary_predictions) if len(all_binary_predictions) > 0 else np.array([])
        all_binary_labels = np.concatenate(all_binary_labels) if len(all_binary_labels) > 0 else np.array([])
        val_loss = total_loss_val / (len(all_predictions) + len(all_binary_predictions))

        if len(all_binary_labels) > 0:
            acc = accuracy_score(all_binary_labels, all_binary_predictions)
        else:
            acc = np.nan

        pearson_corr = stats.pearsonr(all_labels, all_predictions)
        rmse = math.sqrt(np.square(np.subtract(all_labels, all_predictions)).mean())

        results.append({
            "val_loss": val_loss,
            "pearson_correlation": pearson_corr[0],
            "pearson_correlation_p": pearson_corr[1],
            "all_labels": all_labels,
            "all_predictions": all_predictions,
            "all_binary_labels": all_binary_labels,
            "all_binary_predictions": all_binary_predictions,
            "all_pdbs": all_pdbs,
            "total_val_loss": total_loss_val,
            "val_accuracy": acc
        })

    return model, results, total_loss_train


def train_loop(model: AffinityGNN, train_dataset: AffinityDataset, val_datasets: List[AffinityDataset], args: Namespace) -> \
        Tuple[Dict, AffinityGNN]:
    """ Train model with train dataset and provide results and statistics based on validation set

    Train for specified epochs, saving best model and plotting correlation plots

    Args:
        model: torch module used to train
        train_dataset: Dataset with trainings examples
        val_dataset: Dataset with validation examples
        args: CLI arguments used to define training procedure

    Returns:
        Tuple: Dict with results and statistics, best model
    """

    # create folder to store correlation plots
    plot_path = os.path.join(args.config["plot_path"],
                             f"{args.node_type}/sequential_learning/val_set_{args.validation_set}")
    prediction_path = os.path.join(args.config["prediction_path"],
                             f"{args.node_type}/sequential_learning/val_set_{args.validation_set}")
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    Path(prediction_path).mkdir(parents=True, exist_ok=True)

    wandb, wdb_config, use_wandb, run = configure(args)

    results = {f"{key}_val{i}": [] for key, i in
               itertools.product(["epoch_loss", "epoch_corr", "epoch_rmse", "epoch_acc"], range(len(val_datasets)))}

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer, scheduler = get_optimizer(args, model)

    scale_min = 0 if args.scale_values else args.scale_min
    scale_max = 1 if args.scale_values else args.scale_max

    best_loss = np.inf
    best_pearson_corr = -np.inf
    best_rmse = np.inf
    patience = args.patience

    best_model = deepcopy(model)

    train_dataloader, val_dataloaders = get_dataloader(args, train_dataset, val_datasets)
    for i in range(args.max_epochs):
        model, val_results, total_loss_train = train_epoch(model, train_dataloader, val_dataloaders, optimizer, device,
                                           args.tqdm_output)

        if args.lr_scheduler == "exponential":
            scheduler.step()
        else:
            scheduler.step(metrics=val_results[0]['val_loss']) # TODO we only use the first validation loader for scheduler
            if args.patience is not None:
                patience = args.patience - scheduler.num_bad_epochs

        for val_i, (val_result, val_dataset, val_dataloader) in enumerate(zip(val_results, val_datasets, val_dataloaders)):
            results[f"epoch_loss_val{val_i}"].append(val_result["val_loss"])
            results[f"epoch_corr_val{val_i}"].append(val_result["pearson_correlation"])
            results[f"epoch_acc_val{val_i}"].append(val_result["val_accuracy"])

            # scale back values before rsme calculation
            if len(val_result["all_labels"]) > 0:
                all_labels = val_result["all_labels"]
                all_predictions = val_result["all_predictions"]
                if args.scale_values:
                    all_labels = all_labels * (args.scale_max - args.scale_min) + args.scale_min
                    all_predictions = all_predictions * (args.scale_max - args.scale_min) + args.scale_min
                    val_result["rmse"] = math.sqrt(np.square(np.subtract(all_labels,all_predictions)).mean())
                    results[f"epoch_rmse_val{val_i}"].append(val_result["rmse"])
            else:
                val_result["rmse"] = np.nan
            results[f"epoch_rmse_val{val_i}"].append(val_result["rmse"])

            if val_result["val_loss"] < best_loss: # or val_result["pearson_correlation"] > best_pearson_corr:
                patience = args.patience
                best_loss = val_result["val_loss"]
                best_pearson_corr = val_result["pearson_correlation"]
                best_rmse = val_result["rmse"]

                if not np.isnan(val_result["rmse"]):
                    best_data = [[x, y] for (x, y) in zip(all_predictions, all_labels)]
                    best_table = wandb.Table(data=best_data, columns=["predicted", "true"])
                    wandb.log({"scatter_plot": wandb.plot.scatter(best_table, "predicted", "true",
                                                                title="Label vs. Predictions")})

                    plot_correlation(x=all_labels, y=all_predictions,
                                    path=os.path.join(plot_path, f"{train_dataset.full_dataset_name}.png"))
                    # result_df = pd.DataFrame({
                    #         "pdb": val_result["all_pdbs"],
                    #         "prediction": all_predictions,
                    #         "labels": all_labels
                    # })
                    # result_df.to_csv(os.path.join(prediction_path, f"{train_dataset.full_dataset_name}.csv"),
                    #                 index=False)
                best_model = deepcopy(model)
            elif args.lr_scheduler == 'exponential' and patience is not None:
                patience -= 1

            logger.info(
                f'Epochs: {i + 1} | Train-Loss: {total_loss_train / len(train_dataloader) : .3f}  | '
                f'Val{val_i}-Loss: {val_result["total_val_loss"] / len(val_dataloader) : .3f} | '
                f'Val{val_i}-r: {val_result["pearson_correlation"]: .4f} | '
                f'p-value{val_i} r=0: {val_result["pearson_correlation_p"]: .4f} | RMSE: {val_result["rmse"]} | '
                f'Val{val_i}-Acc: {val_result["val_accuracy"]: .4f} | '
                f'Patience: {patience} '
                f'LR: {optimizer.param_groups[0]["lr"]: .6f}')

            if val_dataset.affinity_type == "-log(Kd)" and np.isnan(val_result["pearson_correlation"]):
                preds_nan = any(np.isnan(val_result["all_predictions"]))
                preds_same = np.all(val_result["all_predictions"] == val_result["all_predictions"][0])
                labels_nan = any(np.isnan(val_result["all_labels"]))
                labels_same = np.all(val_result["all_labels"] == val_result["all_labels"][0])

                results["best_loss"] = best_loss
                results["best_correlation"] = best_pearson_corr
                results["best_rmse"] = best_rmse

                logger.error(
                    f"Pearon correlation is NaN. Preds NaN:{preds_nan}. Preds Same {preds_same}. Labels NaN: {labels_nan}. Labels Same {labels_same}")
                return results, best_model

            wandb_log = {
                f"val{val_i}_loss": val_result["total_val_loss"] / (len(val_dataset) / args.batch_size),
                "train_loss": total_loss_train / (len(train_dataset) / args.batch_size),
                f"val{val_i}_pearson_correlation": val_result["pearson_correlation"],
                f"val{val_i}_rmse": val_result["rmse"],
                f"{val_dataset.full_dataset_name}{val_i}_val_loss": val_result["total_val_loss"] / (
                        len(val_dataset) / args.batch_size),
                f"{val_dataset.full_dataset_name}{val_i}_val_corr": val_result["pearson_correlation"],
                f"{val_dataset.full_dataset_name}{val_i}_val_rmse": val_result["rmse"]
            }

            wandb.log(wandb_log, commit=True)

        stop_training = True

        for param_group in optimizer.param_groups:
            stop_training = stop_training and (
                param_group['lr'] < args.stop_at_learning_rate)

        if args.lr_scheduler == 'exponential' and patience is not None and patience < 0:
            stop_training = True

        if stop_training:
            if use_wandb:
                run.summary[f"{val_datasets[0].full_dataset_name}_val_corr"] = best_pearson_corr
                run.summary[f"{val_datasets[0].full_dataset_name}_val_loss"] = best_loss
                run.summary[f"{val_datasets[0].full_dataset_name}_val_rmse"] = best_rmse
            break

        model.train()

        # only force recomputation in first epoch and then use precomputed graphs
        if args.save_graphs:
            train_dataset.force_recomputation = False
            val_dataset.force_recomputation = False

        if i == 0:
            logger.debug(f"Max memory usage in epoch: {torch.cuda.max_memory_allocated()/(1<<20):,.0f} MB")

    results["best_loss"] = best_loss
    results["best_correlation"] = best_pearson_corr
    results["best_rmse"] = best_rmse

    return results, best_model


def get_optimizer(args: Namespace, model: torch.nn.Module):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

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

def complexes_from_dms_datasets(dataset_names: List, args) -> List:
    """
    Expand DMS dataset names to include the complexes
    """
    def complexes_from_dms_dataset(dataset_name, metadata):
        """
        Return complex names corresponding to dataset_name

        Args:
            dataset_name: Name of the dataset

        """

        if dataset_name.startswith("DMS-"):
            dataset_name = dataset_name.split("-")[1]

            return [":".join([dataset_name, complex["antibody"]["name"], complex["antigen"]["name"]])
                    for complex in metadata[dataset_name]["complexes"]]
        else:
            return [dataset_name]

    with open(os.path.join(args.config["DATASETS"]["path"], args.config["DATASETS"]["DMS"]["metadata"]), "r") as f:
        metadata = yaml.safe_load(f)

    unique_sets = np.unique([ds_name.split("#")[0] for ds_name in dataset_names]).tolist()
    with_complex_datasets = np.concatenate([complexes_from_dms_dataset(dataset_name, metadata) for dataset_name in unique_sets])
    return with_complex_datasets.tolist()


def load_model(num_node_features: int, num_edge_features: int, dataset_names: List[str], args: Namespace,
               device: torch.device = torch.device("cpu")) -> AffinityGNN:
    """ Load a specific model type and initialize it randomly

    Args:
        model_type: Name of the model to be loaded
        num_features: Dimension of features of inputs to the model
        args: CLI arguments
        device: Device the model will be loaded on

    Returns:
        nn.Module: model on specified device
    """
    if args.pretrained_model in args.config["MODELS"]:
        pretrained_model_path = args.config["MODELS"][args.pretrained_model]["model_path"]
    else:
        pretrained_model_path = ""

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


def get_dataloader(args: Namespace, train_dataset: AffinityDataset, val_datasets: List[AffinityDataset]) -> Tuple[
    DataLoader, DataLoader]:
    """ Get dataloader for train and validation dataset

    Use the DGL Dataloader for the DeepRefine Inputs

    Args:
        args: CLI arguments
        train_dataset: Torch dataset for train examples
        val_dataset: Torch dataset for validation examples

    Returns:
        Tuple: Train dataloader, validation dataloader
    """

    if args.batch_size == "max":
        batch_size = 0
    else:
        batch_size = args.batch_size

    if train_dataset.relative_data:
        train_dataset.update_valid_pairs()

    train_dataloader = DL_torch(train_dataset, num_workers=args.num_workers, batch_size=batch_size,
                                shuffle=args.shuffle, collate_fn=AffinityDataset.collate)

    val_dataloaders = [DL_torch(val_dataset, num_workers=args.num_workers, batch_size=batch_size,
                              collate_fn=AffinityDataset.collate) for val_dataset in val_datasets]

    return train_dataloader, val_dataloaders


def train_val_split(config: Dict, dataset_name: str, validation_set: int, validation_size: int = 20) -> Tuple[List, List]:
    """ Split data in a train and a validation subset

    For the abag_affinity datasets, we use the predefined split given in the csv, otherwise use random split

    Args:
        config: Dict with configuration info
        dataset_name: Name of the dataset
        validation_set: Integer identifier of the validation split (1,2,3)

    Returns:
        Tuple: List with indices for train and validation set
    """
    train_size = (100 - validation_size) / 100

    if "-" in dataset_name:
        # DMS data
        dataset_name, publication_code = dataset_name.split("-")
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_path = os.path.join(summary_path, publication_code + ".csv")
        summary_df = pd.read_csv(summary_path, index_col=0)

        if config["DATASETS"][dataset_name]["affinity_types"][publication_code] == "E":  # should be unnecessary
            summary_df = summary_df[(~summary_df["E"].isna()) &(~summary_df["NLL"].isna())]
        else:
            summary_df = summary_df[~summary_df["-log(Kd)"].isna()]
        # filter datapoints with missing PDB files
        data_path = Path(config["DATASETS"]["path"]) / config["DATASETS"][dataset_name]["folder_path"] / config["DATASETS"][dataset_name]["mutated_pdb_path"]
        summary_df = summary_df[summary_df.filename.apply(lambda fn: (data_path / fn).exists())]

        affinity_type = config["DATASETS"][dataset_name]["affinity_types"][publication_code]
        if affinity_type == "E":
            summary_df = summary_df[~summary_df.index.duplicated(keep='first')]

            # Normalize between 0 and 1 on a per-complex basis. This way value ranges of E and NLL fit, when computing possible pairs
            e_values = summary_df.groupby(summary_df.index.map(lambda i: i.split("-")[0]))["E"].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            e_values = e_values.values.reshape(-1,1).astype(np.float32)

            nll_values = summary_df["NLL"].values
            # Scale the NLLs to (0-1). The max NLL value in DMS_curated.csv is 4, so 0-1-scaling should be fine
            if np.max(nll_values) > np.min(nll_values):  # test that all values are not the same
                nll_values = (nll_values - np.min(nll_values)) / (np.max(nll_values) - np.min(nll_values))
                assert (nll_values < 0.7).sum() > (nll_values > 0.7).sum(), "Many NLL values are 'large'"
            else:
                nll_values = np.full_like(nll_values, 0.5)

            # TODO refactor this block (just always split, as it includes the else-case)
            if len(e_values) > 50000:
                n_splits = int(len(e_values) / 50000) + 1
                has_valid_partner = set()
                e_splits = np.array_split(e_values, n_splits)
                nll_splits = np.array_split(nll_values, n_splits)
                total_elements = 0
                for i in range(len(e_splits)):
                    for j in range(i, len(e_splits)):
                        split_e_dists = sp.distance.cdist(e_splits[i], e_splits[j])
                        split_nll_avg = (nll_splits[i][:, None] + nll_splits[j]) / 2
                        valid_pairs = (split_e_dists - split_nll_avg) >= 0
                        has_valid_partner_id = np.where(np.sum(valid_pairs, axis=1) > 0)[0] + total_elements
                        has_valid_partner.update(has_valid_partner_id)
                    total_elements += len(e_splits[i])
                has_valid_partner = np.fromiter(has_valid_partner, int, len(has_valid_partner))
                valid_partners = None
            else:
                e_dists = sp.distance.cdist(e_values, e_values)
                nll_avg = (nll_values[:, None] + nll_values) / 2

                valid_pairs = (e_dists - nll_avg) >= 0
                has_valid_partner = np.where(np.sum(valid_pairs, axis=1) > 0)[0]
                valid_partners = {
                    summary_df.index[idx]: set(summary_df.index[np.where(valid_pairs[idx])[0]].values.tolist()) for idx
                    in has_valid_partner}

            data_points_with_valid_partner = set(summary_df.index[has_valid_partner])
            total_valid_data_points = len(data_points_with_valid_partner)
            logger.debug(f"There are in total {len(data_points_with_valid_partner)} data points with valid partner")

            train_ids = set()
            val_ids = set()
            while len(data_points_with_valid_partner) > 0:
                pdb_idx = data_points_with_valid_partner.pop()
                if valid_partners is not None:
                    possible_ids = set(valid_partners[pdb_idx])
                    if len(possible_ids - train_ids) > 0:
                        # choose one of the unused ids
                        other_idx = random.choice(list(possible_ids - train_ids))
                    else:
                        # chosse one randomly with the risk that this idx is already in the train data
                        other_idx = random.choice(list(possible_ids))
                    data_points_with_valid_partner.discard(other_idx)
                else:
                    other_idx = pdb_idx
                if len(train_ids) >= total_valid_data_points * train_size:  # add to val ids
                    val_ids.add(pdb_idx)
                    val_ids.add(other_idx)
                else:
                    train_ids.add(pdb_idx)
                    train_ids.add(other_idx)

        elif affinity_type == "-log(Kd)":
            pdb_idx = summary_df.index.values.tolist()
            random.shuffle(pdb_idx)
            train_ids = pdb_idx[:int(train_size*len(pdb_idx))]
            val_ids = pdb_idx[int(train_size*len(pdb_idx)):]
        else:
            raise ValueError(
                f"Wrong affinity type given - expected one of (-log(Kd), E) but got {affinity_type}")

        train_ids = list(train_ids)
        val_ids = list(val_ids)

        #train_ids = ["phillips21_bindin:cr9114:h1wiscon05-SD166N;ID188S;AD194T;ND195A;KD210I".lower()] + train_ids[                                                                                               :5]
        #val_ids = val_ids[:2]

    else:
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path, index_col=0)

        summary_df["validation"] = summary_df["validation"].fillna("").astype("str")

        val_pdbs = summary_df.loc[summary_df["validation"].str.contains(str(validation_set)), "pdb"]
        train_pdbs = summary_df.loc[(~summary_df["validation"].str.contains(str(validation_set))) & (~summary_df["test"]), "pdb"]

        if "mutated_pdb_path" in config["DATASETS"][dataset_name]:  # only use files that were generated
            data_path = os.path.join(config["DATASETS"]["path"],
                                     config["DATASETS"][dataset_name]["folder_path"],
                                     config["DATASETS"][dataset_name]["mutated_pdb_path"])
            # TODO use `filename` in summary_df (see above! DRY?)
            all_files = glob.glob(data_path + "/*/*") + glob.glob(data_path + "/*")
            available_files = set(
                file_path.split("/")[-2].lower() + "-" + file_path.split("/")[-1].split(".")[0].lower() for file_path in
                all_files if file_path.split(".")[-1] == "pdb")
            summary_df = summary_df[summary_df.index.isin(available_files)]

        summary_df = summary_df[summary_df["-log(Kd)"].notnull()]

        val_ids = summary_df[summary_df["pdb"].isin(val_pdbs)].index.tolist()
        train_ids = summary_df[summary_df["pdb"].isin(train_pdbs)].index.tolist()

    return train_ids, val_ids


def load_datasets(config: Dict, dataset: str, validation_set: int, args: Namespace) -> Tuple[
    AffinityDataset, List[AffinityDataset]]:
    """ Get train and validation datasets for a specific dataset and data type

    1. Get train and validation splits
    2. Load the dataset in the specified data type class

    Args:
        config: training configuration as dict
        dataset: Name of the dataset:Usage of data (absolute, relative) - eg. SKEMPI.v2:relative
        validation_set: Integer identifier of the validation split (1,2,3)
        args: CLI arguments

    Returns:
        Tuple: Train and validation dataset
    """
    dataset_name, loss_types = dataset.split("#")

    # We missuse the data_type to allow setting different losses
    # The losses used for this dataset come after a # seperated by a comma, when multiple losses are used
    # Optionally, the losses can contain some weight using -
    # E.g. data_type = relative#l2-1,l1-0.1,relative_l1-2,relative_2-0.1,relative_ce-1

    if "relative" in loss_types:
        relative_data = True
    else:
        relative_data = False

    validation_size = args.validation_size if dataset == args.target_dataset else args.transfer_learning_validation_size  # TODO this does not look right!
    # validation_size = 10  # it's in percent
    train_ids, val_ids = train_val_split(config, dataset_name, validation_set, validation_size)

    if args.test:
        train_ids = train_ids[:20]
        val_ids = val_ids[:5]

    logger.debug(f"Get dataLoader for {dataset_name}#{loss_types}")
    train_data = AffinityDataset(config, args.relaxed_pdbs, dataset_name, loss_types,
                                 train_ids,
                                 node_type=args.node_type,
                                 max_nodes=args.max_num_nodes,
                                 interface_distance_cutoff=args.interface_distance_cutoff,
                                 interface_hull_size=args.interface_hull_size,
                                 max_edge_distance=args.max_edge_distance,
                                 pretrained_model=args.pretrained_model,
                                 scale_values=args.scale_values,
                                 scale_min=args.scale_min,
                                 scale_max=args.scale_max,
                                 relative_data=relative_data,
                                 save_graphs=args.save_graphs,
                                 force_recomputation=args.force_recomputation,
                                 preprocess_data=args.preprocess_graph,
                                 num_threads=args.num_workers,
                                 load_embeddings=args.embeddings_path,
                                 )

    val_datas = [AffinityDataset(config, relaxed, dataset_name, loss_types,  # TODO @marco val should be done with the same loss as training, right?
                                 val_ids,
                                 node_type=args.node_type,
                                 max_nodes=args.max_num_nodes,
                                 interface_distance_cutoff=args.interface_distance_cutoff,
                                 interface_hull_size=args.interface_hull_size,
                                 max_edge_distance=args.max_edge_distance,
                                 pretrained_model=args.pretrained_model,
                                 scale_values=args.scale_values,
                                 scale_min=args.scale_min,
                                 scale_max=args.scale_max,
                                 relative_data=relative_data,
                                 save_graphs=args.save_graphs,
                                 force_recomputation=args.force_recomputation,
                                 preprocess_data=args.preprocess_graph,
                                 num_threads=args.num_workers,
                                 load_embeddings=args.embeddings_path
                                 )
                 for relaxed in [bool(args.relaxed_pdbs)]
                 ]  # TODO disabling , not args.relaxed_pdbs for now. Enable once we generated all relaxed data

    return train_data, val_datas


def get_bucket_dataloader(args: Namespace, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset]):
    """ Get dataloader for bucket learning using a specific batch sampler

    Args:
        args: CLI arguments
        train_datasets: List of datasets with training examples
        val_datasets: List of datasets with validation examples
        train_bucket_size: Size of the training buckets
        config: config object from weights&bias

    Returns:
        Tuple: train dataloader, validation dataloader
    """
    train_buckets = []
    absolute_data_indices = []
    relative_data_indices = []
    relative_E_data_indices = []

    # Reshuffle pairs
    for idx, train_dataset in enumerate(train_datasets):
        if train_dataset.relative_data:
            train_dataset.update_valid_pairs()


    if args.bucket_size_mode == "min":
        train_bucket_size = [min([len(dataset) for dataset in train_datasets])] * len(train_datasets)
    elif args.bucket_size_mode == "geometric_mean":
        data_sizes = [len(dataset) for dataset in train_datasets]
        train_bucket_size =  [int(gmean(data_sizes))] * len(train_datasets)
    elif args.bucket_size_mode == "double_geometric_mean":
        data_sizes = [len(dataset) for dataset in train_datasets]
        geometric_mean = gmean(data_sizes)
        train_bucket_size = [ int(gmean([geometric_mean, size])) for size in data_sizes]
    else:
        raise ValueError(f"bucket_size_mode argument not supported: Got {args.bucket_size_mode} "
                         f"but expected one of [min, geometric_mean]")
    i = 0

    for idx, train_dataset in enumerate(train_datasets):
        if len(train_dataset) >= train_bucket_size[idx]:
            if train_dataset.dataset_name in args.target_dataset:  # args.target_dataset includes :absolute
                # always take all data points from the target dataset
                indices = range(len(train_dataset))
            else:
                # sample without replacement
                indices = random.sample(range(len(train_dataset)), train_bucket_size[idx])
        else:
            # sample with replacement
            indices = random.choices(range(len(train_dataset)), k=train_bucket_size[idx])

        train_buckets.append(Subset(train_dataset, indices))
        if train_dataset.relative_data and train_dataset.affinity_type == "E":
            relative_E_data_indices.extend(list(range(i, i + len(indices))))
        elif train_dataset.relative_data and train_dataset.affinity_type == "-log(Kd)":
            relative_data_indices.extend(list(range(i, i + len(indices))))
        else:
            absolute_data_indices.extend(list(range(i, i + len(indices))))
        i += len(indices)

    # shorten relative data for validation because DMS datapoints tend to have a lot of validation data if we use 10% split
    # TODO: find better way
    for dataset in val_datasets:
        if dataset.relative_data:
            dataset.relative_pairs = dataset.relative_pairs[:100]

    train_dataset = ConcatDataset(train_buckets)
    batch_sampler = generate_bucket_batch_sampler([absolute_data_indices, relative_data_indices, relative_E_data_indices], args.batch_size,
                                                  shuffle=args.shuffle)

    train_dataloader = DL_torch(train_dataset, num_workers=args.num_workers,
                                collate_fn=AffinityDataset.collate, batch_sampler=batch_sampler)
    # TODO: Change Batch Size
    val_dataloader = DL_torch(ConcatDataset(val_datasets), num_workers=args.num_workers, batch_size=1,
                              collate_fn=AffinityDataset.collate)

    return train_dataloader, val_dataloader


def generate_bucket_batch_sampler(data_indices_list: List, batch_size: int, shuffle: bool = False) -> List[List[int]]:
    """ Generate batches for different data types only combining examples of the same type

    Args:
        data_indices_list: List of data indices list
        batch_size: Size of the batches
        shuffle: Boolean indicator if examples and batches are to be shuffled

    Returns:
        List: Data indices in batches
    """

    all_batches = []

    for data_indices in data_indices_list:  # different batches for different data types
        if len(data_indices) == 0: continue  # ignore empty indices
        batches = []

        if shuffle:
            random.shuffle(data_indices)

        i = 0
        while i < len(data_indices):  # batch samples of same type
            batches.append(data_indices[i:i + batch_size])
            i += batch_size

        all_batches.extend(batches)

    if shuffle:
        random.shuffle(all_batches)

    return all_batches


def bucket_learning(model: AffinityGNN, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset],
                    args: Namespace) -> Tuple[Dict, AffinityGNN]:
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
                             f"{args.node_type}/bucket_learning/val_set_{args.validation_set}")
    Path(plot_path).mkdir(exist_ok=True, parents=True)

    dataset2optimize = args.target_dataset.split("#")[0]

    wandb, wdb_config, use_wandb, run = configure(args)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer, scheduler = get_optimizer(args, model)

    scale_min = 0 if args.scale_values else args.scale_min
    scale_max = 1 if args.scale_values else args.scale_max

    # initialize training values
    best_loss = np.inf
    best_pearson_corr = -np.inf
    best_rmse = np.inf
    patience = args.patience
    results = {
        "epoch_loss": [],
        "epoch_corr": [],
        "abag_epoch_loss": [],
        "abag_epoch_corr": []
    }

    best_model = deepcopy(model)

    for epoch in range(args.max_epochs):
        # create new buckets for each epoch
        train_dataloader, val_dataloader = get_bucket_dataloader(args, train_datasets, val_datasets)

        model, val_results, total_loss_train = train_epoch(model, train_dataloader, [val_dataloader], optimizer, device,
                                           args.tqdm_output)
        assert len(val_results) == 1, "If more than one val dataset is desired, need to implement another for loop in some way as in `train_loop`"
        val_result = val_results[0]

        dataset_results = {}

        wandb_log = {
            "val_loss": val_result["total_val_loss"] / len(val_dataloader),
            "train_loss": total_loss_train / len(train_dataloader),
            "pearson_correlation": val_result["pearson_correlation"],
        }

        i = 0

        for val_dataset in val_datasets:
            preds = val_result["all_predictions"][i:i + len(val_dataset)]
            labels = val_result["all_labels"][i:i + len(val_dataset)]
            # preds and labels are here always the energy so we can always use the l2 loss
            val_loss = np.mean((labels - preds)**2)
            if args.scale_values:
                all_labels = labels * (args.scale_max - args.scale_min) + args.scale_min
                all_predictions = preds * (args.scale_max - args.scale_min) + args.scale_min
            else:
                all_labels = labels
                all_predictions = preds
            rmse = math.sqrt(np.square(np.subtract(all_labels, all_predictions)).mean())
            pearson_corr = stats.pearsonr(labels, preds)
            if val_dataset.relative_data:
                binary_preds = val_result["all_binary_predictions"][i:i + len(val_dataset)]
                binary_labels = val_result["all_binary_labels"][i:i + len(val_dataset)]
                val_accuracy = accuracy_score(binary_labels, binary_preds)
            else:
                val_accuracy = np.nan

            logger.info(
                f'Epochs: {epoch + 1}  | Dataset: {val_dataset.full_dataset_name} | Val-Loss: {val_loss: .3f} | '
                f'Val-r: {pearson_corr[0]: .4f} | p-value r=0: {pearson_corr[1]: .4f} | '
                f'RMSE: {rmse} | '
                f'Val-Acc: {val_accuracy: .4f}')

            if val_dataset.affinity_type == "-log(Kd)" and np.isnan(pearson_corr[0]):
                preds_nan = any(np.isnan(preds))
                labels_nan = any(np.isnan(labels))
                logger.error(f"Pearson correlation is NaN. Preds NaN:{preds_nan}. Labels NaN: {labels_nan}")
                return results, best_model

            dataset_results[val_dataset.full_dataset_name] = {
                "val_loss": val_loss,
                "pearson_correlation": pearson_corr[0],
                "pearson_correlation_p": pearson_corr[1],
                "all_labels": labels,
                "all_predictions": preds,
                "rmse": rmse
            }
            wandb_log[f"{val_dataset.full_dataset_name}_val_loss"] = val_loss
            wandb_log[f"{val_dataset.full_dataset_name}_val_corr"] = pearson_corr[0]
            wandb_log[f"{val_dataset.full_dataset_name}_val_rmse"] = rmse
            i += len(val_dataset)

        results["epoch_loss"].append(val_result["val_loss"])
        results["epoch_corr"].append(val_result["pearson_correlation"])

        if args.lr_scheduler == "exponential":
            scheduler.step()
        else:
            scheduler.step(metrics=val_result['val_loss'])
            if args.patience is not None:
                patience = args.patience - scheduler.num_bad_epochs

        if dataset2optimize in dataset_results:
            results["abag_epoch_loss"].append(dataset_results[dataset2optimize]["val_loss"])
            results["abag_epoch_corr"].append(dataset_results[dataset2optimize]["pearson_correlation"])

        if dataset_results[dataset2optimize]["val_loss"] < best_loss:  # or dataset_results[dataset2optimize]["pearson_correlation"] > best_pearson_corr:
            patience = args.patience
            best_loss = dataset_results[dataset2optimize]["val_loss"]
            best_pearson_corr = dataset_results[dataset2optimize]["pearson_correlation"]
            best_rmse = dataset_results[dataset2optimize]["rmse"]
            for val_dataset in val_datasets:  # correlation plot for each dataset

                plot_correlation(x=dataset_results[val_dataset.full_dataset_name]["all_labels"],
                                 y=dataset_results[val_dataset.full_dataset_name]["all_predictions"],
                                 path=os.path.join(plot_path, f"{val_dataset.full_dataset_name}.png"))

            plot_correlation(x=val_result["all_labels"], y=val_result["all_predictions"],
                             path=os.path.join(plot_path, "all_bucket_predictions.png"))

            best_data = [[x, y]
                         for x, y
                         in zip(dataset_results[dataset2optimize]["all_predictions"],
                                dataset_results[dataset2optimize]["all_labels"])]
            best_table = wandb.Table(data=best_data, columns=["predicted", "true"])
            wandb.log({"scatter_plot": wandb.plot.scatter(best_table, "predicted", "true",
                                                          title="Label vs. Predictions")})

            best_model = deepcopy(model)
        elif args.lr_scheduler == 'exponential' and patience is not None:
            patience -= 1

        logger.info(
            f'Epochs: {epoch + 1} | Total-Train-Loss: {total_loss_train / len(train_dataloader) : .3f}'
            f' | Total-Val-Loss: {val_result["total_val_loss"] / len(val_dataloader) : .3f} | Patience: {patience} ')

        wandb.log(wandb_log, commit=True)

        stop_training = True

        for param_group in optimizer.param_groups:
            stop_training = stop_training and (
                param_group['lr'] < args.stop_at_learning_rate)

        if args.lr_scheduler == 'exponential' and patience is not None and patience < 0:
            stop_training = True

        if stop_training:
            if use_wandb:
                run.summary[f"{dataset2optimize}_val_loss"] = best_loss
                run.summary[f"{dataset2optimize}_val_corr"] = best_pearson_corr
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
    results["best_rmse"] = best_rmse

    return results, best_model


def finetune_frozen(model: AffinityGNN, train_dataset: Union[AffinityDataset, List[AffinityDataset]], val_dataset: Union[AffinityDataset, List[AffinityDataset]],
                      args: Namespace, lr_reduction: float = 1e-01) -> Tuple[Dict, AffinityGNN]:
    """ Utility to finetune the previously frozen model components (e.g. published pretrained model or dataset-specific layers)

    Args:
        model: Model with a pretrained encoder
        train_dataset: Dataset used for finetuning
        val_dataset: Validation dataset
        args: CLI Arguments
        lr_reduction: Value the learning rate gets multiplied by

    Returns:
        Tuple: Finetuned model, results as dict
    """

    # lower learning rate for pretrained model finetuning
    args.learning_rate = args.learning_rate * lr_reduction
    args.stop_at_learning_rate = args.stop_at_learning_rate * lr_reduction

    logger.info(f"Fintuning pretrained model with lr={args.learning_rate}")
    model.unfreeze()

    if args.train_strategy == "bucket_train":
        results, model = bucket_learning(model, train_dataset, val_dataset, args)
    else:
        results, model = train_loop(model, train_dataset, val_dataset, args)

    logger.info("Fintuning pretrained model completed")
    logger.debug(results)

    return results, model


def log_gradients(model: AffinityGNN):
    from torch_geometric.nn import GATv2Conv, GCNConv
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
                   device: torch.device = torch.device("cpu"), plot_path: str = None) -> Tuple[float, float, pd.DataFrame]:
    # TODO Why do we have different validations at different places (train_epoch, bucket_learning, ...)
    total_loss_val = 0
    all_predictions = np.array([])
    all_labels = np.array([])
    all_pdbs = []
    model.eval()
    for data in tqdm(dataloader, disable=not tqdm_output):
        output, label = forward_step(model, data, device)

        loss = get_loss(data["loss_criterion"], label, output)

        total_loss_val += loss.item()

        all_predictions = np.append(all_predictions, output["x"].flatten().detach().cpu().numpy())

        all_labels = np.append(all_labels, label["x"].detach().cpu().numpy())
        all_pdbs.extend([ filepath.split("/")[-1].split(".")[0] for filepath in data["input"]["filepath"]])
        # if len(all_labels) > 2:
            # break

    val_loss = total_loss_val / (len(all_predictions))
    try:
        pearson_corr = stats.pearsonr(all_labels, all_predictions)[0]
    except ValueError:
        logging.warning(f"nan in predictions or labels:\n{all_labels}\n{all_predictions}")
        pearson_corr = None

    # TODO pull out the plotting too
    if plot_path is not None:
        # scale prediction back to original values
        if args.scale_values:
            all_labels = all_labels * (args.scale_max - args.scale_min) + args.scale_min
            all_predictions = all_predictions * (args.scale_max - args.scale_min) + args.scale_min

        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plot_correlation(x=all_labels, y=all_predictions,
                         path=plot_path)

    res_df = pd.DataFrame({
                "pdb": all_pdbs,
                "prediction": all_predictions,
                "labels": all_labels
            })
    return pearson_corr, val_loss, res_df


def get_benchmark_score(model: AffinityGNN, args: Namespace, tqdm_output: bool = True, plot_path: str = None) -> Tuple[float, float, pd.DataFrame]:

    dataset = AffinityDataset(args.config, args.relaxed_pdbs, "AntibodyBenchmark", "L2",
                              node_type=args.node_type,
                              max_nodes=args.max_num_nodes,
                              interface_distance_cutoff=args.interface_distance_cutoff,
                              interface_hull_size=args.interface_hull_size,
                              max_edge_distance=args.max_edge_distance,
                              pretrained_model=args.pretrained_model,
                              scale_values=args.scale_values,
                              scale_min=args.scale_min,
                              scale_max=args.scale_max,
                              relative_data=False,
                              save_graphs=args.save_graphs,
                              force_recomputation=args.force_recomputation,
                              preprocess_data=args.preprocess_graph,
                              num_threads=args.num_workers,
                              load_embeddings=args.embeddings_path,
                              )

    dataloader = DL_torch(dataset, num_workers=args.num_workers, batch_size=1,
                              collate_fn=AffinityDataset.collate)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return evaluate_model(model, dataloader, args=args, tqdm_output=tqdm_output, device=device,
                          plot_path=plot_path)


def get_abag_test_score(model: AffinityGNN, args: Namespace, tqdm_output: bool = True, plot_path: Optional[str] = None,
                        validation_set: Optional[int] = None) -> Tuple[float, float, pd.DataFrame]:


    summary_path, _ = get_data_paths(args.config, "abag_affinity")
    summary_df = pd.read_csv(summary_path, index_col=0)
    if validation_set is None:
        summary_df = summary_df[summary_df["test"]]
    else:
        summary_df = summary_df[summary_df["validation"] == validation_set]

    test_pdbs_ids = summary_df.index.tolist()

    dataset = AffinityDataset(args.config, args.relaxed_pdbs, "abag_affinity", "L2", test_pdbs_ids,
                              node_type=args.node_type,
                              max_nodes=args.max_num_nodes,
                              interface_distance_cutoff=args.interface_distance_cutoff,
                              interface_hull_size=args.interface_hull_size,
                              max_edge_distance=args.max_edge_distance,
                              pretrained_model=args.pretrained_model,
                              scale_values=args.scale_values,
                              scale_min=args.scale_min,
                              scale_max=args.scale_max,
                              relative_data=False,
                              save_graphs=args.save_graphs,
                              force_recomputation=args.force_recomputation,
                              preprocess_data=args.preprocess_graph,
                              num_threads=args.num_workers,
                              load_embeddings=args.embeddings_path,
                              )

    dataloader = DL_torch(dataset, num_workers=args.num_workers, batch_size=1,
                              collate_fn=AffinityDataset.collate)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return evaluate_model(model, dataloader, args=args, tqdm_output=tqdm_output, device=device, plot_path=plot_path)


def get_skempi_corr(model: AffinityGNN, args: Namespace, tqdm_output: bool = True, plot_path: str = None) -> Tuple[float, float, float, pd.DataFrame]:
    """
    Take the available Skempi mutations for validation
    """

    dataset = AffinityDataset(args.config, args.relaxed_pdbs, "SKEMPI.v2", "L2",
                              node_type=args.node_type,
                              max_nodes=args.max_num_nodes,
                              interface_distance_cutoff=args.interface_distance_cutoff,
                              interface_hull_size=args.interface_hull_size,
                              max_edge_distance=args.max_edge_distance,
                              pretrained_model=args.pretrained_model,
                              scale_values=args.scale_values,
                              scale_min=args.scale_min,
                              scale_max=args.scale_max,
                              relative_data=False,
                              save_graphs=args.save_graphs,
                              force_recomputation=args.force_recomputation,
                              preprocess_data=args.preprocess_graph,
                              num_threads=args.num_workers,
                              load_embeddings=args.embeddings_path,
                              )

    dataloader = DL_torch(dataset, num_workers=args.num_workers, batch_size=1,
                            collate_fn=AffinityDataset.collate)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pearson_corr, val_loss, res_df = evaluate_model(model, dataloader, args=args, tqdm_output=tqdm_output, device=device,
                        plot_path=plot_path)

    # take everything after dash (-)
    res_df["mutation"] = res_df["pdb"].apply(lambda v: v.split("-")[1])
    res_df["pdb"] = res_df["pdb"].apply(lambda v: v.split("-")[0])
    # split results by PDBs and compute separate correlations
    grouped_correlations = res_df.groupby("pdb").apply(lambda group: stats.pearsonr(group.labels, group.prediction)[0])

    res_df["grouped_correlations"] = res_df["pdb"].apply(grouped_correlations.get)

    return np.mean(grouped_correlations), pearson_corr, val_loss, res_df

    # results.append(res)

    # return np.mean([v[0] for v in results]), np.mean([v[1] for v in results])
