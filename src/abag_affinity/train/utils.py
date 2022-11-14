import glob
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
import logging
import torch
from scipy import stats
from torch import nn
from torch.optim import Adam
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader as DL_torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from scipy.stats.mstats import gmean

from ..dataset import AffinityDataset
from ..model import AffinityGNN, TwinWrapper
from ..train.wandb_config import configure
from ..utils.config import get_data_paths
from ..utils.visualize import plot_correlation

logger = logging.getLogger(__name__)  # setup module logger


def forward_step(model: AffinityGNN, data: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        output = twin_model(data)

    else:
        data["input"]["graph"] = data["input"]["graph"].to(device)
        if "deeprefine_graph" in data["input"]:
            data["input"]["deeprefine_graph"] = data["input"]["deeprefine_graph"].to(device)
        output = model(data["input"])
        output["relative"] = data["relative"]
        output["affinity_type"] = data["affinity_type"]

    label = get_label(data, device)

    return output, label


def get_label(data: Dict, device: torch.device) -> torch.Tensor:
    if data["affinity_type"] == "-log(Kd)":
        if data["relative"]:
            return (data["input"][0]["graph"].y - data["input"][1]["graph"].y).to(device)
        else:
            return data["input"]["graph"].y.to(device)
    elif data["affinity_type"] == "E":
        assert data["relative"], "Enrichment values can only be used relative"

        label_1 = (data["input"][0]["graph"].y > data["input"][1]["graph"].y).float()
        label_2 = (data["input"][1]["graph"].y > data["input"][0]["graph"].y).float()
        label = torch.stack((label_1, label_2)).T
        return label.to(device)
    else:
        raise ValueError(f"Wrong affinity type given - expected one of (-log(Kd), E) but got {data['affinity_type']}")


def get_loss(criterion, label: torch.Tensor, output: torch.Tensor, min: float, max: float) -> torch.Tensor:
    if output["affinity_type"] == "-log(Kd)":
        return criterion(output["x"], label)
    elif output["affinity_type"] == "E":
        loss = torch.nn.functional.cross_entropy(output["x_prob"], label)

        all_preds = torch.cat((output["x1"], output["x2"]))

        # add loss for too large values
        over_max = (all_preds > max).float()
        loss += torch.sum((all_preds - max)**2 * over_max)

        # add loss for too small values
        below_min = (all_preds < min).float()
        loss += torch.sum((min - all_preds)**2 * below_min)

        return loss
    else:
        raise ValueError(f"Wrong affinity type given - expected one of (-log(Kd), E) but got {output['affinity_type']}")


def train_epoch(model: AffinityGNN, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion,
                optimizer: torch.optim, device: torch.device = torch.device("cpu"), tqdm_output: bool = True) -> Tuple[
    AffinityGNN, Dict]:
    """ Train model for one epoch given the train and validation dataloaders

    Args:
        model: torch module used to train
        train_dataloader: Dataloader with training examples
        val_dataloader: Dataloader with validation examples
        criterion: Function used to calculate loss
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

        loss = get_loss(criterion, label, output, 4, 14) #TODO: define min and max from args
        total_loss_train += loss.item()

        loss.backward()
        optimizer.step()

    total_loss_val = 0
    all_predictions = np.array([])
    all_labels = np.array([])

    model.eval()
    for data in tqdm(val_dataloader, disable=not tqdm_output):
        output, label = forward_step(model, data, device)

        loss = get_loss(criterion, label, output, 4, 14) #TODO: define min and max from args
        total_loss_val += loss.item()

        all_predictions = np.append(all_predictions, output["x"].flatten().detach().cpu().numpy())
        if data["relative"] and data["affinity_type"] == "E":
            torch.argmax(label.detach().cpu(), dim=1)
        else:
            label = label.detach().cpu()
        all_labels = np.append(all_labels, label.numpy())

    val_loss = total_loss_val / (len(all_predictions))
    pearson_corr = stats.pearsonr(all_labels, all_predictions)

    results = {
        "val_loss": val_loss,
        "pearson_correlation": pearson_corr[0],
        "pearson_correlation_p": pearson_corr[1],
        "all_predictions": all_predictions,
        "all_labels": all_labels,
        "total_train_loss": total_loss_train,
        "total_val_loss": total_loss_val
    }

    return model, results


def train_loop(model: AffinityGNN, train_dataset: AffinityDataset, val_dataset: AffinityDataset, args: Namespace) -> \
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
    Path(plot_path).mkdir(exist_ok=True, parents=True)

    wandb, wdb_config, use_wandb, run = configure(args)

    results = {
        "epoch_loss": [],
        "epoch_corr": []
    }

    train_dataloader, val_dataloader = get_dataloader(args, train_dataset, val_dataset)
    if val_dataset.relative_data:
        data_type = "relative"
    else:
        data_type = "absolute"
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = get_loss_function(args, device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    best_loss = np.inf
    best_pearson_corr = -np.inf
    patience = args.patience

    best_model = deepcopy(model)

    for i in range(args.max_epochs):
        model, epoch_results = train_epoch(model, train_dataloader, val_dataloader, criterion, optimizer, device,
                                           args.tqdm_output)

        results["epoch_loss"].append(epoch_results["val_loss"])
        results["epoch_corr"].append(epoch_results["pearson_correlation"])

        if epoch_results["val_loss"] < best_loss: # or epoch_results["pearson_correlation"] > best_pearson_corr:
            patience = args.patience
            best_loss = min(epoch_results["val_loss"], best_loss)
            best_pearson_corr = max(epoch_results["pearson_correlation"], best_pearson_corr)
            plot_correlation(x=epoch_results["all_labels"], y=epoch_results["all_predictions"],
                             path=os.path.join(plot_path, f"{train_dataset.dataset_name}-{data_type}.png"))
            best_model = deepcopy(model)
        else:
            patience -= 1

        logger.info(
            f'Epochs: {i + 1} | Train-Loss: {epoch_results["total_train_loss"] / len(train_dataloader) : .3f}'
            f'  | Val-Loss: {epoch_results["total_val_loss"] / len(val_dataloader) : .3f} | Val-r: {epoch_results["pearson_correlation"]: .4f} | p-value r=0: {epoch_results["pearson_correlation_p"]: .4f} | Patience: {patience} ')

        if np.isnan(epoch_results["pearson_correlation"]):
            preds_nan = any(np.isnan(epoch_results["all_predictions"]))
            preds_same = np.all(epoch_results["all_predictions"] == epoch_results["all_predictions"][0])
            labels_nan = any(np.isnan(epoch_results["all_labels"]))
            labels_same = np.all(epoch_results["all_labels"] == epoch_results["all_labels"][0])

            logger.error(
                f"Pearon correlation is NaN. Preds NaN:{preds_nan}. Preds Same {preds_same}. Labels NaN: {labels_nan}. Labels Same {labels_same}")
            return results, best_model

        wandb_log = {
            "val_loss": epoch_results["total_val_loss"] / (len(val_dataset) / args.batch_size),
            "train_loss": epoch_results["total_train_loss"] / (len(train_dataset) / args.batch_size),
            "pearson_correlation": epoch_results["pearson_correlation"],
            f"{val_dataset.dataset_name}:{data_type}_val_loss": epoch_results["total_val_loss"] / (
                    len(val_dataset) / args.batch_size),
            f"{val_dataset.dataset_name}:{data_type}_val_corr": epoch_results["pearson_correlation"]
        }

        wandb.log(wandb_log, commit=True)

        if patience < 0:
            if use_wandb:
                run.summary["best_pearson"] = best_pearson_corr
                run.summary["best_loss"] = best_loss
            break

        model.train()

        # only force recomputation in first epoch and then use precomputed graphs
        if args.save_graphs:
            train_dataset.force_recomputation = False
            val_dataset.force_recomputation = False

    results["best_loss"] = best_loss
    results["best_correlation"] = best_pearson_corr

    return results, best_model


def get_loss_function(args: Namespace, device: torch.device):
    if args.loss_function == "L1":
        loss_fn = nn.L1Loss().to(device)
    elif args.loss_function == "L2":
        loss_fn = nn.MSELoss().to(device)
    else:
        raise ValueError("Loss_Function must either be 'L1' or 'L2'")
    return loss_fn


def load_model(num_node_features: int, num_edge_features: int, args: Namespace,
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
    model = AffinityGNN(num_node_features, num_edge_features,
                        num_nodes=args.max_num_nodes,
                        pretrained_model=args.pretrained_model, pretrained_model_path=pretrained_model_path,
                        gnn_type=args.gnn_type,
                        layer_type=args.layer_type, num_gnn_layers=args.num_gnn_layers, size_halving=args.size_halving,
                        node_type=args.node_type,
                        aggregation_method=args.aggregation_method,
                        nonlinearity=args.nonlinearity,
                        num_fc_layers=args.num_fc_layers,
                        device=device)

    return model


def get_dataloader(args: Namespace, train_dataset: AffinityDataset, val_dataset: AffinityDataset) -> Tuple[
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

    train_dataloader = DL_torch(train_dataset, num_workers=args.num_workers, batch_size=batch_size,
                                shuffle=args.shuffle, collate_fn=AffinityDataset.collate)
    val_dataloader = DL_torch(val_dataset, num_workers=args.num_workers, batch_size=batch_size,
                              collate_fn=AffinityDataset.collate)

    return train_dataloader, val_dataloader


def train_val_split(config: Dict, dataset_name: str, validation_set: int) -> Tuple[List, List]:
    """ Split data in a train and a validation subset

    For Dataset_v1 use the predefined split given in the csv, otherwise use random split

    Args:
        config: Dict with configuration info
        dataset_name: Name of the dataset
        validation_set: Integer identifier of the validation split (1,2,3)

    Returns:
        Tuple: List with indices for train and validation set
    """
    if dataset_name == "Dataset_v1":  # use predefined split
        summary_file, pdb_path = get_data_paths(config, dataset_name)
        dataset_summary = pd.read_csv(summary_file, index_col=0)
        val_ids = list(dataset_summary[dataset_summary["validation"] == 1].index)
        train_ids = list(
            dataset_summary[
                (dataset_summary["validation"] != validation_set) & (dataset_summary["test"] == False)].index)

    elif "-" in dataset_name:
        # DMS data
        dataset_name, publication = dataset_name.split("-")
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path, index_col=0)
        summary_df = summary_df[summary_df["publication"] == publication]
        all_data_points = summary_df.index.values
        random.shuffle(all_data_points)
        cutoff = int(len(all_data_points) * 0.9)
        train_ids = all_data_points[:cutoff]
        val_ids = all_data_points[cutoff:]
    else:
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path, index_col=0)

        # TODO: Integrate multiple (corresponding to dataset_v1 split) train/val splits for other domains
        val_pdbs = summary_df.loc[summary_df["validation"] == 1, "pdb"]
        train_pdbs = summary_df.loc[summary_df["validation"] != 1, "pdb"]

        if "mutated_pdb_path" in config["DATA"][dataset_name]:  # only use files that were generated
            data_path = os.path.join(config["DATA"]["path"],
                                     config["DATA"][dataset_name]["folder_path"],
                                     config["DATA"][dataset_name]["mutated_pdb_path"])
            all_files = glob.glob(data_path + "/*/*") + glob.glob(data_path + "/*")
            available_files = set(
                file_path.split("/")[-2].lower() + "-" + file_path.split("/")[-1].split(".")[0].lower() for file_path in
                all_files if file_path.split(".")[-1] == "pdb")
            summary_df = summary_df[
                (summary_df["data_location"] == "RESOURCES") | summary_df.index.isin(available_files)]

        summary_df = summary_df[summary_df["-log(Kd)"].notnull()]

        val_ids = summary_df[summary_df["pdb"].isin(val_pdbs)].index.tolist()
        train_ids = summary_df[summary_df["pdb"].isin(train_pdbs)].index.tolist()

    return train_ids, val_ids


def load_datasets(config: Dict, dataset: str, validation_set: int, args: Namespace) -> Tuple[
    AffinityDataset, AffinityDataset]:
    """ Get train and validation datasets for a specific dataset and data type

    1. Get train and validation splits
    2. Load the dataset in the specified data type class

    Args:
        config: training configuration as dict
        dataset: Name of the dataset:Usage of data (absolute, relative) - eg. SKEMPI.v2:relative
        data_type: Type of the dataset
        validation_set: Integer identifier of the validation split (1,2,3)
        args: CLI arguments

    Returns:
        Tuple: Train and validation dataset
    """

    dataset_name, data_usage = dataset.split(":")

    if data_usage == "relative":
        relative_data = True
    else:
        relative_data = False

    train_ids, val_ids = train_val_split(config, dataset_name, validation_set)

    if args.test:
        train_ids = train_ids[:20]
        val_ids = val_ids[:5]

    logger.debug(f"Get dataLoader for {dataset_name}:{data_usage}")
    train_data = AffinityDataset(config, dataset_name, train_ids,
                                 node_type=args.node_type,
                                 max_nodes=args.max_num_nodes,
                                 interface_distance_cutoff=args.interface_distance_cutoff,
                                 interface_hull_size=args.interface_hull_size,
                                 max_edge_distance=args.max_edge_distance,
                                 pretrained_model=args.pretrained_model,
                                 scale_values=args.scale_values,
                                 relative_data=relative_data,
                                 save_graphs=args.save_graphs,
                                 force_recomputation=args.force_recomputation,
                                 preprocess_data=args.preprocess_graph,
                                 num_threads=args.num_workers
                                 )
    val_data = AffinityDataset(config, dataset_name, val_ids,
                               node_type=args.node_type,
                               max_nodes=args.max_num_nodes,
                               interface_distance_cutoff=args.interface_distance_cutoff,
                               interface_hull_size=args.interface_hull_size,
                               max_edge_distance=args.max_edge_distance,
                               pretrained_model=args.pretrained_model,
                               scale_values=args.scale_values,
                               relative_data=relative_data,
                               save_graphs=args.save_graphs,
                               force_recomputation=args.force_recomputation,
                               preprocess_data=args.preprocess_graph,
                               num_threads=args.num_workers
                               )

    return train_data, val_data


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

    if args.bucket_size_mode == "min":
        train_bucket_size = min([len(dataset) for dataset in train_datasets]) * len(train_datasets)
    elif args.bucket_size_mode == "geometric_mean":
        data_sizes = [len(dataset) for dataset in train_datasets]
        train_bucket_size = gmean(data_sizes) * len(train_datasets)
    elif args.bucket_size_mode == "double_geometric_mean":
        data_sizes = [len(dataset) for dataset in train_datasets]
        geometric_mean = gmean(data_sizes)
        train_bucket_size = [ gmean([geometric_mean, size]) for size in data_sizes]
    else:
        raise ValueError(f"bucket_size_mode argument not supported: Got {args.bucket_size_mode} "
                         f"but expected one of [min, geometric_mean]")
    i = 0
    for idx, train_dataset in enumerate(train_datasets):
        if len(train_dataset) >= train_bucket_size[idx]:
            # sample without replacement
            indices = random.sample(range(len(train_dataset)), train_bucket_size[idx])
        else:
            # sample with replacement
            indices = random.choices(range(len(train_dataset)), k=train_bucket_size[idx])

        train_buckets.append(Subset(train_dataset, indices))
        if train_dataset.relative_data:
            relative_data_indices.extend(list(range(i, i + len(indices))))
        else:
            absolute_data_indices.extend(list(range(i, i + len(indices))))
        i += len(indices)

    # shorten relative data for validation
    # TODO: find better way
    for dataset in val_datasets:
        if dataset.relative_data:
            dataset.data_points = dataset.data_points[:100]

    train_dataset = ConcatDataset(train_buckets)
    batch_sampler = generate_bucket_batch_sampler([absolute_data_indices, relative_data_indices], args.batch_size,
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

    dataset2optimize = "Dataset_v1:absolute"

    wandb, wdb_config, use_wandb, run = configure(args)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = get_loss_function(args, device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # initialize training values
    best_loss = 1000
    best_pearson_corr = 0
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
        train_dataloader, val_dataloader = get_bucket_dataloader(args, train_datasets, val_datasets, args.bucket_size_mode)

        model, epoch_results = train_epoch(model, train_dataloader, val_dataloader, criterion, optimizer, device,
                                           args.tqdm_output)

        dataset_results = {}

        wandb_log = {
            "val_loss": epoch_results["total_val_loss"] / len(val_dataloader),
            "train_loss": epoch_results["total_train_loss"] / len(train_dataloader),
            "pearson_correlation": epoch_results["pearson_correlation"]
        }

        i = 0
        for val_dataset in val_datasets:
            preds = epoch_results["all_predictions"][i:i + len(val_dataset)]
            labels = epoch_results["all_labels"][i:i + len(val_dataset)]
            val_loss = criterion(torch.from_numpy(preds).to(device),
                                 torch.from_numpy(labels).to(device))  # mean_squared_error(labels, preds) / len(labels)
            pearson_corr = stats.pearsonr(labels, preds)
            if val_dataset.relative_data:
                data_type = "relative"
            else:
                data_type = "absolute"
            logger.info(
                f'Epochs: {epoch + 1}  | Dataset: {val_dataset.dataset_name}:{data_type} | Val-Loss: {val_loss: .3f} '
                f'  | Val-r: {pearson_corr[0]: .4f} | p-value r=0: {pearson_corr[1]: .4f} ')

            if np.isnan(pearson_corr[0]):
                preds_nan = any(np.isnan(preds))
                labels_nan = any(np.isnan(labels))
                logger.error(f"Pearson correlation is NaN. Preds NaN:{preds_nan}. Labels NaN: {labels_nan}")
                return results, best_model
            dataset_results[val_dataset.dataset_name + ":" + data_type] = {
                "val_loss": val_loss,
                "pearson_correlation": pearson_corr[0],
                "pearson_correlation_p": pearson_corr[1],
                "all_labels": labels,
                "all_predictions": preds
            }
            wandb_log[f"{val_dataset.dataset_name}:{data_type}_val_loss"] = val_loss
            wandb_log[f"{val_dataset.dataset_name}:{data_type}_val_corr"] = pearson_corr[0]
            i += len(val_dataset)

        results["epoch_loss"].append(epoch_results["val_loss"])
        results["epoch_corr"].append(epoch_results["pearson_correlation"])

        if dataset2optimize in dataset_results:
            results["abag_epoch_loss"].append(dataset_results[dataset2optimize]["val_loss"])
            results["abag_epoch_corr"].append(dataset_results[dataset2optimize]["pearson_correlation"])

        if dataset_results[dataset2optimize]["val_loss"] < best_loss:# or dataset_results[dataset2optimize]["pearson_correlation"] > best_pearson_corr:
            patience = args.patience
            best_loss = min(dataset_results[dataset2optimize]["val_loss"], best_loss)
            best_pearson_corr = max(dataset_results[dataset2optimize]["pearson_correlation"], best_pearson_corr)
            for val_dataset in val_datasets:  # correlation plot for each dataset
                if val_dataset.relative_data:
                    data_type = "relative"
                else:
                    data_type = "absolute"
                dataset_name = val_dataset.dataset_name + ":" + data_type

                plot_correlation(x=dataset_results[dataset_name]["all_labels"],
                                 y=dataset_results[dataset_name]["all_predictions"],
                                 path=os.path.join(plot_path, f"{val_dataset.dataset_name}-{data_type}.png"))

            plot_correlation(x=epoch_results["all_labels"], y=epoch_results["all_predictions"],
                             path=os.path.join(plot_path, "all_bucket_predictions.png"))

            best_model = deepcopy(model)
        else:
            patience -= 1

        logger.info(
            f'Epochs: {epoch + 1} | Train-Loss: {epoch_results["total_train_loss"] / len(train_dataloader) : .3f}'
            f'  | Val-Loss: {epoch_results["total_val_loss"] / len(val_dataloader) : .3f} | Val-r: {epoch_results["pearson_correlation"]: .4f} | p-value r=0: {epoch_results["pearson_correlation_p"]: .4f} | Patience: {patience} ')

        wandb.log(wandb_log, commit=True)

        if patience < 0:
            if use_wandb:
                run.summary["best_pearson"] = best_pearson_corr
                run.summary["best_loss"] = best_loss
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

    return results, best_model


def finetune_pretrained(model: AffinityGNN, train_dataset: Union[AffinityDataset, List[AffinityDataset]], val_dataset: Union[AffinityDataset, List[AffinityDataset]],
                      args: Namespace, lr_reduction: float = 1e-01) -> Tuple[Dict, AffinityGNN]:
    """ Utility to finetune the pretrained model using a lowered learning rate

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

    logger.info(f"Fintuning pretrained model with lr={args.learning_rate}")

    # make pretrained model trainable
    model.pretrained_model.requires_grad = True
    if hasattr(model.pretrained_model, 'deep_refine'):
        model.pretrained_model.deep_refine.unfreeze()

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