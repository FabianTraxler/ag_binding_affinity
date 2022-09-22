import glob
import torch
from torch import nn
from torch_geometric.data import DataLoader, Data, HeteroData
from torch.utils.data import DataLoader as DL_torch
from torch.utils.data import Subset, ConcatDataset
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, Tuple, List, Union
import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
from scipy import stats
from argparse import Namespace
from pathlib import Path
import logging

from abag_affinity.dataset import AffinityDataset, BoundComplexGraphs, DDGBackboneInputs, HeteroGraphs, DeepRefineBackboneInputs
from abag_affinity.model import GraphConv, GraphConvAttention, FSGraphConv, DDGBackbone, \
    GraphConvAttentionModelWithBackbone, ResidueKpGNN, TwinWrapper, ModelWithBackbone, DeepRefineBackbone
from abag_affinity.utils.config import get_data_paths
from abag_affinity.train.wandb_config import configure
from abag_affinity.utils.visualize import plot_correlation


logger = logging.getLogger(__name__) # setup module logger


def forward_step(model: nn.Module, data: Union[List, Data, HeteroData, Dict], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if not isinstance(data, (Data, HeteroData, Dict)) and len(data) == 2:  # relative data available
        twin_model = TwinWrapper(model)
        twin_model.train()
        if isinstance(data[0], dict):  # DeepRefine Dataloader
            data[0]["graph"] = data[0]["graph"].to(device)
            data[1]["graph"] = data[1]["graph"].to(device)
            output = twin_model(data[0], data[1]).flatten()
            label = torch.from_numpy(data[0]["affinity"] - data[1]["affinity"])
        else:
            output = twin_model(data[0].to(device), data[1].to(device)).flatten()
            label = data[0].y - data[1].y
    elif isinstance(data, dict):  # DeepRefine Dataloader
        label = torch.from_numpy(data["affinity"])
        data["graph"] = data["graph"].to(device)
        output = model(data)
    elif isinstance(data, (Data, HeteroData)):
        output = model(data.to(device)).flatten()
        label = data.y
    else:
        raise ValueError(f"Wrong data type for forward step. Expected one of (Data, HeteroData, List or Dict) "
                         f"but got {type(data)}")

    return output, label


def train_epoch(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion,
                optimizer: torch.optim, device: torch.device = torch.device("cpu"), ) -> Tuple[nn.Module, Dict]:
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

    for data in tqdm(train_dataloader):
        optimizer.zero_grad()

        output, label = forward_step(model, data, device)

        loss = criterion(output, label.to(device))
        total_loss_train += loss.item()

        loss.backward()
        optimizer.step()

    total_loss_val = 0
    all_predictions = np.array([])
    all_labels = np.array([])

    model.eval()
    for data in tqdm(val_dataloader):
        output, label = forward_step(model, data, device)

        loss = criterion(output, label.to(device))
        total_loss_val += loss.item()

        all_predictions = np.append(all_predictions, output.flatten().detach().cpu().numpy())
        all_labels = np.append(all_labels, label.flatten().detach().cpu().numpy())


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


def train_loop(model: nn.Module, train_dataset: AffinityDataset, val_dataset: AffinityDataset, args: Namespace) -> Tuple[Dict, nn.Module]:
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
    plot_path = os.path.join(args.config["plot_path"], f"{args.node_type}/sequential_learning/val_set_{args.validation_set}")
    Path(plot_path).mkdir(exist_ok=True, parents=True)

    wandb, wdb_config, use_wandb, run, this_run = configure(args)

    results = {
        "epoch_loss": [],
        "epoch_corr": []
    }

    train_dataloader, val_dataloader = get_dataloader(args, train_dataset, val_dataset, wdb_config)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion1 = nn.L1Loss().to(device) #nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr= wdb_config.learning_rate)

    best_loss = np.inf
    best_pearson_corr = -np.inf
    patience = wdb_config.patience

    best_model = deepcopy(model)

    for i in range(wdb_config.max_epochs):
        model, epoch_results = train_epoch(model, train_dataloader, val_dataloader, criterion1, optimizer, device)

        results["epoch_loss"].append(epoch_results["val_loss"])
        results["epoch_corr"].append(epoch_results["pearson_correlation"])

        if epoch_results["val_loss"] < best_loss or epoch_results["pearson_correlation"] > best_pearson_corr:
            patience = wdb_config.patience
            best_loss = min(epoch_results["val_loss"], best_loss)
            best_pearson_corr = max(epoch_results["pearson_correlation"], best_pearson_corr)
            plot_correlation(x=epoch_results["all_labels"], y=epoch_results["all_predictions"],
                             path=os.path.join(plot_path, f"{train_dataset.dataset_name}.png"))
            best_model = deepcopy(model)
        else:
            patience -= 1

        logger.info(
            f'Epochs: {i + 1} | Train-Loss: {epoch_results["total_train_loss"] / len(train_dataloader) : .3f}'
            f'  | Val-Loss: {epoch_results["total_val_loss"] / len(val_dataloader) : .3f} | Val-r: {epoch_results["pearson_correlation"]: .4f} | p-value r=0: {epoch_results["pearson_correlation_p"]: .4f} | Patience: {patience} ')

        wandb_log = {
            "val_loss": epoch_results["total_val_loss"] / (len(val_dataset) / wdb_config.batch_size),
            "train_loss":epoch_results["total_train_loss"] / (len(train_dataset) / wdb_config.batch_size),
            "pearson_correlation": epoch_results["pearson_correlation"],
            f"{val_dataset.dataset_name}_val_loss": epoch_results["total_val_loss"] / (len(val_dataset) / wdb_config.batch_size),
            f"{val_dataset.dataset_name}_val_corr": epoch_results["pearson_correlation"]
        }

        wandb.log(wandb_log)

        if patience < 0:
            if use_wandb:
                run.summary["best_pearson"] = best_pearson_corr
                run.summary["best_loss"] = best_loss
            break

        model.train()

    results["best_loss"] = best_loss
    results["best_correlation"] = best_pearson_corr

    return results, best_model


def load_model(model_type: str, num_node_features: int, num_edge_features: int, args: Namespace, device: torch.device = torch.device("cpu")) -> nn.Module:
    """ Load a specific model type and initialize it randomly

    Args:
        model_type: Name of the model to be loaded
        num_features: Dimension of features of inputs to the model
        args: CLI arguments
        device: Device the model will be loaded on

    Returns:
        nn.Module: model on specified device
    """
    if model_type == "GraphConv":
        model = GraphConv(num_node_features, num_edge_features).to(device)
    elif model_type == "GraphAttention":
        model = GraphConvAttention(num_node_features, num_edge_features).to(device)
    elif model_type == "FixedSizeGraphConv":
        model = FSGraphConv(num_node_features, num_edge_features, args.max_num_nodes).to(device)
    elif model_type == "DDGBackboneFC":
        encoder = DDGBackbone("./binding_ddg_predictor/data/model.pt", device)
        model = GraphConvAttentionModelWithBackbone(encoder, num_nodes=args.max_num_nodes, device=device)
    elif model_type == "KpGNN":
        if args.node_type == "residue":
            model = ResidueKpGNN(num_node_features, device)
        elif args.node_type == "atom":
            # TODO: Implement Model
            pass
    elif model_type == "DeepRefineBackbone":
        encoder = DeepRefineBackbone(args)
        model = GraphConvAttentionModelWithBackbone(encoder, device=device)

    else:
        raise ValueError("Please specify valid model (GraphConv, GraphAttention, FixedSizeGraphConv, DDGBackboneFC, KpGNN)")

    return model


def get_dataloader(args: Namespace, train_dataset: AffinityDataset, val_dataset: AffinityDataset, wdb_config) -> Tuple[DataLoader, DataLoader]:
    """ Get dataloader for train and validation dataset

    Use the DGL Dataloader for the DeepRefine Inputs

    Args:
        args: CLI arguments
        train_dataset: Torch dataset for train examples
        val_dataset: Torch dataset for validation examples
        wdb_config: training configuration as weight&bias config object

    Returns:
        Tuple: Train dataloader, validation dataloader
    """
    if args.data_type == "DeepRefineInputs": # use DGL Dataloader for DeepRefine Inputs
        train_dataloader = DL_torch(train_dataset, num_workers=wdb_config.num_workers, batch_size=wdb_config.batch_size,
                                      shuffle=True, collate_fn=DeepRefineBackboneInputs.collate)
        val_dataloader = DL_torch(val_dataset, num_workers=wdb_config.num_workers, batch_size=wdb_config.batch_size,
                                  collate_fn=DeepRefineBackboneInputs.collate)
    else:
        train_dataloader = DataLoader(train_dataset, num_workers=wdb_config.num_workers, batch_size=wdb_config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, num_workers=wdb_config.num_workers, batch_size=wdb_config.batch_size)

    return train_dataloader, val_dataloader


def train_val_split(config: Dict, dataset_name: str, validation_set: int) -> Tuple[List, List]:
    """ Split data in a train a validation subset

    For Dataset_v1 use the predefined split given in the csv, otherwise use random split

    Args:
        config: Dict with configuration info
        dataset_name: Name of the dataset
        validation_set: Integer identifier of the validation split (1,2,3)

    Returns:
        Tuple: List with indices for train and validation set
    """
    if dataset_name == "Dataset_v1": # use predefined split
        summary_file, pdb_path = get_data_paths(config, dataset_name)
        dataset_summary = pd.read_csv(summary_file, index_col=0)
        val_ids = list(dataset_summary[dataset_summary["validation"] == 1].index)
        train_ids = list(
            dataset_summary[(dataset_summary["validation"] != validation_set) & (dataset_summary["test"] == False)].index)
    else:
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path, index_col=0)
        pdb_ids = summary_df["pdb"].unique().tolist()

        # random split
        random.shuffle(pdb_ids)
        split = int(len(pdb_ids) * 0.8)
        val_pdbs = pdb_ids[split:]
        train_pdbs = pdb_ids[:split]

        if "mutated_pdb_path" in config["DATA"][dataset_name]: # only use files that were generated
            data_path =  os.path.join(config["DATA"]["path"],
                                        config["DATA"][dataset_name]["folder_path"],
                                        config["DATA"][dataset_name]["mutated_pdb_path"])
            all_files = glob.glob(data_path + "/*/*") + glob.glob(data_path + "/*")
            available_files = set( file_path.split("/")[-2].lower() + "-" + file_path.split("/")[-1].split(".")[0].lower() for file_path in all_files if file_path.split(".")[-1] == "pdb")
            summary_df = summary_df[(summary_df["data_location"] == "RESOURCES") | summary_df.index.isin(available_files)]

        summary_df = summary_df[summary_df["-log(Kd)"].notnull()]

        val_ids = summary_df[summary_df["pdb"].isin(val_pdbs)].index.tolist()
        train_ids = summary_df[summary_df["pdb"].isin(train_pdbs)].index.tolist()

    return train_ids, val_ids


def load_datasets(config: Dict, dataset_name: str, data_type: str, validation_set: int, args: Namespace) -> Tuple[AffinityDataset, AffinityDataset]:
    """ Get train and validation datasets for a specific dataset and data type

    1. Get train and validation splits
    2. Load the dataset in the specifies data type class

    Args:
        config: training configuration as dict
        dataset_name: Name of the dataset
        data_type: Type of the dataset
        validation_set: Integer identifier of the validation split (1,2,3)
        args: CLI arguments

    Returns:
        Tuple: Train and validation dataset
    """

    train_ids, val_ids = train_val_split(config, dataset_name, validation_set)

    if dataset_name in ["SKEMPI.v2"]:
        relative_data = True
    else:
        relative_data = False

    logger.debug(f"Get {data_type} dataLoader for {dataset_name}")

    if data_type == "BoundComplexGraphs":
        train_data = BoundComplexGraphs(config, dataset_name, train_ids, max_nodes=args.max_num_nodes,node_type=args.node_type,
                                  relative_data=relative_data, interface_hull_size=args.interface_hull_size,
                                  scale_values=args.scale_values, save_graphs=args.save_graphs, force_recomputation=args.force_recomputation)
        val_data = BoundComplexGraphs(config, dataset_name, val_ids, max_nodes=args.max_num_nodes,node_type=args.node_type,
                                  relative_data=relative_data, interface_hull_size=args.interface_hull_size,
                                  scale_values=args.scale_values, save_graphs=args.save_graphs, force_recomputation=args.force_recomputation)
    elif data_type == "DDGBackboneInputs":
        train_data = DDGBackboneInputs(config, dataset_name, train_ids,  args.max_num_nodes, relative_data=relative_data,
                                       scale_values=args.scale_values, save_graphs=args.save_graphs, force_recomputation=args.force_recomputation)
        val_data = DDGBackboneInputs(config, dataset_name, val_ids,  args.max_num_nodes, relative_data=relative_data,
                                     scale_values=args.scale_values, save_graphs=args.save_graphs, force_recomputation=args.force_recomputation)
    elif data_type == "HeteroGraphs":
        train_data = HeteroGraphs(config, dataset_name, train_ids, node_type=args.node_type, relative_data=relative_data,
                                  save_graphs=args.save_graphs, scale_values=args.scale_values, force_recomputation=args.force_recomputation)
        val_data = HeteroGraphs(config, dataset_name, val_ids, node_type=args.node_type, relative_data=relative_data,
                                save_graphs=args.save_graphs, scale_values=args.scale_values, force_recomputation=args.force_recomputation)
    elif data_type == "DeepRefineInputs":
        train_data = DeepRefineBackboneInputs(config, dataset_name, train_ids, relative_data=relative_data,
                                              num_threads=args.num_workers, preprocess_data=args.preprocess_graph,
                                              save_graphs=args.save_graphs, interface_hull_size=args.interface_hull_size
                                              , force_recomputation=args.force_recomputation)
        val_data = DeepRefineBackboneInputs(config, dataset_name, val_ids, relative_data=relative_data,
                                            num_threads=args.num_workers, preprocess_data=args.preprocess_graph,
                                            save_graphs=args.save_graphs, interface_hull_size=args.interface_hull_size
                                            , force_recomputation=args.force_recomputation)
    else:
        raise ValueError("Please specify dataset type (SimpleGraphs, FixedSizeGraphs, DDGBackboneInputs, HeteroGraphs, DeepRefineInputs)")

    return train_data, val_data


def get_bucket_dataloader(args: Namespace, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset],
                          train_bucket_size: int, config):
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
    i = 0
    for train_dataset in train_datasets:
        indices = random.sample(range(len(train_dataset)), train_bucket_size)
        train_buckets.append(Subset(train_dataset, indices))
        if train_dataset.relative_data:
            relative_data_indices.extend(list(range(i, i + len(indices))))
        else:
            absolute_data_indices.extend(list(range(i, i + len(indices))))
        i += len(indices)

    train_dataset = ConcatDataset(train_buckets)
    batch_sampler = generate_bucket_batch_sampler([absolute_data_indices, relative_data_indices], args.batch_size,
                                                  shuffle=True)

    if args.data_type == "DeepRefineInputs":
        train_dataloader = DL_torch(train_dataset, num_workers=config.num_workers,
                                    collate_fn=DeepRefineBackboneInputs.collate, batch_sampler=batch_sampler)
        val_dataloader = DL_torch(ConcatDataset(val_datasets), num_workers=config.num_workers, batch_size=1,
                                  collate_fn=DeepRefineBackboneInputs.collate)
    else:
        train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)
        val_dataloader = DataLoader(ConcatDataset(val_datasets), num_workers=args.num_workers, batch_size=1,
                                    shuffle=False)

    return train_dataloader, val_dataloader


def generate_bucket_batch_sampler(data_indices_list:List, batch_size: int, shuffle: bool = False) -> List[List[int]]:
    """ Generate batches for different data types only combining examples of the same type

    Args:
        data_indices_list: List of data indices list
        batch_size: Size of the batches
        shuffle: Boolean indicator if examples and batches are to be shuffled

    Returns:
        List: Data indices in batches
    """

    all_batches = []

    for data_indices in data_indices_list: # differnt batches for differnt data types
        if len(data_indices) == 0: continue # ignore empty indices
        batches = []

        if shuffle:
            random.shuffle(data_indices)

        i = 0
        while i < len(data_indices): # batch samples of same type
            batches.append(data_indices[i:i + batch_size])
            i += batch_size

        all_batches.extend(batches)

    if shuffle:
        random.shuffle(all_batches)

    return all_batches


def bucket_learning(model: torch.nn.Module, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset],
                    args: Namespace, bucket_size: int = 1000) -> Tuple[Dict, nn.Module]:
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
    plot_path = os.path.join(args.config["plot_path"], f"{args.node_type}/bucket_learning/val_set_{args.validation_set}")
    Path(plot_path).mkdir(exist_ok=True, parents=True)

    wandb, config, use_wandb, run, this_run = configure(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.L1Loss().to(device) #nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr= args.learning_rate)

    # initialize training values
    best_loss = 1000
    best_pearson_corr = 0
    patience = config.patience
    results = {
        "epoch_loss": [],
        "epoch_corr": [],
        "abag_epoch_loss": [],
        "abag_epoch_corr": []
    }

    best_model = deepcopy(model)

    train_bucket_size = min([len(dataset) for dataset in train_datasets] + [bucket_size])

    for epoch in range(args.max_epochs):
        # create new buckets for each epoch
        train_dataloader, val_dataloader = get_bucket_dataloader(args, train_datasets, val_datasets, train_bucket_size, config)

        model, epoch_results = train_epoch(model, train_dataloader, val_dataloader, criterion, optimizer, device)

        dataset_results = {}

        wandb_log = {
            "val_loss": epoch_results["total_val_loss"] / len(val_dataloader),
            "train_loss":epoch_results["total_train_loss"] / len(train_dataloader),
            "pearson_correlation": epoch_results["pearson_correlation"]
        }

        i = 0
        for val_dataset in val_datasets:
            preds = epoch_results["all_predictions"][i:i+len(val_dataset)]
            labels = epoch_results["all_labels"][i:i+len(val_dataset)]
            val_loss = criterion(torch.from_numpy(preds).to(device), torch.from_numpy(labels).to(device)) #mean_squared_error(labels, preds) / len(labels)
            pearson_corr = stats.pearsonr(labels, preds)
            logger.info(
                f'Epochs: {epoch + 1}  | Dataset: {val_dataset.dataset_name} | Val-Loss: {val_loss: .3f} '
                f'  | Val-r: {pearson_corr[0]: .4f} | p-value r=0: {pearson_corr[1]: .4f} ')
            dataset_results[val_dataset.dataset_name] = {
                "val_loss": val_loss,
                "pearson_correlation": pearson_corr[0],
                "pearson_correlation_p": pearson_corr[1],
                "all_labels": labels,
                "all_predictions": preds
            }
            wandb_log[f"{val_dataset.dataset_name}_val_loss"] = val_loss
            wandb_log[f"{val_dataset.dataset_name}_val_corr"] = pearson_corr[0]
            i += len(val_dataset)

        results["epoch_loss"].append(epoch_results["val_loss"])
        results["epoch_corr"].append(epoch_results["pearson_correlation"])

        if "Dataset_v1" in dataset_results:
            results["abag_epoch_loss"].append(dataset_results["Dataset_v1"]["val_loss"])
            results["abag_epoch_corr"].append(dataset_results["Dataset_v1"]["pearson_correlation"])

        if dataset_results["Dataset_v1"]["val_loss"] < best_loss or dataset_results["Dataset_v1"]["pearson_correlation"] > best_pearson_corr:
            patience = config.patience
            best_loss = min(dataset_results["Dataset_v1"]["val_loss"], best_loss)
            best_pearson_corr = max(dataset_results["Dataset_v1"]["pearson_correlation"], best_pearson_corr)
            for val_dataset in val_datasets: # correlation plot for each dataset
                dataset_name = val_dataset.dataset_name
                plot_correlation(x=dataset_results[dataset_name]["all_labels"], y=dataset_results[dataset_name]["all_predictions"],
                                 path=os.path.join(plot_path, f"{dataset_name}.png"))

            plot_correlation(x=epoch_results["all_labels"], y=epoch_results["all_predictions"],
                             path=os.path.join(plot_path, "all_bucket_predictions.png"))

            best_model = deepcopy(model)
        else:
            patience -= 1

        logger.info(
            f'Epochs: {epoch + 1} | Train-Loss: {epoch_results["total_train_loss"] / len(train_dataloader) : .3f}'
            f'  | Val-Loss: {epoch_results["total_val_loss"] / len(val_dataloader) : .3f} | Val-r: {epoch_results["pearson_correlation"]: .4f} | p-value r=0: {epoch_results["pearson_correlation_p"]: .4f} | Patience: {patience} ')


        wandb.log(wandb_log)

        if patience < 0:
            if use_wandb:
                run.summary["best_pearson"] = best_pearson_corr
                run.summary["best_loss"] = best_loss
            break

        model.train()

    results["best_loss"] = best_loss
    results["best_correlation"] = best_pearson_corr

    return results, best_model


def finetune_backbone(model: ModelWithBackbone, train_dataset: AffinityDataset, val_dataset: AffinityDataset,
                      args: Namespace, lr_reduction: float = 1e-02) -> Tuple[nn.Module, Dict]:
    """ Utility to finetune the backbone model using a lowered learning rate

    Args:
        model: Model with a backbone
        train_dataset: Dataset used for finetuning
        val_dataset: Validation dataset
        args: CLI Arguments
        lr_reduction: Value the learning rate gets multiplied by

    Returns:
        Tuple: Finetuned model, results as dict
    """

    # lower learning rate for backbone finetuning
    args.learning_rate = args.learning_rate * lr_reduction

    # make backbone model trainable
    model.backbone_model.requires_grad = True

    results, model = train_loop(model, train_dataset, val_dataset, args)

    logger.info("Fintuning Backbone completed")
    logger.debug(results)

    return model, results

