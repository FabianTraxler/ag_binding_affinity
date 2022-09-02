import torch
from torch import nn
from torch_geometric.data import DataLoader, Data
from torch_geometric.data import Data
from torch.utils.data import Subset, ConcatDataset
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
from scipy import stats
from sklearn.metrics import mean_squared_error
from argparse import Namespace
from pathlib import Path
import logging

from abag_affinity.dataset import AffinityDataset, SimpleGraphs, FixedSizeGraphs, DDGBackboneInputs, HeteroGraphs, InterfaceGraphs
from abag_affinity.model import GraphConv, GraphConvAttention, FSGraphConv, DDGBackbone, \
    GraphConvAttentionModelWithBackbone, KpGNN, TwinWrapper, ModelWithBackbone
from abag_affinity.utils.config import get_data_paths, get_resources_paths
from abag_affinity.train.wandb_config import configure
from abag_affinity.utils.visualize import plot_correlation

random.seed(10)
logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion, optimizer: torch.optim, device: torch.device = torch.device("cpu"), ) -> Tuple[nn.Module, Dict]:
    total_loss_train = 0.0
    model.train()

    for data in tqdm(train_dataloader):
        optimizer.zero_grad()
        if not isinstance(data, Data) and len(data) == 2: # relative data available
            twin_model = TwinWrapper(model)
            twin_model.train()
            output = twin_model(data[0].to(device), data[1].to(device)).flatten()
            label = data[0].y - data[1].y
        else:

            output = model(data.to(device)).flatten()
            label = data.y

        loss = criterion(output, label.to(device))

        total_loss_train += loss.item()

        loss.backward()
        optimizer.step()

    model.eval()
    total_loss_val = 0

    all_predictions = np.array([])
    all_labels = np.array([])

    for data in val_dataloader:
        if not isinstance(data, Data) and len(data) == 2: # relative data available
            twin_model = TwinWrapper(model)
            output = twin_model(data[0].to(device), data[1].to(device)).flatten()
            label = data[0].y - data[1].y
        else:
            output = model(data.to(device)).flatten()
            label = data.y

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


def train_loop(model: nn.Module, train_dataset: AffinityDataset, val_dataset: AffinityDataset, args: Namespace):
    Path(os.path.join(args.config["plot_path"], f"sequential_learning/val_set_{args.validation_set}")).mkdir(exist_ok=True, parents=True)
    wandb, config, use_wandb, run, this_run = configure(args)

    results = {
        "epoch_loss": [],
        "epoch_corr": []
    }

    train_dataloader = DataLoader(train_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=config.num_workers, batch_size=config.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion1 = nn.L1Loss().to(device) #nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr= config.learning_rate)

    best_loss = 1000
    best_pearson_corr = 0
    patience = config.patience

    best_model = deepcopy(model)

    for i in range(config.max_epochs):
        model, epoch_results = train_epoch(model, train_dataloader, val_dataloader, criterion1, optimizer, device)

        results["epoch_loss"].append(epoch_results["val_loss"])
        results["epoch_corr"].append(epoch_results["pearson_correlation"])

        if epoch_results["val_loss"] < best_loss or epoch_results["pearson_correlation"] > best_pearson_corr:
            patience = config.patience
            best_loss = min(epoch_results["val_loss"], best_loss)
            best_pearson_corr = max(epoch_results["pearson_correlation"], best_pearson_corr)
            plot_correlation(x=epoch_results["all_labels"], y=epoch_results["all_predictions"],
                             path=os.path.join(args.config["plot_path"], "sequential_learning/val_set_{}/{}.png".format(args.validation_set, train_dataset.dataset_name)))
            best_model = deepcopy(model)
        else:
            patience -= 1

        logger.info(
            f'Epochs: {i + 1} | Train-Loss: {epoch_results["total_train_loss"] / (len(train_dataset) / config.batch_size): .3f}'
            f'  | Val-Loss: {epoch_results["total_val_loss"] / (len(val_dataset) / config.batch_size): .3f} | Val-r: {epoch_results["pearson_correlation"]: .4f} | p-value r=0: {epoch_results["pearson_correlation_p"]: .4f} | Patience: {patience} ')

        wandb_log = {
            "val_loss": epoch_results["total_val_loss"] / (len(val_dataset) / config.batch_size),
            "train_loss":epoch_results["total_train_loss"] / (len(train_dataset) / config.batch_size),
            "pearson_correlation": epoch_results["pearson_correlation"],
            f"{val_dataset.dataset_name}_val_loss": epoch_results["total_val_loss"] / (len(val_dataset) / config.batch_size),
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


def load_model(model_type: str, num_features: int, args: Namespace, device: torch.device = torch.device("cpu")):
    if model_type == "GraphConv":
        model = GraphConv(num_features).to(device)
    elif model_type == "GraphAttention":
        model = GraphConvAttention(num_features).to(device)
    elif model_type == "FixedSizeGraphConv":
        model = FSGraphConv(num_features, args.max_num_nodes).to(device)
    elif model_type == "DDGBackboneFC":
        encoder = DDGBackbone("./binding_ddg_predictor/data/model.pt", device)
        model = GraphConvAttentionModelWithBackbone(encoder, num_nodes=args.max_num_nodes, device=device)
    elif model_type == "KpGNN":
        model = KpGNN(num_features, device)
    else:
        raise ValueError("Please specify valid model (GraphConv, GraphAttention, FixedSizeGraphConv, DDGBackboneFC, KpGNN)")

    return model


def load_datasets(config: Dict, dataset_name: str, data_type: str, validation_set: int, args: Namespace) -> Tuple[AffinityDataset, AffinityDataset]:
    logger.debug("Get relevant PDB Ids")
    if dataset_name == "Dataset_v1":
        summary_file, pdb_path = get_data_paths(config, dataset_name)
        dataset_summary = pd.read_csv(summary_file)
        val_ids = list(dataset_summary[dataset_summary["validation"] == 1]["pdb"].values)
        train_ids = list(
            dataset_summary[(dataset_summary["validation"] != validation_set) & (dataset_summary["test"] == False)][
                "pdb"].values)
    elif dataset_name == "PDBBind":
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path)
        mask = (summary_df["delta_G"].isna()) | (summary_df["chain_infos"].str[0] != "{")
        summary_df = summary_df[~mask]

        pdb_ids = summary_df["pdb"].values
        # random split for PDBBind
        random.shuffle(pdb_ids)
        split = int(len(pdb_ids) * 0.8)
        val_ids = pdb_ids[split:]
        train_ids = pdb_ids[:split]
    elif dataset_name == "SKEMPI.v2":
        mutation_path = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                                          config["DATA"][dataset_name]["mutated_pdb_path"])
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path)
        summary_df = summary_df[~summary_df["-log(Kd)_mut"].isna()]
        available_affinities = set(summary_df.apply(lambda row: row["#Pdb"].split("_")[0] + "_" + row["Mutation(s)_cleaned"], axis=1).values)

        pdb_ids = summary_df["pdb"].unique().tolist()
        available_pdb_ids = []
        pdb_mutation_codes = []
        for pdb_id in pdb_ids:
            pdb_id = pdb_id.upper()
            pdb_path = os.path.join(mutation_path, pdb_id)
            if os.path.exists(pdb_path):
                pdb_mutation_codes.extend([pdb_id + "_" + mutation_code.split(".")[0] for mutation_code in os.listdir(pdb_path)])
                available_pdb_ids.append(pdb_id)


        pdb_mutation_codes = available_affinities.intersection(set(pdb_mutation_codes))
        # random split for PDBBind
        random.shuffle(available_pdb_ids)
        split = int(len(available_pdb_ids) * 0.8)
        val_pdb_ids = available_pdb_ids[split:]
        train_pdb_ids = available_pdb_ids[:split]

        train_ids = [ code for code in pdb_mutation_codes if code.split("_")[0] in train_pdb_ids]
        val_ids = [ code for code in pdb_mutation_codes if code.split("_")[0] in val_pdb_ids][:5]
    else:
        raise ValueError("Please specify dataset name (Dataset_v1, PDBBind, SKEMPI.v2)")

    if dataset_name in ["SKEMPI.v2"]:
        relative_data = True
    else:
        relative_data = False

    logger.debug("Get DataLoader")

    if data_type == "SimpleGraphs":
        train_data = SimpleGraphs(config, dataset_name, train_ids, node_type=args.node_type, relative_data=relative_data)
        val_data = SimpleGraphs(config, dataset_name, val_ids, node_type=args.node_type, relative_data=relative_data)
    elif data_type == "FixedSizeGraphs":
        train_data = FixedSizeGraphs(config, dataset_name, train_ids, args.max_num_nodes, node_type=args.node_type, relative_data=relative_data)
        val_data = FixedSizeGraphs(config, dataset_name, val_ids,  args.max_num_nodes, node_type=args.node_type, relative_data=relative_data)
    elif data_type == "InterfaceGraphs":
        train_data = InterfaceGraphs(config, dataset_name, train_ids, node_type=args.node_type, relative_data=relative_data)
        val_data = InterfaceGraphs(config, dataset_name, val_ids, node_type=args.node_type, relative_data=relative_data)
    elif data_type == "DDGBackboneInputs":
        train_data = DDGBackboneInputs(config, dataset_name, train_ids,  args.max_num_nodes, relative_data=relative_data)
        val_data = DDGBackboneInputs(config, dataset_name, val_ids,  args.max_num_nodes, relative_data=relative_data)
    elif data_type == "HeteroGraphs":
        train_data = HeteroGraphs(config, dataset_name, train_ids, node_type=args.node_type, relative_data=relative_data)
        val_data = HeteroGraphs(config, dataset_name, val_ids, node_type=args.node_type, relative_data=relative_data)
    else:
        raise ValueError("Please specify dataset type (SimpleGraphs, FixedSizeGraphs, DDGBackboneInputs, HeteroGraphs)")

    return train_data, val_data


def generate_bucket_batch_sampler(data_indices_list:List, batch_size: int, shuffle: bool = False):

    all_batches = []

    for data_indices in data_indices_list:
        if len(data_indices) == 0: continue
        batches = []

        if shuffle:
            random.shuffle(data_indices)

        i = 0
        while i < len(data_indices):
            batches.append(data_indices[i:i + batch_size])
            i += batch_size

        batches.append(data_indices[i - batch_size:])
        all_batches.extend(batches)

    if shuffle:
        random.shuffle(all_batches)

    return  all_batches


def finetune_backbone(model: ModelWithBackbone, train_dataset: AffinityDataset, val_dataset: AffinityDataset):
    global LEARNING_RATE
    LEARNING_RATE = 1e-05

    model.backbone_model.requires_grad = True

    results, model = train_loop(model, train_dataset, val_dataset)


    logger.info("Fintuning Backbone completed")
    logger.debug(results)


def bucket_learning(model: torch.nn.Module, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset], args: Namespace, bucket_size: int = 1000):
    Path(os.path.join(args.config["plot_path"], f"bucket_learning/val_set_{args.validation_set}")).mkdir(exist_ok=True, parents=True)

    wandb, config, use_wandb, run, this_run = configure(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.L1Loss().to(device) #nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr= args.learning_rate)

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
    val_bucket_size = min([len(dataset) for dataset in val_datasets] + [bucket_size])

    for epoch in range(args.max_epochs):
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
        batch_sampler = generate_bucket_batch_sampler([absolute_data_indices, relative_data_indices], args.batch_size)
        train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)
        #val_buckets = []
        #for val_dataset in val_datasets:
        #    indices = random.sample(range(len(val_dataset)), val_bucket_size)
        #    val_buckets.append(Subset(val_dataset, indices))
        val_dataloader = DataLoader(ConcatDataset(val_datasets), num_workers=args.num_workers, batch_size=1, shuffle=False)

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
            for val_dataset in val_datasets:
                dataset_name = val_dataset.dataset_name
                plot_correlation(x=dataset_results[dataset_name]["all_labels"], y=dataset_results[dataset_name]["all_predictions"],
                                 path=os.path.join(args.config["plot_path"], "/bucket_learning/val_set_{}/{}.png".format(args.validation_set, dataset_name)))

            plot_correlation(x=epoch_results["all_labels"], y=epoch_results["all_predictions"],
                             path=os.path.join(args.config["plot_path"], "bucket_learning/val_set_{}/all_bucket_predictions.png"))

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

