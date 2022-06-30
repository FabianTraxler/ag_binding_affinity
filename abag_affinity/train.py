"""This module provides all training utilities for the model streams in table_header_standardization"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb

import seaborn as sns
from scipy import stats

from abag_affinity.utils.config import get_data_paths, read_yaml
from abag_affinity.dataset.data_loader import AffinityDataset, get_pdb_ids, SimpleGraphs, FixedSizeGraphs, DDGBackboneInputs
from abag_affinity.dataset.hetero_data_loader import HeteroGraphs
from abag_affinity.model.graph_conv_v1 import GraphConv
from abag_affinity.model.fixed_size_graph_conv import FSGraphConv
#from abag_affinity.model.binding_ddg_backbone_gcn import DDGBackboneFC
from abag_affinity.model.kp_gnn import KpGNN


torch.cuda.empty_cache()

use_wandb = False
name = "DDGBackboneFC_v1"

BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 20
MAX_NUM_NODES = 25


def configure():
    """ Configure Weights&Bias with the respective parameters and project

    Returns:
        Tuple of Wandb instances
    """
    if use_wandb:
        run = wandb.init(project="abab_binding_affinity")
        wandb.run.name = name
        run_id = "fabian22/abab_binding_affinity/{}".format(run.id)
        api = wandb.Api()

        this_run = api.run(run_id)
    else:
        run = wandb.init(project="abag_binding_affinity", mode="disabled")
        this_run = None


    config = wandb.config
    config.batch_size = BATCH_SIZE
    config.max_epochs = EPOCHS
    config.learning_rate = LEARNING_RATE
    config.patience = PATIENCE
    config.max_num_nodes = MAX_NUM_NODES

    return wandb, config, use_wandb, run, this_run


def train_loop(model: nn.Module, train_dataset: AffinityDataset, val_dataset: AffinityDataset):
    wandb, config, use_wandb, run, this_run = configure()

    results = {
        "epoch_loss": [],
        "epoch_corr": []
    }

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=config.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion1 = nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr= LEARNING_RATE)

    best_loss = 1000
    best_pearson_corr = 0
    patience = config.patience

    for i in range(EPOCHS):
        total_loss_train = 0.0

        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            output = model(data).flatten()

            loss = criterion1(output, data.y.to(device))

            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        total_loss_val = 0

        all_predictions = np.array([])
        all_labels = np.array([])

        for data in val_dataloader:

            output = model(data).flatten()
            loss = criterion1(output, data.y.to(device))

            total_loss_val += loss.item()

            all_predictions = np.append(all_predictions, output.flatten().detach().cpu().numpy())
            all_labels = np.append(all_labels, data.y.flatten().detach().cpu().numpy())

        val_loss = total_loss_val / (len(val_dataset) / BATCH_SIZE)
        pearson_corr = stats.pearsonr(all_labels, all_predictions)

        results["epoch_loss"].append(val_loss)
        results["epoch_corr"].append(pearson_corr[0])

        if val_loss < best_loss:# and pearson_corr[0] > best_pearson_corr:
            patience = config.patience
            best_loss = val_loss
            best_pearson_corr = pearson_corr[0]
            ax = sns.scatterplot(x=all_labels, y=all_predictions)
            ax.set_title("True vs Predicted")
            ax.set_xlabel("True")
            ax.set_ylabel("Prediction")
            plt.savefig("../plots/predictions.png")
            plt.close()
        else:
            patience -= 1

        print(
            f'Epochs: {i + 1} | Train-Loss: {total_loss_train / (len(train_dataset) / BATCH_SIZE): .3f}'
            f'  | Val-Loss: {total_loss_val / (len(val_dataset) / BATCH_SIZE): .3f} | Val-r: {pearson_corr[0]: .4f} | p-value r=0: {pearson_corr[1]: .4f} ')

        wandb_log = {
            "val_loss": total_loss_val / (len(val_dataset) / BATCH_SIZE),
            "train_loss":total_loss_train / (len(train_dataset) / BATCH_SIZE),
            "pearson_correlation": pearson_corr[0]
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

    return results, model


def model_train(model_type:str, data_type: str, validation_set: int = 1):

    """
    Instantiate model training
    """

    config_file = "../abag_affinity/config.yaml"
    dataset_name = "Dataset_v1"

    config = read_yaml(config_file)
    summary_file, pdb_path = get_data_paths(config, dataset_name)
    dataset_summary = pd.read_csv(summary_file)
    val_ids = list(dataset_summary[dataset_summary["validation"] == validation_set]["pdb"].values)
    train_ids = list(dataset_summary[(dataset_summary["validation"] != validation_set) & (dataset_summary["test"] == False)]["pdb"].values)

    if data_type == "SimpleGraphs":
        train_data = SimpleGraphs(config_file, dataset_name, train_ids)
        val_data = SimpleGraphs(config_file, dataset_name, val_ids)
    elif data_type == "FixedSizeGraphs":
        train_data = FixedSizeGraphs(config_file, dataset_name, train_ids, MAX_NUM_NODES)
        val_data = FixedSizeGraphs(config_file, dataset_name, val_ids, MAX_NUM_NODES)
    elif data_type == "DDGBackboneInputs":
        train_data = DDGBackboneInputs(config_file, dataset_name, train_ids, MAX_NUM_NODES)
        val_data = DDGBackboneInputs(config_file, dataset_name, val_ids, MAX_NUM_NODES)
    elif data_type == "HeteroGraphs":
        train_data = HeteroGraphs(config_file, dataset_name, train_ids)
        val_data = HeteroGraphs(config_file, dataset_name, val_ids)
    else:
        return
    print("Val Set:", str(validation_set), " | Train Size:", len(train_data), " | Test Size:", len(val_data))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if model_type == "GraphConv":
        model = GraphConv(train_data.num_features).to(device)
    elif model_type == "FixedSizeGraphConv":
        model = FSGraphConv(train_data.num_features, MAX_NUM_NODES).to(device)
    elif model_type == "DDGBackboneFC":
        model = DDGBackboneFC("./binding_ddg_predictor/data/model.pt", device)
    elif model_type == "KpGNN":
        model = KpGNN(train_data.num_features).to(device)
    else:
        print("Please specify valid model and dataloader")
        return
    results, model = train_loop(model, train_data, val_data)
    return results


def cross_validation(model_type: str, data_type: str):
    losses = []
    correlations = []
    for i in range(1, 4):
        print("Validation on split {} and training with all other splits\n".format(i))
        results = model_train(model_type, data_type, validation_set=i)
        losses.append(results["best_loss"])
        correlations.append(results["best_correlation"])
        print("\n")

    print("Average Loss: {} ({})".format(np.mean(losses), np.std(losses)))
    print("Average Pearson Correlation: {} ({})".format(np.mean(correlations), np.std(correlations)))


def pretrain_model(model_type: str, data_type: str):
    pass


if __name__ == "__main__":
    cross_validation("KpGNN", "HeteroGraphs")
    #cross_validation("GraphConv", "SimpleGraphs")
