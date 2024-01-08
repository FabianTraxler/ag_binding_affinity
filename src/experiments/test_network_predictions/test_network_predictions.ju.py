# %%
from typing import Dict, List, Tuple, Union, Optional, Callable

import sys
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

from abag_affinity.dataset import AffinityDataset
from abag_affinity.dataset.advanced_data_utils import complexes_from_dms_datasets, get_bucket_dataloader, load_datasets
from abag_affinity.model import AffinityGNN, TwinWrapper
from abag_affinity.train.wandb_config import configure
from abag_affinity.utils.config import get_data_paths, read_config
from abag_affinity.utils.visualize import plot_correlation
from abag_affinity.utils.argparse_utils import read_args_from_file, parse_args
from abag_affinity.train.utils import load_model
from abag_affinity.model import regression_heads
# %load_ext autoreload
# %autoreload 2
# %%
args_file = "base_args.txt"

sys.argv = sys.argv[:1]
args = parse_args(args_file=args_file)

config = read_config(args.config_file)

args.batch_size = 1
args.cuda = False
use_cuda = False
device = 'cpu'

train_data, val_datas = load_datasets(config, args.target_dataset, args.validation_set, args)

# wandb_benchmark_log = run_and_log_benchmarks(model, args, wandb_inst)
# %%
# model = load_model(train_data.num_features, train_datasets[0].num_edge_features, dataset_names, args, device)
# def load_model(num_node_features: int, num_edge_features: int, dataset_names: List[str], args: Namespace,
#                device: torch.device = torch.device("cpu")) -> AffinityGNN:

model = AffinityGNN.load_from_checkpoint('/home/mihail/Documents/workspace/ag_binding_affinity/results/models/2023-12-20_19-00-38_fix_labels_abag_test/model.pt',
                                         map_location='cpu')

model.to('cpu')
train_dataloader, val_dataloaders = get_bucket_dataloader(args, [train_data], val_datas)
# %%
preds_per_residue = {}
for data in train_dataloader:
    data_copy = deepcopy(data)
    # print('data', data)
    # print('data input size', data['input']['graph']['node'].x.shape)
    output = model(data_copy['input'])
    #
    print(output)
    res_types = np.where(data['input']['graph']['node'].x[:, :20] == 1)[1]
    # with np.printoptions(threshold=np.inf):
    #     print('data graph', data['input']['graph']['node'].x.numpy())

    out2 = model.graph_conv(data['input']['graph'])

    x = out2["node"].x

    batch = regression_heads.get_node_batches(out2).to(x.device)

    if model.regression_head.aggregation_method in ["interface_sum", "interface_mean", "interface_size"]:
        # get interface edges
        interface_node_indices = out2["node", "interface", "node"].edge_index.view(-1).unique()
        batch = batch[interface_node_indices]
        x = x[interface_node_indices]
        res = res_types[interface_node_indices]
    # compute node-wise affinity contribution from graph embedding
    for fc_layer in model.regression_head.fc_layers[:-1]:
        x = fc_layer(x)
        x = model.regression_head.activation(x)
    x = model.regression_head.fc_layers[-1](x)
    print(x.shape)
    for i in range(res.shape[0]):
        if res[i] in preds_per_residue.keys():
            preds_per_residue[res[i].item()].append(x[i].item())
        else:
            preds_per_residue[res[i].item()] = [x[i].item()]

print(preds_per_residue)
# %%
means_per_residue = [np.mean(preds_per_residue[i]) for i in preds_per_residue.keys()]
stds_per_residue = [np.std(preds_per_residue[i]) for i in preds_per_residue.keys()]
print('average_residue_scores', means_per_residue, '\n')
print('standard dev per residue', stds_per_residue, '\n')
print('standard dev across residues', np.std(means_per_residue))

# %%
preds_per_residue = {}
for data in val_dataloaders[0]:
    # print('data', data)
    # print('data input size', data['input']['graph']['node'].x.shape)
    # output = model(data['input'])
    #
    # print(output)
    res_types = np.where(data['input']['graph']['node'].x[:, :20] == 1)[1]
    # with np.printoptions(threshold=np.inf):
    #     print('data graph', data['input']['graph']['node'].x.numpy())

    out2 = model.graph_conv(data['input']['graph'])

    x = out2["node"].x

    batch = regression_heads.get_node_batches(out2).to(x.device)

    if model.regression_head.aggregation_method in ["interface_sum", "interface_mean", "interface_size"]:
        # get interface edges
        interface_node_indices = out2["node", "interface", "node"].edge_index.view(-1).unique()
        batch = batch[interface_node_indices]
        x = x[interface_node_indices]
        res = res_types[interface_node_indices]
    # compute node-wise affinity contribution from graph embedding
    for fc_layer in model.regression_head.fc_layers[:-1]:
        x = fc_layer(x)
        x = model.regression_head.activation(x)
    x = model.regression_head.fc_layers[-1](x)
    print(x.shape)
    for i in range(res.shape[0]):
        if res[i] in preds_per_residue.keys():
            preds_per_residue[res[i].item()].append(x[i].item())
        else:
            preds_per_residue[res[i].item()] = [x[i].item()]

print(preds_per_residue)
# %%
means_per_residue = [np.mean(preds_per_residue[i]) for i in preds_per_residue.keys()]
stds_per_residue = [np.std(preds_per_residue[i]) for i in preds_per_residue.keys()]
print('average_residue_scores', means_per_residue, '\n')
print('standard dev per residue', stds_per_residue, '\n')
print('standard dev across residues', np.std(means_per_residue))
# %%
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
                          preprocessed_to_scratch=args.preprocessed_to_scratch,
                          num_threads=args.num_workers,
                          load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path)
                          )

dataloader = DL_torch(dataset, num_workers=args.num_workers, batch_size=1,
                      collate_fn=AffinityDataset.collate)
# %%
preds_per_residue = {}
for data in dataloader:
    # print('data', data)
    # print('data input size', data['input']['graph']['node'].x.shape)
    # output = model(data['input'])
    #
    # print(output)
    res_types = np.where(data['input']['graph']['node'].x[:, :20] == 1)[1]
    # with np.printoptions(threshold=np.inf):
    #     print('data graph', data['input']['graph']['node'].x.numpy())

    out2 = model.graph_conv(data['input']['graph'])

    x = out2["node"].x

    batch = regression_heads.get_node_batches(out2).to(x.device)

    if model.regression_head.aggregation_method in ["interface_sum", "interface_mean", "interface_size"]:
        # get interface edges
        interface_node_indices = out2["node", "interface", "node"].edge_index.view(-1).unique()
        batch = batch[interface_node_indices]
        x = x[interface_node_indices]
        res = res_types[interface_node_indices]
    # compute node-wise affinity contribution from graph embedding
    for fc_layer in model.regression_head.fc_layers[:-1]:
        x = fc_layer(x)
        x = model.regression_head.activation(x)
    x = model.regression_head.fc_layers[-1](x)
    print(x.shape)
    for i in range(res.shape[0]):
        if res[i] in preds_per_residue.keys():
            preds_per_residue[res[i].item()].append(x[i].item())
        else:
            preds_per_residue[res[i].item()] = [x[i].item()]

print(preds_per_residue)
# %%
means_per_residue = [np.mean(preds_per_residue[i]) for i in preds_per_residue.keys()]
stds_per_residue = [np.std(preds_per_residue[i]) for i in preds_per_residue.keys()]
print('average_residue_scores', means_per_residue, '\n')
print('standard dev per residue', stds_per_residue, '\n')
print('standard dev across residues', np.std(means_per_residue))
# %%
train_y = []
train_pred = []
val_y = []
bench_y = []
val_pred = []
bench_pred = []

preds_per_residue = {}
for data in train_dataloader:
    # print('data', data)
    # print('data input size', data['input']['graph']['node'].x.shape)
    # output = model(data['input'])
    #
    # print(output)
    train_y.append(data['input']['graph']['-log(Kd)'].item())
    out = model(data['input'])
    train_pred.append(out['-log(Kd)'].item())
train_pred = np.array(train_pred)
train_y = np.array(train_y)
# print(train_pred)
# print(train_y)

for data in val_dataloaders[0]:
    # print('data', data)
    # print('data input size', data['input']['graph']['node'].x.shape)
    # output = model(data['input'])
    #
    # print(output)
    val_y.append(data['input']['graph']['-log(Kd)'].item())
    out = model(data['input'])
    val_pred.append(out['-log(Kd)'].item())
val_pred = np.array(val_pred)
val_y = np.array(val_y)

for data in dataloader:
    # print('data', data)
    # print('data input size', data['input']['graph']['node'].x.shape)
    # output = model(data['input'])
    #
    # print(output)
    bench_y.append(data['input']['graph']['-log(Kd)'].item())
    out = model(data['input'])
    bench_pred.append(out['-log(Kd)'].item())

bench_pred = np.array(bench_pred)
bench_y = np.array(bench_y)

train_mean = np.mean(train_y)

train_train_rmse = np.sqrt(np.mean((train_y - train_mean) ** 2))
train_val_rmse = np.sqrt(np.mean((val_y - train_mean) ** 2))
train_bench_rmse = np.sqrt(np.mean((bench_y - train_mean) ** 2))

train_rmse = np.sqrt(np.mean((train_y - train_pred) ** 2))
val_rmse = np.sqrt(np.mean((val_y - val_pred) ** 2))
bench_rmse = np.sqrt(np.mean((bench_y - bench_pred) ** 2))

print(train_rmse, val_rmse, bench_rmse)
print(train_train_rmse, train_val_rmse, train_bench_rmse)
# %%
np.corrcoef([train_rmse, val_rmse, bench_rmse], [train_train_rmse, train_val_rmse, train_bench_rmse])
