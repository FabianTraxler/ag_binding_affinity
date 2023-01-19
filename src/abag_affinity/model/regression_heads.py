"""Submodule for Knowledge primed networks (networks that derive its strucutre from previous knowledge)"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import HeteroData

from .utils import nonlinearity_function, aggregation_methods


def get_edge_batches(num_edges: int, data: HeteroData):
    batch = torch.zeros(num_edges, dtype=torch.int64)
    if hasattr(data, "num_graphs") and data.num_graphs > 1:
        i = 0
        j = 0
        for ii, data_graph in enumerate(data.to_data_list()):
            if data_graph["node", "interface", "node"].edge_index.shape[1] == 0:
                batch = torch.hstack([batch, torch.zeros(1, dtype=torch.int64)])
                j += 1
            else:
                j += data_graph["node", "interface", "node"].edge_index.shape[1]
            batch[i:j] = ii
            i = j

    return batch


def get_node_batches(data: HeteroData):
    batch = torch.zeros(data.num_nodes, dtype=torch.int64)
    if hasattr(data, "num_graphs") and data.num_graphs > 1:
        i = 0
        for ii, data_graph in enumerate(data.to_data_list()):
            j = i + data_graph.num_nodes
            batch[i:j] = ii
            i = j

    return batch


class EdgeRegressionHead(torch.nn.Module):
    """Calculate binding affinity based on interface edges"""

    def __init__(self, node_dim: int, num_layers: int = 3, nonlinearity: str = "relu", size_halving: bool = True,
                 device: torch.device = torch.device("cpu")):
        super(EdgeRegressionHead, self).__init__()
        # embed each interface edge
        input_dim = 2 * node_dim + 1  # 2x node + distance
        step_size = int(input_dim / num_layers) if size_halving else 0
        out_dim = input_dim - step_size

        self.layers = []
        for i in range(num_layers):
            if i == num_layers - 1:  # last layer
                out_dim = 1
            self.layers.append(nn.Linear(input_dim, out_dim))
            input_dim = out_dim
            out_dim = input_dim - step_size
        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = nonlinearity_function[nonlinearity]()

        self.double()
        self.device = device

    def forward(self, data: HeteroData):
        x = data["node"].x
        # get interface edges
        interface_edges = data["node", "interface", "node"].edge_index
        interface_distances = data["node", "interface", "node"].edge_attr.unsqueeze(1)
        # get interface edge embeddings
        edge_embeddings = torch.hstack([x[interface_edges[0]], x[interface_edges[1]], interface_distances])

        for layer in self.layers[:-1]:
            edge_embeddings = layer(edge_embeddings)
            edge_embeddings = self.activation(edge_embeddings)
        edge_embeddings = self.layers[-1](edge_embeddings)

        # handle batches
        batch = get_edge_batches(len(edge_embeddings), data).to(self.device)

        # affinity = global_mean_pool(edge_embeddings, torch.zeros(len(edge_embeddings)).long())
        affinity = global_add_pool(edge_embeddings, batch)

        return affinity


class RegressionHead(torch.nn.Module):
    def __init__(self, node_feat_dim: int, num_nodes: int = None, aggregation_method: str = "sum",
                 nonlinearity: str = "relu", size_halving: bool = True,
                 num_fc_layers: int = 3, device: torch.device = torch.device("cpu")):
        super(RegressionHead, self).__init__()

        # define activation function
        self.activation = nonlinearity_function[nonlinearity]()

        # define graph aggregation function
        if aggregation_method == "attention":
            self.aggregation = aggregation_methods[aggregation_method](nn.Linear(node_feat_dim, 1))
        else:
            self.aggregation = aggregation_methods[aggregation_method]

        # define regression head
        self.fc_layers = []
        if aggregation_method == "fixed_size":  # multiply node embedding size times num nodes
            in_dim = node_feat_dim * num_nodes
        else:
            in_dim = node_feat_dim
        step_size = int(in_dim / num_fc_layers) if size_halving else 0
        out_dim = in_dim - step_size
        for i in range(num_fc_layers):
            if i == num_fc_layers - 1:  # last layer
                out_dim = 1
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = in_dim - step_size

        self.fc_layers = torch.nn.ModuleList(self.fc_layers)

        self.double()
        self.device = device

    def forward(self, data: HeteroData):
        x = data["node"].x

        batch = get_node_batches(data).to(self.device)

        # calculate graph embedding
        x = self.aggregation(x, batch)

        # calculate affinity
        for fc_layer in self.fc_layers[:-1]:
            x = fc_layer(x)
            x = self.activation(x)
        x = self.fc_layers[-1](x)
        return x
