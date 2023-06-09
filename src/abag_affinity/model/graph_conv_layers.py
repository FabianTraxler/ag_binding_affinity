import torch
from torch_geometric.data import HeteroData

from .utils import layer_types, nonlinearity_function, layer_type_edge_dim


class GuidedGraphConv(torch.nn.Module):
    """Graph Convolutional Neural network with attention mechanism
    Utilize structure properties to get better node embeddings
    1. Embed atoms first based on atoms in same residue
    2. Embed atoms based on close (<2A) atoms on same protein
    3. Get Interface edges (atoms closer than 5A and on different proteins) and embed them based on atom embeddings and distance
    4. Sum over all those edges to get binding affinity
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 node_type: str,
                 layer_type: str = "GAT", num_gat_heads: int = 3,
                 num_gnn_layers: int = 3,
                 channel_halving: bool = True, channel_doubling: bool = False,
                 nonlinearity: str = "relu"):

        super(GuidedGraphConv, self).__init__()
        if node_type == "atom":
            self.edges = ["same_residue", "same_protein"]
        elif node_type == "residue":
            self.edges = ["peptide_bond", "same_protein"]
        else:
            raise ValueError(f"Please provide a valid NodeType ('atom', 'residue') - found {node_type}")

        # define GNN Layers
        self.gnn_layers = []
        in_dim = 0

        for _ in self.edges:
            type_gnn_layer = []
            in_dim += node_feat_dim  # add initial embedding after every edge_layer (skip connections)
            for i in range(int(num_gnn_layers / len(self.edges))):
                out_dim = int(in_dim / 2) if channel_halving else in_dim
                out_dim = int(out_dim * 2) if channel_doubling else out_dim
                out_dim = max(out_dim, 1)  # guarantee minimum size == 1

                if layer_type_edge_dim[layer_type]:
                    type_gnn_layer.append(layer_types[layer_type](in_dim, out_dim, edge_dim=1,
                                                                  share_weights=True, dropout=0.25, heads=num_gat_heads))
                else:
                    num_gat_heads = 1
                    type_gnn_layer.append(layer_types[layer_type](in_dim, out_dim))
                in_dim = out_dim * num_gat_heads
            self.gnn_layers.append(torch.nn.ModuleList(type_gnn_layer))

        self.gnn_layers = torch.nn.ModuleList(self.gnn_layers)

        self.embedding_dim = in_dim + node_feat_dim

        self.activation = nonlinearity_function[nonlinearity]()

    def forward(self, data: HeteroData):
        x = data["node"].x.float()
        x_orig = x
        for i, edge_type in enumerate(self.edges):
            edge_idx = data["node", edge_type, "node"].edge_index
            edge_attr = data["node", edge_type, "node"].edge_attr
            for gnn_layer in self.gnn_layers[i]:
                x = gnn_layer(x, edge_idx, edge_attr)
                x = self.activation(x)
            x = torch.cat((x, x_orig), dim=1)

        data["node"].x = x


        return data


class NaiveGraphConv(torch.nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 layer_type: str = "GAT", num_gat_heads: int = 3,
                 num_gnn_layers: int = 3,
                 channel_halving: bool = True, channel_doubling: bool = False,
                 nonlinearity: str = "relu"):
        super(NaiveGraphConv, self).__init__()

        self.edge_feat_dim = edge_feat_dim

        # define GNN Layers
        self.gnn_layers = []
        in_dim = node_feat_dim
        for i in range(num_gnn_layers):
            out_dim = int(in_dim / 2) if channel_halving else in_dim
            out_dim = int(out_dim * 2) if channel_doubling else out_dim
            out_dim = max(out_dim, 1) # guarantee minimum size == 1
            if layer_type_edge_dim[layer_type]:
                self.gnn_layers.append(layer_types[layer_type](in_dim, out_dim, edge_dim=edge_feat_dim,
                                                               share_weights=True, dropout=0.25, heads=num_gat_heads))
            else:
                num_gat_heads = 1
                self.edge_feat_dim = 1
                self.gnn_layers.append(layer_types[layer_type](in_dim, out_dim))
            in_dim = out_dim * num_gat_heads

        self.gnn_layers = torch.nn.ModuleList(self.gnn_layers)

        self.activation = nonlinearity_function[nonlinearity]()

        self.embedding_dim = in_dim # = last out dim or in dim if num_layer = 0

    def forward(self, data: HeteroData):
        x = data["node"].x
        edge_index = data["node", "edge", "node"].edge_index
        edge_attr = data["node", "edge", "node"].edge_attr

        if self.edge_feat_dim == 1:
            edge_attr = edge_attr[:, 0].flatten()

        # calculate node embeddings
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            self.activation(x)

        data["node"].x = x

        return data


