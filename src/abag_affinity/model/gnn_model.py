"""Submodule for Graph Neural Networks trained to predict binding affinity"""

import torch
from typing import Dict

from .utils import pretrained_models, pretraind_embeddings, NoOpModel
from .regression_heads import EdgeRegressionHead, RegressionHead
from .graph_conv_layers import NaiveGraphConv, GuidedGraphConv


class AffinityGNN(torch.nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 num_nodes: int = None,
                 pretrained_model: str = "", pretrained_model_path: str = "",
                 gnn_type: str = "5A-proximity",
                 layer_type: str = "GAT", num_gnn_layers: int = 3, size_halving: bool = True,
                 node_type: str = "residue",
                 aggregation_method: str = "sum",
                 nonlinearity: str = "relu",
                 num_fc_layers: int = 3,
                 device: torch.device = torch.device("cpu")):
        super(AffinityGNN, self).__init__()

        # define pretrained embedding model
        if pretrained_model != "":
            self.pretrained_model = pretrained_models[pretrained_model](pretrained_model_path, device)
            node_feat_dim = self.pretrained_model.embedding_size
        else:
            self.pretrained_model = NoOpModel() # only return input

        # define GNN Layers
        if gnn_type == "guided":
            self.graph_conv = GuidedGraphConv(node_feat_dim, edge_feat_dim, node_type, layer_type, num_gnn_layers, size_halving,
                                             nonlinearity)
        elif "proximity" in gnn_type:
            self.graph_conv = NaiveGraphConv(node_feat_dim, edge_feat_dim, layer_type, num_gnn_layers, size_halving,
                                             nonlinearity)
        else:
            raise ValueError(f"Invalid gnn_type given: Got {gnn_type} but expected one of ('guided', 'proximity')")
        # define regression head
        if aggregation_method == "edge":
            self.regression_head = EdgeRegressionHead(self.graph_conv.embedding_dim, num_fc_layers, nonlinearity, device)
        else:
            self.regression_head = RegressionHead(self.graph_conv.embedding_dim, num_nodes, aggregation_method,
                                                  nonlinearity, num_fc_layers, device)

        self.float()

        self.device = device
        self.to(device)

    def forward(self, data: Dict) -> Dict:
        """

        Args:
            data: Dict with "graph": HeteroData and "filename": str and optional "deeprefine_graph": dgl.DGLGraph

        Returns:
            torch.Tensor
        """
        # calculate pretrained node embeddings
        data = pretraind_embeddings(data, self.pretrained_model)

        # calculate node embeddings
        graph = self.graph_conv(data["graph"])

        # calculate binding affinity
        affinity = self.regression_head(graph)

        output = {
            "x": affinity
        }
        return output
