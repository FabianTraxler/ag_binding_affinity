"""Submodule for Graph Neural Networks trained to predict binding affinity"""

import torch
import torch.nn as nn

from typing import Dict, List, Optional

from .utils import pretrained_models, pretrained_embeddings, NoOpModel
from .regression_heads import EdgeRegressionHead, RegressionHead
from .graph_conv_layers import NaiveGraphConv, GuidedGraphConv
import pytorch_lightning as pl


class AffinityGNN(pl.LightningModule):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 num_nodes: int = None,
                 pretrained_model: str = "", pretrained_model_path: str = "",
                 gnn_type: str = "5A-proximity",
                 layer_type: str = "GAT", num_gat_heads: int = 3, num_gnn_layers: int = 3,
                 channel_halving: bool = True, channel_doubling: bool = False,
                 node_type: str = "residue",
                 aggregation_method: str = "sum",
                 nonlinearity: str = "relu",
                 num_fc_layers: int = 3, fc_size_halving: bool = True,
                 device: torch.device = torch.device("cpu"),
                 scaled_output: bool = False,
                 dataset_names: List = None,
                 args=None):  # provide args so they can be saved by the LightningModule (hparams)
        """
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            num_nodes: Number of nodes in the graph
            pretrained_model: Name of pretrained model to use
            pretrained_model_path: Path to pretrained model
            gnn_type: Type of GNN to use
            layer_type: Type of GNN layer to use
            num_gat_heads: Number of GAT heads to use
            num_gnn_layers: Number of GNN layers to use
            channel_halving: Halve the number of channels after each GNN layer
            channel_doubling: Double the number of channels after each GNN layer
            node_type: Type of node to use
            aggregation_method: Method to aggregate node embeddings
            nonlinearity: Nonlinearity to use
            num_fc_layers: Number of fully connected layers to use
            fc_size_halving: Halve the size of the fully connected layers after each layer
            device: Device to use
            scaled_output: Whether to scale the output to the range [0, 1]
            dataset_names: Names of all used datasets (for dataset-adjustment layers)
            args: Arguments passed to the LightningModule
        """

        super(AffinityGNN, self).__init__()
        self.save_hyperparameters()

        # define pretrained embedding model
        if pretrained_model != "":
            self.pretrained_model = pretrained_models[pretrained_model](pretrained_model_path, device)
            node_feat_dim = self.pretrained_model.embedding_size
        else:
            self.pretrained_model = NoOpModel() # only return input

        # define GNN Layers
        if gnn_type == "guided":
            self.graph_conv = GuidedGraphConv(node_feat_dim=node_feat_dim,
                                              edge_feat_dim=edge_feat_dim,
                                              node_type=node_type,
                                              layer_type=layer_type, num_gat_heads=num_gat_heads,
                                              num_gnn_layers=num_gnn_layers,
                                              channel_halving=channel_halving, channel_doubling=channel_doubling,
                                              nonlinearity=nonlinearity)
        elif "proximity" in gnn_type:
            self.graph_conv = NaiveGraphConv(node_feat_dim=node_feat_dim,
                                             edge_feat_dim=edge_feat_dim,
                                             layer_type=layer_type, num_gat_heads=num_gat_heads,
                                             num_gnn_layers=num_gnn_layers,
                                             channel_halving=channel_halving, channel_doubling=channel_doubling,
                                             nonlinearity=nonlinearity)
        elif gnn_type == "identity":
            self.graph_conv = nn.Identity()
            setattr(self.graph_conv, "embedding_dim", node_feat_dim)
        else:
            raise ValueError(f"Invalid gnn_type given: Got {gnn_type} but expected one of ('guided', 'proximity')")
        # define regression head
        if aggregation_method == "edge":
            self.regression_head = EdgeRegressionHead(self.graph_conv.embedding_dim, num_layers=num_fc_layers,
                                                      size_halving=fc_size_halving, nonlinearity=nonlinearity,
                                                      device=device)
        else:
            self.regression_head = RegressionHead(self.graph_conv.embedding_dim, num_nodes=num_nodes,
                                                  aggregation_method=aggregation_method, size_halving=fc_size_halving,
                                                  nonlinearity=nonlinearity,  num_fc_layers=num_fc_layers, device=device)
        # Dataset-specific output layers
        self.dataset_names = dataset_names
        self.dataset_layers = nn.ModuleList([nn.Linear(1, 1) for ds_name in dataset_names if ds_name.endswith("absolute")])  # TODO could use more sophisticated modules (1 -> gelu -> 2 -> gelu  -> 1)
        self.scaled_output = scaled_output

        self.float()

        self.to(device)

    def forward(self, data: Dict, dataset_adjustment: Optional[str]=None) -> Dict:
        """

        Args:
            data: Dict with "graph": HeteroData and "filename": str and optional "deeprefine_graph": dgl.DGLGraph

        Returns:
            torch.Tensor
        """
        output = {}
        # calculate pretrained node embeddings
        data = pretrained_embeddings(data, self.pretrained_model)

        # calculate node embeddings
        graph = self.graph_conv(data["graph"])

        # calculate binding affinity
        affinity = self.regression_head(graph)

        # dataset-specific scaling (could be done before or after scale_output)
        output["dataset_adjusted"] = bool(dataset_adjustment)
        if dataset_adjustment:
            dataset_index = self.dataset_names.index(dataset_adjustment)
            affinity = self.dataset_layers[dataset_index](affinity)

        # scale output to [0, 1] to make it it easier for the model
        if self.scaled_output:
            affinity = torch.sigmoid(affinity)
            num_excessive = (affinity == 0).sum() + (affinity == 1).sum()
            if num_excessive > 0:
                print(f"WARNING: Vanishing gradients in {num_excessive} of {len(affinity.flatten())} due to excessively large values from NN.")

        output["x"] = affinity
        return output

    def on_save_checkpoint(self, checkpoint):
        """
        Drop frozen parameters (don't save them)
        """

        for param_name in [
            param_name for param_name, param in self.named_parameters() if not param.requires_grad
        ]:
            try:
                del checkpoint["state_dict"][f"model.{param_name}"]
            except KeyError:
                print(f"Key {param_name} not found")

    def training_step(self, *args):
        pass

    def train_dataloader(self, *args):
        pass
    def configure_optimizers(self, *args):
        pass
