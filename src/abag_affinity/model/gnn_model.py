"""Submodule for Graph Neural Networks trained to predict binding affinity"""

import argparse
import logging
import torch
import torch.nn as nn

from typing import Dict, List, Optional

from .utils import pretrained_models, pretrained_embeddings, NoOpModel
from .regression_heads import EdgeRegressionHead, RegressionHead
from .graph_conv_layers import NaiveGraphConv, GuidedGraphConv
import pytorch_lightning as pl


class DatasetAdjustment(nn.Module):
    """
    Dataset-specific adjustment layer (linear layer). Initially frozen by default

    Theoretically there is a sigmoidal relationship between the log-affinity and the enrichment value. However, in pooled (DMS) experiments the relationship is much more complex (and potentially linear).
    As we observed that DMS modeling works better with sigmoid, we include it here

    TODO: Can be experimented with to see whether it improves learning from distinct datasets (e.g. by adding an additional layer)
    Args:
        output_sigmoid: Whether to apply a sigmoid to the output

    """
    def __init__(self, layer_type, out_n):
        """
        As we initialize with weight=1 and bias=0, implementing bias_only is as simple as only unfreezing bias_only in requires_grad_
        """
        super(DatasetAdjustment, self).__init__()
        self.layer_type = layer_type
        if self.layer_type in ["identity", "bias_only", "regression", "regression_sigmoid"]:
            self.linear = nn.Linear(1, out_n)
            self.linear.weight.data.fill_(1)
            self.linear.bias.data.fill_(0)
        else:
            raise NotImplementedError("'mlp' is not implemented at the moment")

        super().requires_grad_(False)  # Call original version to freeze all parameters

    def forward(self, x: torch.Tensor, layer_selector: torch.Tensor):
        """
        Args:
            x: torch.Tensor of shape (batch_size, 1)
            layer_selector: torch.Tensor of shape (batch_size) with values between 0 and n_out. If -1, return x
        """
        x_all = self.linear(x)
        # Select the correct output node (dataset-specificity)
        x_selected = x_all[torch.arange(x_all.size(0)), layer_selector]
        if self.layer_type.endswith("_sigmoid"):
            x_selected = torch.sigmoid(x_selected)
            num_excessive = (x_selected == 0).sum() + (x_selected == 1).sum()
            if num_excessive > 0:
                print(f"WARNING: Vanishing gradients in {num_excessive} of {len(x_selected.flatten())} due to excessively large values from NN.")

        # "-1"-selector is interpreted as returning x
        x_selected = torch.where(layer_selector == -1, x.squeeze(-1), x_selected).unsqueeze(-1)

        return x_selected

    def requires_grad_(self, requires_grad: bool = True) -> nn.Module:
        """
        Overwriting requires_grad_ to enable training of bias only
        """
        if self.layer_type == "bias_only":
            self.linear.bias.requires_grad_(requires_grad)
            return self
        elif self.layer_type == "identity":
            return self
        else:
            return super().requires_grad_(requires_grad)

class AffinityGNN(pl.LightningModule):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 args: argparse.Namespace,  # provide args so they can be saved by the LightningModule (hparams) and for DatasetAdjustment
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
                 dataset_names: List = None):
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
            scaled_output: Whether to scale the output to the range using a sigmoid [0, 1]. Warning, this does not work too nicely apparently
            dataset_names: Names of all used datasets (for dataset-adjustment layers. avoid :absolute, :relative identifiers)
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
        self.dataset_specific_layer = DatasetAdjustment(args.dms_output_layer_type, len(dataset_names))
        self.scaled_output = scaled_output

        self.float()

        self.to(device)

    def forward(self, data: Dict) -> Dict:
        """

        Args:
            data: Dict with "graph": HeteroData and "filename": str and optional "deeprefine_graph": dgl.DGLGraph
            dataset_adjustment: Whether to run a dataset-specific module (linear layer) at the end (to map E-values to binding affinities)

        Returns:
            Dict containing neglogkd and evalue. Note that evalue is same as neglogkd (identity), the corresponding `dataset_adjustment` value was None
        """
        output = {}
        # calculate pretrained node embeddings
        data = pretrained_embeddings(data, self.pretrained_model)

        # calculate node embeddings
        graph = self.graph_conv(data["graph"])

        # calculate binding affinity
        neglogkd = self.regression_head(graph)

        # dataset-specific scaling (could be done before or after scale_output)
        dataset_indices = torch.Tensor([self.dataset_names.index(dataset) if dataset is not None else -1
                                        for dataset in data["dataset_adjustment"]]).long().to(neglogkd.device)
        evalue = self.dataset_specific_layer(neglogkd, dataset_indices)

        if self.dataset_specific_layer.layer_type.endswith("_sigmoid") and not self.scaled_output and any(data["dataset_adjustment"]):
            raise NotImplementedError("Would need to allow scaling the sigmoidal values back to the original range")

        return {
            "-log(Kd)": -log(Kd),
            "E": evalue,
        }

    def unfreeze(self):
        """
        Unfreeze potentially frozen modules

        TODO I should just use requires_grad_ everywhere
        """

        # make pretrained model trainable
        self.pretrained_model.requires_grad = True
        try:
            self.pretrained_model.unfreeze()
        except AttributeError:
            logging.warning("Pretrained model does not have an unfreeze method")

        # unfreeze datasets-specific layers
        self.dataset_specific_layer.requires_grad_(True)

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
