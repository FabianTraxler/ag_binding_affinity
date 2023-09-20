"""Submodule for Graph Neural Networks trained to predict binding affinity"""

import argparse
import logging
import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional

from .utils import pretrained_models, pretrained_embeddings, NoOpModel, PositiveLinear
from .regression_heads import EdgeRegressionHead, RegressionHead
from .graph_conv_layers import NaiveGraphConv, GuidedGraphConv
import pytorch_lightning as pl
from collections import defaultdict

# We compute the default values for the dataset specific outputlayer using linear regression on available paired data
# More details can be found in data_analysis/Finding_Optimal_h_function.ipynb
DATASET_WEIGHT_BIAS_DICT = defaultdict(lambda: (0.87047189, 0.34777069727887455))
DATASET_WEIGHT_BIAS_DICT['madan21_mutat_hiv'] = (0.8282454412193105, 0.41644568799226406)
DATASET_WEIGHT_BIAS_DICT['mason21_comb_optim_therap_antib_by_predic_combined_H3_3'] = (1.0829338825283161, 0.1493535863888128)
DATASET_WEIGHT_BIAS_DICT['wu17_in'] = (0.746111647985783, 0.5749515719406949)
DATASET_WEIGHT_BIAS_DICT['mason21_comb_optim_therap_antib_by_predic_combined_H3_2'] = (0.49509497516218, 0.3880937320273913)
DATASET_WEIGHT_BIAS_DICT['mason21_comb_optim_therap_antib_by_predic_combined_H3_1'] = (0.7443947282767032, 0.28162449514249066)
DATASET_WEIGHT_BIAS_DICT['wu20_differ_ha_h3_h1'] = (0.9122811283046462, 0.31676705347051387)

class DatasetAdjustment(nn.Module):
    """
    Dataset-specific adjustment layer (linear layer). Initially frozen by default

    Theoretically there is a sigmoidal relationship between the log-affinity and the enrichment value. However, in pooled (DMS) experiments the relationship is much more complex (and potentially linear).
    As we observed that DMS modeling works better with sigmoid, we include it here

    TODO: Can be experimented with to see whether it improves learning from distinct datasets (e.g. by adding an additional layer)

    """
    def __init__(self, layer_type, out_n, dataset_names = None):
        """
        Args:
        layer_type: The kind of layer used for the prediction
        Available types: "identity", "bias_only", "regression", "regression_sigmoid", "positive_regression","positive_regression_sigmoid","mlp"

        out_n: Number of outputs, should be the number of datasets
        dataset_names: A list of dataset_names (same length as out_n) to initialize with a datasetspecific prior
        """
        super(DatasetAdjustment, self).__init__()
        self.layer_type = layer_type
        if self.layer_type in ["identity", "bias_only", "regression", "regression_sigmoid", "mlp"]:
            self.linear = nn.Linear(1, out_n)
            weights = torch.ones((out_n, 1))
            bias = torch.zeros((out_n))
            if dataset_names is not None and self.layer_type not in ["identity","bias_only"]:
                # When a dataset is given, we initialize with a precomputed h function if available
                # TODO this does not make sense when using sigmoid activation
                for i, ds_name in enumerate(dataset_names):
                    weights[i,0] = DATASET_WEIGHT_BIAS_DICT[ds_name.split(":")[0]][0]
                    bias[i] = DATASET_WEIGHT_BIAS_DICT[ds_name.split(":")[0]][1]
            self.linear.weight.data = weights
            self.linear.bias.data = bias
        elif self.layer_type in ["positive_regression", "positive_regression_sigmoid"]:
            self.linear = PositiveLinear(1, out_n)
            if dataset_names is not None:
                log_weights = torch.zeros((out_n, 1))
                bias = torch.zeros((out_n))
                # When a dataset is given, we initialize with a precomputed h function if available
                for i, ds_name in enumerate(dataset_names):
                    log_weights[i, 0] = math.log(DATASET_WEIGHT_BIAS_DICT[ds_name.split(":")[0]][0])
                    bias[i] = DATASET_WEIGHT_BIAS_DICT[ds_name.split(":")[0]][1]
                self.linear.log_weight.data = log_weights
                self.linear.bias.data = bias
        else:
            raise NotImplementedError(f"'{self.layer_type}' is not implemented at the moment")
        if self.layer_type == 'mlp':
            self.linear_mlp = nn.Linear(out_n, out_n)
            self.linear_mlp.weight.data.copy_(torch.eye(out_n))
            self.linear_mlp.bias.data.fill_(0)
            self.nonlinear = nn.Identity() 

        super().requires_grad_(False)  # Call original version to freeze all parameters

    def forward(self, x: torch.Tensor, layer_selector: torch.Tensor):
        """
        Args:
            x: torch.Tensor of shape (batch_size, 1)
            layer_selector: torch.Tensor of shape (batch_size) with values between 0 and n_out. If -1, return x
        """
        x_all = self.linear(x)
        if self.layer_type == 'mlp':
            x_all = self.linear_mlp(self.nonlinear(x_all))
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


    def unfreeze(self):
        """
        Unfreeze weights to enable trainig for different settings
        """
        if self.layer_type == 'mlp':
            self.nonlinear = nn.SiLU()

        if self.layer_type == "bias_only":
            self.linear.bias.requires_grad_(True)
            return self
        elif self.layer_type == "identity":
            return self
        else:
            return super().requires_grad_(True)


class AffinityGNN(pl.LightningModule):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 args: argparse.Namespace,  # provide args so they can be saved by the LightningModule (hparams) and for DatasetAdjustment
                 num_nodes: int = None,
                 pretrained_model: str = "", pretrained_model_path: str = None,
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
        self.dataset_specific_layer = DatasetAdjustment(args.dms_output_layer_type, len(dataset_names), dataset_names)
        self.scaled_output = scaled_output

        self.float()

        self.to(device)

    def forward(self, data: Dict) -> Dict:
        """

        Args:
            data: Dict with "graph": HeteroData and "filename": str and optional "deeprefine_graph": dgl.DGLGraph
            dataset_adjustment: Whether to run a dataset-specific module (linear layer) at the end (to map E-values to binding affinities)

        Returns:
            Dict containing neg_log_kd and e_value. Note that e_value is same as neg_log_kd (identity), the corresponding `dataset_adjustment` value was None
        """
        output = {}
        # calculate pretrained node embeddings
        data = pretrained_embeddings(data, self.pretrained_model)

        # calculate node embeddings
        graph = self.graph_conv(data["graph"])

        # calculate binding affinity
        neg_log_kd = self.regression_head(graph)

        # dataset-specific scaling (could be done before or after scale_output)
        dataset_indices = torch.Tensor([self.dataset_names.index(dataset) if dataset is not None else -1
                                        for dataset in data["dataset_adjustment"]]).long().to(neg_log_kd.device)
        e_value = self.dataset_specific_layer(neg_log_kd, dataset_indices)

        if self.dataset_specific_layer.layer_type.endswith("_sigmoid") and not self.scaled_output and any(data["dataset_adjustment"]):
            raise NotImplementedError("Would need to allow scaling the sigmoidal values back to the original range")

        return {
            "-log(Kd)": neg_log_kd,
            "E": e_value,
        }

    def unfreeze(self):
        """
        Unfreeze potentially frozen modules
        """

        # make pretrained model trainable
        self.pretrained_model.requires_grad = True
        try:
            self.pretrained_model.unfreeze()
        except AttributeError:
            logging.warning("Pretrained model does not have an unfreeze method")

        # unfreeze datasets-specific layers
        self.dataset_specific_layer.unfreeze()
        
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
