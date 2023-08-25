"""Submodule for Graph Neural Networks trained to predict binding affinity"""

import torch
import torch.nn as nn

from typing import Dict

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
                 args=None,
                 feature_list=[]):  # provide args so they can be saved by the LightningModule (hparams)
        super(AffinityGNN, self).__init__()
        self.save_hyperparameters()
        if args is not None:
            self.max_edge_distance = args.max_edge_distance
            self.interface_distance_cutoff = args.interface_distance_cutoff
        else:
            self.max_edge_distance = 5.0
            self.interface_distance_cutoff = 5.0

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

        self.feature_list = feature_list
        self.float()

        self.to(device)

    def forward(self, data: Dict) -> Dict:
        """

        Args:
            data: Dict with "graph": HeteroData and "filename": str and optional "deeprefine_graph": dgl.DGLGraph

        Returns:
            torch.Tensor
        """
        # calculate pretrained node embeddings
        data = pretrained_embeddings(data, self.pretrained_model)

        # calculate node embeddings
        graph = self.graph_conv(data["graph"])

        # calculate binding affinity
        affinity = self.regression_head(graph)

        output = {
            "x": affinity
        }
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

    def check_feature_list(self, feature_list):
        if feature_list != self.feature_list: 
            if set(feature_list) != set(self.feature_list):
                raise ValueError(f"Feature list of the model does not match the feature list given. Model's feature list: {self.feature_list}, given feature list: {feature_list}")
            else:
                raise ValueError(f"The order of the features in the list of the model does not match the one of the givel list. Model's feature list: {self.feature_list}, given feature list: {feature_list}")
        return True


    def get_affinity_inputs(self, residue_positions, node_features, chain_type=None, chain_idx=None, atom_positions =None, pairwise_features = None):
        """
        Convert Data to the format of the Affinity Prediction model

        args:
           residue_positions: torch.Tensor (Batchsize x NNodes x 3) - The 3D positions of $C_\alpha$
           node_features: torch.Tensor (Batchsize x NNodes x NFeatures) - The Node Features per Amino Acid
           chain_type: torch.Tensor (Batchsize x NNodes) - The chain type of each node, "A" for 'Heavy/Light' and "B" for binder/antigen
           chain_idx: torch.Tensor (Batchsize x NNodes) - The chain index of each node
           atom_positions: torch.Tensor (Batchsize x NNodes x NAtoms x 3) - The 3D positions of each atom
           pairwise_features: torch.Tensor (Batchsize x NNodes x NNodes x NFeatures) - The pairwise features of each atom
        """

        # maybe not needed now
        data_batch = []
        for i in range(residue_positions.shape[0]):
            atom_pos = atom_positions[i]
            residue_pos = residue_positions[i]
            data = {'graph': None}
            # calculate average distance across all atoms given per residue
            if atom_positions is not None:
                flattened_pos = atom_pos.view(-1, 3)
                distances = torch.norm(flattened_positions.unsqueeze(0) - flattened_positions.unsqueeze(1), dim=-1)
                distances = distances.view(atom_positions.shape[1], atom_positions.shape[2],
                                           atom_positions.shape[1], atom_positions.shape[2])

                distances = torch.mean(distances, dim=(1, 3))
            # calculate distance between all pairs of residues/C-alpha atoms
            else:
                distances = torch.norm(residue_pos.unsqueeze(0) - residue_pos.unsqueeze(1), dim=-1)

            node_features = node_features
            residue_infos = {'positions': residue_pos}

            A = torch.zeros((4, residue_pos.shape[0], residue_pos.shape[0]))

            contact_map = distances < self.max_edge_distance

            # A[0,:,:] = inverse pairwise distances - only below distance cutoff otherwise 0
            # A[1,:,:] = neighboring amino acid - 1 if connected by peptide bond
            # A[2,:,:] = same protein - 1 if same chain
            # A[3,:,:] = distances

            # scale distances
            A[0, contact_map] = distances[contact_map] / self.max_edge_distance
            A[3] = distances

            if chain_type is not None:
                chain_t = chain_type[i]
                non_zeros = (chain_t != 0)
                non_zero_mat = non_zeros.unsqueeze(0) * non_zeros.unsqueeze(1)
                A[2] = ((chain_t.unsqueeze(0) - chain_t.unsqueeze(1)) == 0) * non_zero_mat
                if chain_idx is not None:
                    chain_i = chain_idx[i]
                    A[1] = ((chain_i.unsqueeze(0) - chain_i.unsqueeze(1)) == 1) * non_zero_mat * A[2]

            #TODO: properly assign graph

            data['graph'] = HeteroData({'A': A, 'node_features': node_features, 'residue_infos': residue_infos})
            data_batch.append(data)

            # TODO: batch data together
            return {
                "node_features": node_features,
                "residue_infos": residue_infos,
                "residue_atom_coordinates": residue_atom_coordinates,
                "adjacency_tensor": adj_tensor,
                "affinity": affinity,
                "closest_residues": closest_nodes,
                "atom_names": atom_names
            }

        return data
