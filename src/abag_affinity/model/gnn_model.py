"""Submodule for Graph Neural Networks trained to predict binding affinity"""

import torch
import torch.nn as nn
import pdb

from openfold.model.structure_module import InvariantPointAttention, BackboneUpdate, StructureModuleTransition
from openfold.utils.rigid_utils import Rigid, Rotation


from typing import Dict

from .utils import pretrained_models, pretraind_embeddings, NoOpModel
from .regression_heads import EdgeRegressionHead, RegressionHead
from .graph_conv_layers import NaiveGraphConv, GuidedGraphConv


class AffinityGNN(torch.nn.Module):
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


class IPABindingPredictorInterface(AffinityGNN):
    """ Interface class that wraps the IPABindingPredictor and converts the input data to a
        format appropriate to the IPA predictor
    """
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
                 device: torch.device = torch.device("cpu")):
        # requred to call AffinityGNN for testing if needed, keep for now
        # kwargs = locals()
        # kwargs.pop("self")
        # kwargs.pop("__class__")
        # super().__init__(**kwargs)
        super(AffinityGNN, self).__init__()
        self.ipa_model = IPABindingPredictor()

        self.float()
        self.device = device
        self.to(device)

    def forward(self, data: Dict):
        # do some preprocessing to convert 'node_of_embeddings' to dict field 'single' that has the proper format

        # pdb.set_trace()
        # output = super().forward(data)
        # call IPABindingPredictor
        x = self.ipa_model({"single": data["of_node"]})

        return {"x": x}



class IPABindingPredictor(nn.Module):
    """ Binding predictor based on the IPA model,
        which takes OpenFold embeddings as inputs and outputs the binding affinity
    """
    def __init__(
        self,
        c_s: int = 384,  # AF2: 384
        c_z: int = 128,  # AF2: 128 (but they sum up multiple embedding in these 128 dimensions.....)
        c_ipa: int = 16,  # AF2
        no_heads_ipa: int = 12,  # Anand: 4, AF2: 12
        no_qk_points: int = 4,  # Anand: 4, AF2: 4 probably increase?
        no_v_points: int = 8,  # Anand: 4, AF2: 8
        dropout_rate: float = 0.1,  # AF2
        no_blocks: int = 12,  # 8 is AF2, 12 is Anand et al., for sequence it's even 15
        no_transition_layers: int = 1,  # 1 is from AF2
        epsilon: float = 1e-8,  # from IPA
        inf: float = 1e5,  # from IPA
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.dropout_rate = dropout_rate
        self.inf = inf
        self.epsilon = epsilon
        self.no_transition_layers = no_transition_layers
        self.no_blocks = no_blocks
        self.layer_norm_s = nn.LayerNorm(self.c_s)
        self.layer_norm_z = nn.LayerNorm(self.c_z)

        self.linear_in = nn.Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            c_ipa,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = nn.LayerNorm(self.c_s)
        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        self.aggregating_module = AggregatingBindingPredictor(self.c_s, self.c_s)

    def forward(self, data_dict: Dict, node_mask=None, data=None) -> Dict:
        s = data_dict['single']
        if node_mask is None:
            # [*, N]
            node_mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        if "pair" in data_dict.keys():
            z = self.layer_norm_z(data_dict["pair"])
        else:
            z = torch.zeros((s.shape[0], s.shape[1], s.shape[1], self.c_z), device=s.device)

        # [*, N, C_s]
        s = self.linear_in(s)

        if data is not None:
            rigids = Rigid(rots=Rotation(quats=data["orientations"]), trans=data["positions"])
        else:
            rigids = Rigid.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                self.training,
                fmt="quat",
            )
        for _ in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(s, z, rigids, node_mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            rigids = rigids.stop_rot_gradient()

        # Aggregate all s embeddings to compute binding affinity?!
        return self.aggregating_module(s)


class AggregatingBindingPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, s):
        per_node_affinity = self.model(s)
        # Take the sum over all nodes
        return per_node_affinity.sum(-2)
