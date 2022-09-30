"""Module for GNNs utilizing pretrained backbone models to get node embeddings and then prediction binding affininty"""
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch import device
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_add_pool
from typing import Dict, Union, List, Tuple

# DeepRefine modules
from project.modules.deeprefine_lit_modules import LitPSR
# Binding_dgg modules
from abag_affinity.binding_ddg_predictor.models.predictor import DDGPredictor
from .kp_gnn import AtomEdgeModel


def backbone_embeddings(data: Union[Data, List, Dict], backbone_model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """ Get the embeddings of the backbone module

    Convert the input data to the correct format and then feed through the backbone model
    Return node and edge information as well as batch information

    Args:
        data: Input data of different formats
        backbone_model: backbone model to use

    Returns:
        Tuple: node and edge information as well as batch information
    """
    if hasattr(data, "num_graphs") and data.num_graphs > 1:  # batched graphs
        x = torch.tensor([])
        for data_graph in data.to_data_list():
            x = torch.cat((x, backbone_model(data_graph).squeeze(0)))
        _, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_graphs = data.num_graphs
        data_batch = data.batch
    elif isinstance(data, Data):  # single graph
        x = backbone_model(data).squeeze(0)
        _, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_graphs = 1
        data_batch = data.batch
    else:  # DeepRefine inputs
        batched_graph = backbone_model(data)
        x = batched_graph.ndata["f"]
        edge_index = torch.stack(batched_graph.edges()).long()
        edge_attr = batched_graph.edata["f"]
        num_graphs = batched_graph.batch_size
        data_batch = torch.zeros(batched_graph.number_of_nodes())
        total_nodes = 0
        for i, num_nodes in enumerate(batched_graph.batch_num_nodes()):
            data_batch[total_nodes: total_nodes + num_nodes] = i
            total_nodes += num_nodes
        data_batch = data_batch.long()

    return x, data_batch, edge_index, edge_attr, num_graphs


class EdgePredictionModelWithEncoder(torch.nn.Module):
    def __init__(self, backbone_model: torch.nn.Module,  num_nodes: int = None, device: device = torch.device("cpu")):
        super(EdgePredictionModelWithEncoder, self).__init__()
        self.num_nodes = num_nodes

        self.encoder = backbone_model
        self.encoder.requires_grad_(False)

        self.prediction_head = AtomEdgeModel(backbone_model.embedding_size, backbone_model.edge_embedding_size, device)

        self.device = device

        self.relu = nn.LeakyReLU()
        self.to(device)

    def forward(self, data: Union[Data, List, Dict]):
        x, data_batch, edge_index, edge_attr, num_graphs = backbone_embeddings(data, self.encoder)

        hetero_graph = data["hetero_graph"]
        hetero_graph["node"].x = x
        hetero_graph[("node", "feat", "node")].edge_index = edge_index
        hetero_graph[("node", "feat", "node")].edge_attr = edge_attr

        affinity = self.prediction_head(hetero_graph)

        return affinity


class GraphConvAttentionModelWithBackbone(torch.nn.Module):
    """Model with additional Graph Attention Layers before pooling and binding affinity prediction"""
    def __init__(self, backbone_model: torch.nn.Module,  num_nodes: int = None, device: device = torch.device("cpu")):
        super(GraphConvAttentionModelWithBackbone, self).__init__()
        self.num_nodes = num_nodes

        self.backbone_model = backbone_model
        self.backbone_model.requires_grad_(False)

        self.relu = nn.LeakyReLU()

        self.conv_1 = GATv2Conv(backbone_model.embedding_size, 64, heads=2, dropout=0, edge_dim=backbone_model.edge_embedding_size)
        self.conv_2 = GATv2Conv(64 * 2, 64, heads=3, dropout=0, edge_dim=backbone_model.edge_embedding_size)

        if num_nodes is None:
            num_nodes = 1

        self.embedding_size = 64 * 3 * num_nodes

        self.fc_1 = nn.Linear(self.embedding_size, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 16)
        self.fc_4 = nn.Linear(16, 8)
        self.fc_5 = nn.Linear(8, 1)
        self.fc_6 = nn.Linear(1, 1)

        self.device = device

        self.to(device)

    def forward(self, data: Data):
        x, data_batch, edge_index, edge_attr, num_graphs = backbone_embeddings(data, self.backbone_model)

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.relu(x)

        if self.num_nodes is None:
            x = global_add_pool(x, data_batch.to(self.device))
        else:
            x = x.view(num_graphs, self.embedding_size)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        x = self.fc_4(x)
        x = self.relu(x)
        x = self.fc_5(x)
        x = self.fc_6(x)
        return x


class ModelWithBackbone(torch.nn.Module):
    """Model that directly utilizes backbone embeddings and only performs pooling and binding affinity prediction"""
    def __init__(self, backbone_model: torch.nn.Module, num_nodes: int = 25, device: device = torch.device("cpu")):
        super(ModelWithBackbone, self).__init__()
        self.max_nodes = num_nodes

        self.backbone_model = backbone_model
        self.backbone_model.requires_grad_(False)

        self.relu = nn.ReLU()
        if num_nodes is None:
            num_nodes = 1

        self.embedding_size = backbone_model.embedding_size * num_nodes
        self.fc_1 = nn.Linear(self.embedding_size, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

        self.device = device

        self.to(device)

    def forward(self, data: Data):
        x, data_batch, _, _, num_graphs = backbone_embeddings(data, self.backbone_model)

        if self.num_nodes is None:
            x = global_add_pool(x, data_batch.to(self.device))
        else:
            x = x.view(num_graphs, self.embedding_size)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        return x


class DDGBackbone(torch.nn.Module):
    """Wrapper class for the Binding DDG model

    Code: https://github.com/HeliXonProtein/binding-ddg-predictor
    Paper: https://www.pnas.org/doi/10.1073/pnas.2122954119
    """
    def __init__(self, pretrained_model_path: str, device: device = torch.device("cpu")):
        super(DDGBackbone, self).__init__()

        ckpt = torch.load(pretrained_model_path, map_location=torch.device(device))
        config = ckpt['config']
        weight = ckpt['model']

        model = DDGPredictor(config.model).to(device)
        model.load_state_dict(weight)

        self.embedding_size = config.model.node_feat_dim
        self.edge_embedding_size = 3

        self.gat_encoder = model.encoder
        self.device = device
        self.to(device)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.to(self.device)

        atom_pos = x[:, :14*3].reshape(1, -1, 14, 3)
        aa = x[:, 42].reshape(1, -1).long()
        seq = x[:, 43].reshape(1, -1).long()
        chain_seq = x[:, 44].reshape(1, -1).long()
        atom_pos_mask = x[:, 45:].reshape(1, -1, 14).bool()

        x = self.gat_encoder(atom_pos, aa, seq, chain_seq, atom_pos_mask)

        return x


class DeepRefineBackbone(torch.nn.Module):
    """Wrapper class for the DeepRefine model

    Code: https://github.com/BioinfoMachineLearning/DeepRefine
    Paper: https://arxiv.org/abs/2205.10390
    """
    def __init__(self, args, device: device = torch.device("cpu")):
        super(DeepRefineBackbone, self).__init__()

        checkpoint_path = os.path.join(args.config["PROJECT_ROOT"], args.config["MODELS"]["deep_refine"]["model_path"])

        self.deep_refine = LitPSR.load_from_checkpoint(checkpoint_path,
                                            use_wandb_logger=False,
                                            nn_type="EGR",
                                            tmscore_exec_path=os.path.join(str(Path.home()), 'Programs', 'MMalign'),
                                            dockq_exec_path=os.path.join(str(Path.home()), 'Programs', 'DockQ', 'DockQ.py'),
                                            galaxy_exec_path=os.path.join(str(Path.home()), 'Programs', 'GalaxyRefineComplex'),
                                            galaxy_home_path=os.path.join(str(Path.home()), 'Repositories', 'Lab_Repositories', 'GalaxyRefineComplex'),
                                            use_ext_tool_only=False,
                                            experiment_name="DeepRefineBackbone",
                                            strict=False)
        #self.deep_refine.freeze()
        self.embedding_size = 64
        self.edge_embedding_size = 15

        self.device = device
        self.to(device)

    def forward(self, data_dict: Dict):

        assert isinstance(data_dict, dict)

        embedded_graph = self.deep_refine.shared_forward(data_dict["graph"], data_dict["filepath"])

        return embedded_graph
