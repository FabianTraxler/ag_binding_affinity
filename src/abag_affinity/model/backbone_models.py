"""Module for GNNs utilizing pretrained backbone models to get node embeddings and then prediction binding affininty"""
import os
from pathlib import Path
import torch
from torch import device
from typing import Dict

# DeepRefine modules
from project.modules.deeprefine_lit_modules import LitPSR
# Binding_dgg modules
from abag_affinity.binding_ddg_predictor.models.predictor import DDGPredictor


class DDGBackbone(torch.nn.Module):
    """Wrapper class for the Binding DDG model

    Code: https://github.com/HeliXonProtein/binding-ddg-predictor
    Paper: https://www.pnas.org/doi/10.1073/pnas.2122954119
    """
    def __init__(self, pretrained_model_path: str, device: device = torch.device("cpu")):
        super(DDGBackbone, self).__init__()

        self.model_type = "binding_ddg_predictor"

        ckpt = torch.load(pretrained_model_path, map_location=torch.device(device))
        config = ckpt['config']
        weight = ckpt['model']

        model = DDGPredictor(config.model).to(device)
        model.load_state_dict(weight)

        self.embedding_size = config.model.node_feat_dim
        self.edge_embedding_size = 3

        self.backbone_model = model.encoder
        self.device = device
        self.to(device)

    def forward(self, data_dict: Dict):
        graph = data_dict["graph"]
        x = graph["node"].x

        atom_pos = x[:, :14*3].reshape(1, -1, 14, 3)
        aa = x[:, 42].reshape(1, -1).long()
        seq = x[:, 43].reshape(1, -1).long()
        chain_seq = x[:, 44].reshape(1, -1).long()
        atom_pos_mask = x[:, 45:].reshape(1, -1, 14).bool()

        x = self.backbone_model(atom_pos, aa, seq, chain_seq, atom_pos_mask)

        return x


class DeepRefineBackbone(torch.nn.Module):
    """Wrapper class for the DeepRefine model

    Code: https://github.com/BioinfoMachineLearning/DeepRefine
    Paper: https://arxiv.org/abs/2205.10390
    """
    def __init__(self, pretrained_model_path: str, device: device = torch.device("cpu")):
        super(DeepRefineBackbone, self).__init__()

        self.model_type = "deeprefine"

        self.deep_refine = LitPSR.load_from_checkpoint(pretrained_model_path,
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

        embedded_graph_nodes = self.deep_refine.shared_forward(data_dict["deeprefine_graph"], data_dict["filepath"])

        return embedded_graph_nodes
