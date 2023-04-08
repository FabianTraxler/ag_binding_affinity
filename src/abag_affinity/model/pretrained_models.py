"""Module for GNNs utilizing pretrained backbone models to get node embeddings and then prediction binding affininty"""
import os
from pathlib import Path
import torch
from torch import device
from typing import Dict

from openfold.model.primitives import LayerNorm

# Binding_dgg modules
from abag_affinity.binding_ddg_predictor.models.predictor import DDGPredictor
from guided_protein_diffusion.models.ipa_denoiser import IPADenoiser, PairwiseRelPos


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
        x = data_dict["graph"]["node"].x

        atom_pos = x[:, :14*3].reshape(1, -1, 14, 3)
        aa = x[:, 42].reshape(1, -1).long()
        seq = x[:, 43].reshape(1, -1).long()
        chain_seq = x[:, 44].reshape(1, -1).long()
        atom_pos_mask = x[:, 45:].reshape(1, -1, 14).bool()

        x = self.backbone_model(atom_pos, aa, seq, chain_seq, atom_pos_mask)

        return x.squeeze(0)


class DeepRefineBackbone(torch.nn.Module):
    """Wrapper class for the DeepRefine model

    Code: https://github.com/BioinfoMachineLearning/DeepRefine
    Paper: https://arxiv.org/abs/2205.10390
    """

    def __init__(
        self, pretrained_model_path: str, device: device = torch.device("cpu")
    ):
        # DeepRefine modules
        from project.modules.deeprefine_lit_modules import LitPSR
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

        return embedded_graph_nodes.ndata["f"]

    def unfreeze(self):
        self.deep_refine.unfreeze()


class IPABindingEmbedder(torch.nn.Module):
    """
    Wrapper around IPADenoiser to allow use with the features generated from Fabian's pipeline (or, the OF embeddings...).

    Functions as a graph-embedding-processor.

    Very similar to IPAWrapper (for diffusion process, maybe DRY this up?)
    """

    def __init__(self, pretrained_model_path: str, device: device = torch.device("cpu"), c_s: int = 384, c_z: int = 128, relpos_k: int = 32):
        """
        TODO: implement optional weights sharing between blocks
        """
        super(IPABindingEmbedder, self).__init__()

        self.model_type = "IPA"

        # default arguments fit pretrained weights
        self.model = IPADenoiser(no_blocks=2, only_s_updates=True)

        of_weights = torch.load(pretrained_model_path, map_location=torch.device(device))

        # rename keys
        ipa_denoiser_weigths = {  # TODO check that they arrive (names fit etc.)
            k.replace("structure_module.", ""): v
            for k, v in of_weights.items()
            if k.startswith("structure_module.")
        }
        # non-strict loading due to additional modules (that are not reflected in the pretrained weights)
        self.model.load_state_dict(ipa_denoiser_weigths, strict=False)
        self.embedding_size = c_s
        self.c_z = c_z
        self.z_linear = torch.nn.Linear(3, self.c_z)
        self.pairwise_rel_pos = PairwiseRelPos(c_z=self.c_z, relpos_k=relpos_k)  # AF2
        self.layer_norm_z = LayerNorm(c_z)

        # At first, we only train the z-model
        self.freeze_ipa()

    def forward(self, data_dict: Dict):
        # Prepare data for IPA

        nodes = data_dict["graph"]["node"]
        edges = data_dict["graph"]["node", "edge", "node"]

        # edge_index = data["graph"]["node", "edge", "node"].edge_index
        # edge_attr = data["graph"]["node", "edge", "node"].edge_attr

        # the normal z is built from pairwise_rel_pos + the sum of embeddings of the two nodes. we ignore these embeddings for now, as we will later use the original z. instead, we

        A = torch.zeros(nodes.x.shape[0], nodes.x.shape[0], 3).to(nodes.x)
        A[edges.edge_index[0], edges.edge_index[1]] = edges.edge_attr
        z = self.layer_norm_z(self.pairwise_rel_pos(nodes.residue_index.squeeze(-1).to(A)) + self.z_linear(A))

        outputs = self.model(
            data={"positions": nodes.positions, "orientations": nodes.orientations},
            context={"residue_index": nodes.residue_index},
            s=nodes.x,
            z=z,  # TODO generate z via linear layer from edge features
        )

        # Use the final <s> embedding
        return outputs["single"]

    def freeze_ipa(self):
        """
        Freeze the component of the model for which weights have been loaded
        """
        for name, param in self.model.named_parameters():
            # if (
            #     not name.startswith("pairwise_rel_pos.")
            #     and not name.startswith("aatype_module.")
            #     and not name.startswith("inject_")
            # ):
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze published
        """

        for name, param in self.model.named_parameters():
            param.requires_grad = True
