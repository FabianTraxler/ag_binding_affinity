"""Module for GNNs utilizing pretrained backbone models to get node embeddings and then prediction binding affininty"""
import os
from pathlib import Path
import torch
from torch import device
from typing import Dict
from abag_affinity.dataset.utils import get_residue_of_embeddings

from openfold.utils.tensor_utils import tensor_tree_map
from guided_protein_diffusion.datasets import input_pipeline
from guided_protein_diffusion.datasets.loader import common_processing
from guided_protein_diffusion.datasets.data_utils import compute_mean_mad
from guided_protein_diffusion.config import get_path, args as diffusion_args
from guided_protein_diffusion.datasets.abdb import AbDbDataset
from guided_protein_diffusion.diffusion.context import prepare_context
from guided_protein_diffusion.diffusion.utils import remove_mean

from openfold.model.primitives import LayerNorm
from guided_protein_diffusion.diffusion.denoising_module import OpenFoldWrapper

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
        try:
            z = data_dict["z"]
        except KeyError:
            z = self.pairwise_rel_pos(nodes.residue_index.squeeze(-1).to(A))
        z = self.layer_norm_z(z + self.z_linear(A))

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


class DiffusionPipelinePredictor(torch.nn.Module):
    """
    Uses the Diffusion-pipeline as a basis to predict binding affinities

    Should be usable in the same way as AffinityGNN (gnn_model.py)

    In the first stage this class provides an interface to be used in the abag affinity training. Later it will be used within the Diffusion pipeline
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(DiffusionPipelinePredictor, self).__init__()
        self.embedding_size = 384

        self.model_type = "Diffusion"

        from guided_protein_diffusion.utils.interact import init_interactive_environment

        init_interactive_environment(
            ["--dataset", "abdb", "--openfold_time_injection_alpha", "0.0", "--antigen_conditioning"]
        )  # implies --testing
        assert diffusion_args.batch_size == 1

        self.diffusion_dataset = AbDbDataset(root=get_path(["datasets", "abdb"]), max_antigen_length=350)

        self.ipa_binding_embedder = IPABindingEmbedder(*args, **kwargs)
        self.openfold_model = OpenFoldWrapper()

        self.property_norms = compute_mean_mad(diffusion_args.conditioning, diffusion_args.dataset)

        # TODO create a mapping between pdb_fn and index. also, make sure to get the pdb_fn from affinity_pipeline
        self.pdb_fn_to_index = {d["pdb_fn"].lower(): i for i, d in enumerate(self.diffusion_dataset)}
        self.freeze()


    def _load_diffusion_data(self, data: Dict) -> Dict:
        """
        Load data from Diffusion pipeline DataLoader
        """
        pdb_fn = Path(data["filepath"][0]).stem
        index = self.pdb_fn_to_index[pdb_fn.lower()]
        diffusion_data = self.diffusion_dataset[index]
        diffusion_data.pop("pdb_fn")

        # add batch dimension and process diffusion_data (substitute diffusion_data loader)
        diffusion_data = tensor_tree_map(lambda x: x[None], diffusion_data)
        diffusion_data = input_pipeline.process(diffusion_data)
        diffusion_data = common_processing(diffusion_data)

        return tensor_tree_map(lambda x: x.to(diffusion_args.device), diffusion_data)

    def of_embedding(self, data):
        """
        copied from generate_of_embeddings.py
        Returns: A dict with all AlphaFold outputs. "pair" and "single" are the representations that go into the structure_module. "sm" contains all fields genereated by the structure_module, including its intermediiary "s" states
        """
        context = prepare_context(diffusion_args.conditioning, data, self.property_norms)  # relevant information: residue_index
        node_mask = context["node_mask"]

        data["positions"] = remove_mean(data["positions"], node_mask)  # compatibility with diffusion modeling
        t = 0
        features = self.openfold_model._prepare_features(t, data, node_mask, context)
        outputs = self.openfold_model.openfold_model(features)
        return outputs


    def forward(self, affinity_data: Dict) -> Dict:
        """

        Args:
            data: Dict with "graph": HeteroData and "filename": str. As generated by abag_affinity dataset
        Returns:
            A dictionary containing the predicted binding affinity
        """

        # based on <data> load the dataset from our dataloader
        diffusion_data = self._load_diffusion_data(affinity_data)

        # run the openfold model to get the embeddings (later, we should take these embeddings as input from somewhere)
        of_data = self.of_embedding(diffusion_data)
        of_data["input_data"] = diffusion_data

        # pass the single embeddings from OpenFold evoformer into the data dict
        assert of_data["single"].shape[0] == 1, "Only batch size 1 is supported"

        residue_info_keys = ["chain_id", "residue_id"]
        residue_infos = [dict(zip(residue_info_keys, tuple)) for tuple in zip(*[affinity_data["graph"]["node"][key] for key in residue_info_keys])]

        node_features, matched_positions, matched_orientations, matched_residue_index, indices = get_residue_of_embeddings(residue_infos, of_data)

        affinity_data["graph"]["node"].update({
                "x": torch.tensor(node_features, device=diffusion_args.device),  # x needs shape [num_nodes, num_features]
                "positions": matched_positions.to(diffusion_args.device),
                "orientations": matched_orientations.to(diffusion_args.device),
                "residue_index": matched_residue_index.to(diffusion_args.device),
            })
        affinity_data["z"] = of_data["pair"][0, indices, indices, :]
        # run the affinity prediction model
        return self.ipa_binding_embedder(affinity_data)

    def freeze(self):
        """
        Freeze the component of the model for which weights have been loaded
        """
        self.openfold_model.freeze_published_weights()
        self.ipa_binding_embedder.freeze_ipa()

    def unfreeze(self):
        """
        Unfreeze published
        """
        self.ipa_binding_embedder.unfreeze()
