import os
import string
import warnings
from ast import literal_eval
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from biopandas.pdb import PandasPdb
import scipy.spatial as sp
import dgl
import logging
from guided_protein_diffusion.datasets.dataset import load_protein
from openfold.utils.tensor_utils import tensor_tree_map

from guided_protein_diffusion.config import get_path, args as diffusion_args
from guided_protein_diffusion.datasets import input_pipeline
from guided_protein_diffusion.datasets.loader import common_processing
from openfold.np.residue_constants import restype_1to3
from guided_protein_diffusion.diffusion.context import prepare_context
from guided_protein_diffusion.diffusion.utils import remove_mean
from guided_protein_diffusion.datasets.data_utils import compute_mean_mad

from guided_protein_diffusion.diffusion.denoising_module import OpenFoldWrapper
warnings.filterwarnings("ignore")

from abag_affinity.utils.pdb_processing import (clean_and_tidy_pdb,
                                                get_atom_edge_encodings,
                                                get_atom_encodings,
                                                get_distances,
                                                get_residue_edge_encodings,
                                                get_residue_encodings,
                                                get_residue_infos)
from abag_affinity.utils.pdb_reader import read_file
# DeepRefine Imports


alphabet_letters = set(string.ascii_lowercase)
logger = logging.getLogger(__name__)


def get_pdb_path_and_id(row: pd.Series, dataset_name: str, config: Dict, relaxed: bool = False) -> Tuple[str, str]:
    data_location = "DATASETS"
    if "-" in dataset_name:
        dataset_name, publication = dataset_name.split("-")

    if "mutation_code" in row:
        if row["mutation_code"] == "" or isinstance(row["mutation_code"], float):
            pdb_id = row["pdb"] + "-original"
        else:
            pdb_id = row["pdb"] + "-" + row["mutation_code"]

        pdb_file_path = os.path.join(config[data_location]["path"],
                                     config[data_location][dataset_name]["folder_path"],
                                     config[data_location][dataset_name]["mutated_pdb_path"] + ("_relaxed" if relaxed else ""))

    else:
        pdb_id = row["pdb"]
        pdb_file_path = os.path.join(config[data_location]["path"],
                                     config[data_location][dataset_name]["folder_path"],
                                     config[data_location][dataset_name]["pdb_path"] + ("_relaxed" if relaxed else ""))
    pdb_file_path = os.path.join(pdb_file_path, row["filename"])

    return pdb_file_path, pdb_id


def get_graph_dict(pdb_id: str, pdb_file_path: str, embeddings: Dict, 
                   node_type: str, neg_log_kd: Optional[float], e_value: Optional[float] = None, distance_cutoff: int = 5,
                   ca_alpha_contact: bool = False) -> Dict:
    """
    Generate a dictionary with node, edge and meta-information for a given PDB File.

    1. Get residue information
    2. Get distances between nodes (residue or atoms)
    3. Get edge encodings
    4. (Optional) Get subset of nodes that are in interface hull

    Args:
        pdb_id: ID of PDB
        pdb_file_path: Path to PDB File
        embeddings: Path to OpenFold embeddings file or object itself
        affinity: Binding affinity of the complex
        node_type: Type of nodes (residue, atom)
        distance_cutoff: Max. interface distance
        ca_alpha_contact: Indicator if only C-alpha atoms should be used for residue distance calculation

    Returns:
        Dict: Node, edge and meta-information for PDB (node_features, residue_infos, residue_atom_coordinates,
            adjacency_tensor, affinity, closest_residues, atom_names)
    """
    structure, _ = read_file(pdb_id, pdb_file_path)

    structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure)

    if node_type == "residue":
        distances, closest_nodes = get_distances(residue_infos, residue_distance=True, ca_distance=ca_alpha_contact)
        assert len(closest_nodes) == len(residue_infos), "Number of closest nodes does not match number of residues"
        node_features = get_residue_encodings(residue_infos, structure_info)

        if embeddings:  # I think we need to inject them so harshly here, to facilitate backpropagation later. An alternative would be to load the OF data closer to the model..
            node_features, matched_positions, matched_orientations, matched_residue_index, indices = get_residue_embeddings(residue_infos, embeddings)
            # fill residue_infos with matched positions and orientations
            for i in range(len(residue_infos)):
                residue_infos[i]["matched_position"] = matched_positions[i]
                residue_infos[i]["matched_orientation"] = matched_orientations[i]
                residue_infos[i]["matched_residue_index"] = matched_residue_index[i]
            # assert (matched_residue_index > 0).all()  # check if all residues have a match (could also just raise exception in get_residue_embeddings)

        adj_tensor = get_residue_edge_encodings(distances, residue_infos, distance_cutoff=distance_cutoff)


    elif node_type == "atom":
        distances, closest_nodes = get_distances(residue_infos, residue_distance=False)

        node_features, _ = get_atom_encodings(residue_infos, structure_info)
        adj_tensor = get_atom_edge_encodings(distances, node_features, distance_cutoff=distance_cutoff)

    else:
        raise ValueError("Invalid graph_type: Either 'residue' or 'atom'")

    assert len(residue_infos) > 0
    # TODO Current Graph Dict is very long
    # Questions: Do we still need the full residue_infos list
    # Can we cut Down the adjacency tensor here?
    # Optimally, the dict does contain only stuff we need to use for our training?!
    return {
        "node_features": node_features,
        "residue_infos": residue_infos,
        "residue_atom_coordinates": residue_atom_coordinates,
        "adjacency_tensor": adj_tensor,
        "-log(Kd)": neg_log_kd,
        "E": e_value,
        "closest_residues": closest_nodes
    }


def load_graph_dict(row: pd.Series, dataset_name: str, config: Dict, interface_folder: str, node_type: str = "residue",
                    interface_distance_cutoff: int = 5, interface_hull_size: int = None, max_edge_distance: int = 5,
                    affinity_type: str = "-log(Kd)",
                    load_embeddings: Optional[Tuple[str, str]] = None,
                    save_path: Optional[str]=None,
                    relaxed=False
                ) -> Dict:
    """ Load and process a data point and generate a graph and meta-information for it

    TODO @Mihail this function needs rf_embedding capabilities

    1. Get the PDB Path
    2. Get the affinity
    3. Get information on available chains
    4. generate graph

    Args:
        row: Dataframe row for that data point
        dataset_name: Name of the dataset
        config: Dict with config information (paths, ..)
        node_type: Type of nodes (residue, atom)
        distance_cutoff: Interface distance cutoff
        interface_hull_size: interface hull size
        max_edge_distance: Max. distance between nodes
        affinity_type: Type of affinity (Kd, Ki, ..)
        load_embeddings: Tuple of embeddings type and path to embeddings
    Returns:
        Dict: Graph and meta-information for that data point
    """

    pdb_file_path, pdb_id = get_pdb_path_and_id(row, dataset_name, config, relaxed)

    # dataframe loading can lead to empty strings, if values are not present
    neg_log_kd = row["-log(Kd)"] if isinstance(row.get("-log(Kd)"), (int, float)) else np.nan
    e_value = row["E"] if isinstance(row.get("E"), (int, float)) else np.nan

    if interface_hull_size is not None:
        pdb_file_path = reduce2interface_hull(pdb_id, pdb_file_path, interface_distance_cutoff, interface_hull_size)

    if load_embeddings:
        if load_embeddings[1]:
            embeddings = os.path.join(load_embeddings[1], pdb_id + '.pt')
            embeddings = torch.load(embeddings, map_location='cpu')
        elif load_embeddings[0] == "of":
            # NOTE generating OF embeddings might clash with parallel data loading because of GPU usage
            diffusion_data = load_protein(pdb_file_path)
            diffusion_data = tensor_tree_map(lambda x: x.to(diffusion_args.device), diffusion_data)
            embeddings = of_embedding(diffusion_data)
        elif load_embeddings[0] == "rf":
            raise ValueError("Invalid embeddings_type: Either 'of' or 'rf'")
    else:
        embeddings = None

    graph_dict = get_graph_dict(pdb_id, pdb_file_path, embeddings, node_type, neg_log_kd, e_value, distance_cutoff=max_edge_distance)

    graph_dict.pop("atom_names", None)  # remove unnecessary information that takes lot of storage
    assert len(graph_dict["node_features"]) == len(graph_dict["closest_residues"])
    if save_path:
        np.savez_compressed(save_path, **graph_dict)

    return graph_dict



def load_deeprefine_graph(file_name: str, input_filepath: str, pdb_clean_dir: str,
                          interface_distance_cutoff: int = 5, interface_hull_size: int = 7) -> dgl.DGLHeteroGraph:
    """
    (Deprecated?)
    Convert PDB file to a graph with node and edge encodings

        Utilize DeepRefine functionality to get graphs

        Args:
            file_name: Name of the file
            input_filepath: Path to PDB File
            pdb_clean_dir:
            interface_distance_cutoff:
            interface_hull_size:

        Returns:
            Dict: Information about graph, protein and filepath of pdb
    """
    # DeepRefine Imports
    from project.utils.deeprefine_utils import process_pdb_into_graph

    # Process the unprocessed protein
    cleaned_path = tidy_up_pdb_file(file_name, input_filepath, pdb_clean_dir)
    if interface_hull_size is not None:
        cleaned_path = reduce2interface_hull(file_name, cleaned_path, interface_distance_cutoff,
                                             interface_hull_size)

    # get graph info with DeepRefine utility
    graph = process_pdb_into_graph(cleaned_path, "all_atom", 20, 8.0)

    # Combine all distinct node features together into a single node feature tensor
    graph.ndata['f'] = torch.cat((
        graph.ndata['atom_type'],
        graph.ndata['surf_prox']
    ), dim=1)

    # Combine all distinct edge features into a single edge feature tensor
    graph.edata['f'] = torch.cat((
        graph.edata['pos_enc'],
        graph.edata['in_same_chain'],
        graph.edata['rel_geom_feats'],
        graph.edata['bond_type']
    ), dim=1)

    return graph

def scale_affinity(affinity: float, min: float = 0, max: float = 16) -> float:
    """ Scale affinity between 0 and 1 using the given min and max values

    Args:
        affinity: affinity to scale
        min: minimum value
        max: maximum value

    Returns:
        float: scaled affinity
    """

    #if not min < affinity < max: #TODO for abdb we have a few samples
    #    logging.warning(f"Affinity value out of scaling range {min} - {max}: {affinity}")
    affinity = np.clip(affinity, min, max)
    assert min <= affinity <= max, f"Affinity value out of scaling range: {affinity}"

    return (affinity - min) / (max - min)


def get_hetero_edges(graph_dict: Dict, edge_names: List[str], max_interface_edges: int = None,
                     only_one_edge: bool = False, max_interface_distance: int = 5, max_edge_distance: int = 5) -> Tuple[Dict, Dict]:
    """ Convert graph dict to multiple edge types used for HeteroData object

    Iterate over the edge names and add them to the dictionaries

    Add interface edges

    Args:
        graph_dict:
        edge_names:

    Returns:

    """

    adjacency_matrix = graph_dict["adjacency_tensor"]

    all_edges = {}
    edge_attributes = {}

    proximity_idx = edge_names.index("distance")
    # TODO the proximity is NOT 1/distance BUT a prescaled distance after distance cutof during graph generation
    distance_idx = -1
    same_protein_idx = edge_names.index("same_protein")

    for idx, edge_name in enumerate(edge_names):
        edge_type = ("node", edge_name, "node")
        if edge_name == "distance":
            edges = np.where(adjacency_matrix[distance_idx, :, :] < max_edge_distance)
        elif edge_name == "same_protein": # same protein and < distance_cutoff
            edges = np.where((adjacency_matrix[idx, :, :] == 1) & (adjacency_matrix[distance_idx, :, :] < max_edge_distance))
        else:
            edges = np.where(adjacency_matrix[idx, :, :] == 1)

        all_edges[edge_type] = torch.tensor(edges).long()

        proximity = adjacency_matrix[proximity_idx, edges[0], edges[1]]
        distance = adjacency_matrix[distance_idx, edges[0], edges[1]]
        edge_attributes[edge_type] = torch.tensor(distance) / max_edge_distance

    interface_edges = np.where((adjacency_matrix[same_protein_idx, :, :] != 1) & (adjacency_matrix[distance_idx, :, :] < max_interface_distance))

    if only_one_edge:
        interface_edges = np.array(interface_edges)

        node_features = graph_dict["node_features"]
        interface_edges = interface_edges[:, node_features[interface_edges[0], 20] == 1]

    distance = adjacency_matrix[distance_idx, interface_edges[0], interface_edges[1]]

    if max_interface_edges is not None:
        interface_edges = np.array(interface_edges)
        sorted_edge_idx = np.argsort(-distance)[:max_interface_edges] # use negtive values to sort descending
        interface_edges = interface_edges[:, sorted_edge_idx]
        distance = adjacency_matrix[distance_idx, interface_edges[0], interface_edges[1]]

    interface_distance = distance / max_edge_distance #  We scale by cutof to obtain 0-1 tensor

    all_edges[("node", "interface", "node")] = torch.tensor(interface_edges).long()
    edge_attributes[("node", "interface", "node")] = torch.tensor(interface_distance)

    full_edges = np.where(adjacency_matrix[distance_idx, :, :] < max_edge_distance) # Same edge as distance
    full_edge_featutes = np.vstack([adjacency_matrix[distance_idx, full_edges[0], full_edges[1]] / max_edge_distance,
                               adjacency_matrix[1, full_edges[0], full_edges[1]],
                               adjacency_matrix[2, full_edges[0], full_edges[1]],
                               1.-adjacency_matrix[2, full_edges[0], full_edges[1]]]).T

    all_edges[("node", "edge", "node")] = torch.tensor(full_edges).long()
    edge_attributes[("node", "edge", "node")] = torch.tensor(full_edge_featutes)

    return all_edges, edge_attributes


def tidy_up_pdb_file(file_name: str, pdb_filepath: str, out_dir: str) -> str:
        """ Clean and remove artefacts in pdb files that lead to errors in DeepRefine

        1. Tidy PDB with pdb-tools
        2. remove multiple models in pdb
        3. remove HETATMs and alternate locations of atoms
        4. Only keep residues that have C, Ca and N atoms (required by DeepRefine)

        Args:
            file_name: Name of the file
            pdb_filepath: Path of the original PDB file

        Returns:
            str: path of the cleaned pdb_file
        """
        """Run 'pdb-tools' to tidy-up and and create a new cleaned file"""
        # Make a copy of the original PDB filepath to circumvent race conditions with 'pdb-tools'
        cleaned_path = os.path.join(out_dir, file_name + ".pdb")
        Path(cleaned_path).parent.mkdir(exist_ok=True, parents=True)
        if os.path.exists(cleaned_path):
            return cleaned_path
        else:
            clean_and_tidy_pdb("_", pdb_filepath, cleaned_path)
            return cleaned_path


def reduce2interface_hull(file_name: str, pdb_filepath: str,
                          interface_size: int, interface_hull_size: int, ) -> str:
        """ Reduce PDB file to only contain residues in interface-hull

        Interface hull defines as class variable

        1. Get distances between atoms
        2. Get interface atoms
        3. get all atoms in hull around interface
        4. expand to all resiudes that have at least 1 atom in interface hull

        Args:
            file_name: Name of the file
            pdb_filepath: Path of the original pdb file

        Returns:
            str: path to interface pdb file
        """
        head, tail = os.path.split(pdb_filepath)
        interface_path = os.path.join(head, f"interface_hull_{interface_hull_size}", file_name + ".pdb")
        if os.path.exists(interface_path):
            return interface_path

        Path(interface_path).parent.mkdir(exist_ok=True, parents=True)

        pdb = PandasPdb().read_pdb(pdb_filepath)
        atom_df = pdb.df['ATOM']

        atom_df["chain_id"] = atom_df["chain_id"].str.upper()

        prot_1_chains = []
        prot_2_chains = []

        # iterate over chains in pandaspdb object
        for chain in atom_df["chain_id"].unique():
            #assert chain in "LHABCDEF", f"Chain {chain} not in 'LHABCDEF'"
            if chain in "LH":
                prot_1_chains.append(chain)
            else:
                prot_2_chains.append(chain)

        # calcualte distances
        coords = atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy()
        distances = sp.distance_matrix(coords, coords)

        prot_1_idx = atom_df[atom_df["chain_id"].isin(prot_1_chains)].index.to_numpy().astype(int)
        prot_2_idx = atom_df[atom_df["chain_id"].isin(prot_2_chains)].index.to_numpy().astype(int)

        # get interface
        abag_distance = distances[prot_1_idx, :][:, prot_2_idx]
        interface_connections = np.where(abag_distance < interface_size)
        prot_1_interface = prot_1_idx[np.unique(interface_connections[0])]
        prot_2_interface = prot_2_idx[np.unique(interface_connections[1])]

        # get interface hull
        interface_atoms = np.concatenate([prot_1_interface, prot_2_interface])
        interface_hull = np.where(distances[interface_atoms, :] < interface_hull_size)[1]
        interface_hull = np.unique(interface_hull)

        # use complete residues if one of the atoms is in hull
        interface_residues = atom_df.iloc[interface_hull][["chain_id", "residue_number"]].drop_duplicates()
        interface_df = atom_df.merge(interface_residues)

        if len(interface_df) <= 0:
            raise ValueError('Cannot detect interface')

        assert len(interface_df) > 0, f"No atoms after cleaning in file: {pdb_filepath}"

        pdb.df['ATOM'] = interface_df
        pdb.to_pdb(path=interface_path,
                    records=["ATOM"],
                    gz=False,
                    append_newline=True)

        return interface_path

def get_residue_embeddings(residue_infos: list, embs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    # TODO refactor function to make it understand both OF and RF embeddings

    Get latent embeddings calculated for each residue using external representation network from an external file.
    1. Load embedding file
    2. Inject embeddings at correct position by matching chain ID and residue number between residue infos and embeddings,
        because order of residues is different between the two structures

    Note: Here we can use several fields including "single", "sm" -> "single" and all the intermediary states "sm" "states". I would probably start with "sm" -> "single".

    Args:
        residue_infos: List of residue infos
        emb: Dictionary containing latent embeddings

    # TODO we might need an extra rule for 1zv5, which got a broken header and therefore(?) fails to load here

    Returns:
        np.ndarray: Array with the latent embeddings at the appropriate positions - shape (n, 384)
    """

    warned = False

    if not torch.all(embs['input_data']['context_chain_type'] != 0):  # like this, incompatible with dockground dataset
        msg = 'Context chain type not 0\n' + str(embs['input_data']['context_chain_type']) + '\n' + embs['pdb_fn']
        raise ValueError(msg)
        # print(embs['input_data']['context_chain_type'])
        # print(embs['pdb_fn'])
    matched_embs = torch.zeros(len(residue_infos), embs['single'].shape[-1])

    matched_positions = torch.zeros(len(residue_infos), 3)
    matched_orientations = torch.zeros(len(residue_infos), 4)
    matched_residue_index = torch.full((len(residue_infos),), -1, dtype=torch.long)
    indices = torch.zeros(len(residue_infos), dtype=torch.long)

    for i, res in enumerate(residue_infos):
        try:
            chain_id = ord(res['chain_id'])
        except TypeError:
            chain_id = res['chain_id']
        chain_res_id = torch.nonzero(torch.logical_and(embs['input_data']['chain_id_pdb'] == chain_id,
                                      embs['input_data']['residue_index_pdb'] == res['residue_id']), as_tuple=True)

        try:
            n_elem = chain_res_id[0].nelement()
            if n_elem != 1:
                raise ValueError(f'Expected 1 matching residue, but got {n_elem}')

            if 'residue_type' in res:  # TODO should be probably added in data_loader.py to make sure there is no error
                aatype = embs["input_data"]["aatype"][chain_res_id][0]
                aatype = list(restype_1to3.values())[aatype]
                if aatype != res['residue_type'].upper():
                    raise ValueError(f"OF residue type mismatch. AffGNN: {res['residue_type'].upper()}, OF/Diffusion: {aatype}")

            # All checks passed? Then
            matched_embs[i, :] = embs['single'][chain_res_id]
            matched_positions[i, :] = embs['input_data']['positions'][chain_res_id]
            if 'orientations' in embs['input_data'].keys():
                matched_orientations[i, :] = embs['input_data']['orientations'][chain_res_id]
            matched_residue_index[i] = embs['input_data']['residue_index_pdb'][chain_res_id]
            indices[i] = chain_res_id[1][0]  # we made sure earlier that there is only one element in chain_res_id. The first coordinate is batch

        except ValueError as e:
            if not warned:
                warned = True
                logger.warning(f'{e}: PDBID: {embs["pdb_fn"][:4]}, chain ID: {ord(res["chain_id"])}, ' +
                               f'residue ID: {res["residue_id"]} .')  # (Won\'t warn again.)

    matched_embs = matched_embs.numpy()
    return matched_embs, matched_positions, matched_orientations, matched_residue_index, indices

def of_embedding(data):
    """
    run the openfold model to get the embeddings (later, we should take these embeddings as input from somewhere)
    copied from generate_of_embeddings.py
    Returns: A dict with all AlphaFold outputs. "pair" and "single" are the representations that go into the structure_module. "sm" contains all fields genereated by the structure_module, including its intermediiary "s" states
    """

    try:
        of_embedding.openfold_model
    except AttributeError:
        of_embedding.openfold_model = OpenFoldWrapper()
        of_embedding.openfold_model.eval()
        of_embedding.property_norms = compute_mean_mad(diffusion_args.conditioning, diffusion_args.dataset)

    context = prepare_context(diffusion_args.conditioning, data, of_embedding.property_norms)  # relevant information: residue_index
    node_mask = context["node_mask"]
    data["positions"] = remove_mean(data["positions"], node_mask)  # compatibility with diffusion modeling
    t = 0

    node_mask[:] = 1.

    features = of_embedding.openfold_model._prepare_features(t, data, node_mask, context)
    outputs = of_embedding.openfold_model.openfold_model(features)

    outputs["input_data"] = data

    for costly_item_key in [
        "masked_msa_logits",
        "pair",
        "distogram_logits",
        "tm_logits",
        "predicted_tm_score",
        "aligned_confidence_probs",
        "predicted_aligned_error",
        "max_predicted_aligned_error",
    ]:
        del outputs[costly_item_key]

    return outputs


