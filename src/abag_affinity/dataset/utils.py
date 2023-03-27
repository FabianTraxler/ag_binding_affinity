import os
import string
import warnings
from ast import literal_eval
from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from biopandas.pdb import PandasPdb
import scipy.spatial as sp
import dgl
import logging
from openfold.np.residue_constants import restype_1to3

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


def get_pdb_path_and_id(row: pd.Series, dataset_name: str, config: Dict):
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
                                     config[data_location][dataset_name]["mutated_pdb_path"])

    else:
        pdb_id = row["pdb"]
        pdb_file_path = os.path.join(config[data_location]["path"],
                                     config[data_location][dataset_name]["folder_path"],
                                     config[data_location][dataset_name]["pdb_path"])
    pdb_file_path = os.path.join(pdb_file_path, row["filename"])

    return pdb_file_path, pdb_id


def get_graph_dict(pdb_id: str, pdb_file_path: str, of_emb_path: str, affinity: float,
                   node_type: str, distance_cutoff: int = 5, interface_hull_size: int = 10,
                   ca_alpha_contact: bool = False,) -> Dict:
    """ Generate a dictionary with node, edge and meta-information for a given PDB File

    1. Get residue information
    2. Get distances between nodes (residue or atoms)
    3. Get edge encodings
    4. (Optional) Get subset of nodes that are in interface hull

    Args:
        pdb_id: ID of PDB
        pdb_file_path: Path to PDB File
        of_emb_path: Path to OpenFold embeddings file
        affinity: Binding affinity of the complex
        node_type: Type of nodes (residue, atom)
        distance_cutoff: Max. interface distance
        interface_hull_size: Hull size expanding interface
        ca_alpha_contact: Indicator if only C-alpha atoms should be used for residue distance calculation

    Returns:
        Dict: Node, edge and meta-information for PDB (node_features, residue_infos, residue_atom_coordinates,
            adjacency_tensor, affinity, closest_residues, atom_names)
    """
    structure, _ = read_file(pdb_id, pdb_file_path)

    structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure)

    if node_type == "residue":
        distances, closest_nodes = get_distances(residue_infos, residue_distance=True, ca_distance=ca_alpha_contact)

        node_features = get_residue_encodings(residue_infos, structure_info)

        if of_emb_path:  # I think we need to inject them so harshly here, to facilitate backpropagation later. An alternative would be to load the OF data closer to the model..
            node_features, matched_positions, matched_orientations, matched_residue_index = get_residue_of_embeddings(residue_infos, of_emb_path)
            # fill residue_infos with matched positions and orientations
            for i in range(len(residue_infos)):
                residue_infos[i]["matched_position"] = matched_positions[i]
                residue_infos[i]["matched_orientation"] = matched_orientations[i]
                residue_infos[i]["matched_residue_index"] = matched_residue_index[i]
            # assert (matched_residue_index > 0).all()  # check if all residues have a match (could also just raise exception in get_residue_of_embeddings)

        adj_tensor = get_residue_edge_encodings(distances, residue_infos, distance_cutoff=distance_cutoff)
        atom_names = []

    elif node_type == "atom":
        distances, closest_nodes = get_distances(residue_infos, residue_distance=False)

        node_features, atom_names = get_atom_encodings(residue_infos, structure_info)
        adj_tensor = get_atom_edge_encodings(distances, node_features, distance_cutoff=distance_cutoff)

    else:
        raise ValueError("Invalid graph_type: Either 'residue' or 'atom'")

    assert len(residue_infos) > 0

    return {
        "node_features": node_features,
        "residue_infos": residue_infos,
        "residue_atom_coordinates": residue_atom_coordinates,
        "adjacency_tensor": adj_tensor,
        "affinity": affinity,
        "closest_residues":closest_nodes,
        "atom_names": atom_names
    }


def load_graph_dict(row: pd.Series, dataset_name: str, config: Dict, interface_folder: str, node_type: str = "residue",
                    interface_distance_cutoff: int = 5, interface_hull_size: int = None, max_edge_distance: int = 5,
                    affinity_type: str = "-log(Kd)") -> Dict:
    """ Load and process a data point and generate a graph and meta-information for it

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

    Returns:
        Dict: Graph and meta-information for that data point
    """

    pdb_file_path, pdb_id = get_pdb_path_and_id(row, dataset_name, config)

    affinity = row[affinity_type]

    if interface_hull_size is not None:
        interface_path = reduce2interface_hull(pdb_id, pdb_file_path, interface_distance_cutoff, interface_hull_size)

    if 'of_embeddings' in config['DATASETS'][dataset_name]:
        of_emb_path = os.path.join(config['DATASETS']["path"],config['DATASETS'][dataset_name]['folder_path'],config['DATASETS'][dataset_name]['of_embeddings'], pdb_id + '.pt')
    else:
        of_emb_path = None

    return get_graph_dict(pdb_id, interface_path, of_emb_path, affinity, distance_cutoff=max_edge_distance,
                          interface_hull_size=interface_hull_size, node_type=node_type)


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

    if not min < affinity < max:
        a = 0
    assert min < affinity < max, f"Affinity value out of scaling range: {affinity}"

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
    distance_idx = -1
    same_protein_idx = edge_names.index("same_protein")

    for idx, edge_name in enumerate(edge_names):
        edge_type = ("node", edge_name, "node")
        if edge_name == "distance":
            edges = np.where(adjacency_matrix[idx, :, :] > 0.001)
        elif edge_name == "same_protein": # same protein and < distance_cutoff
            edges = np.where((adjacency_matrix[idx, :, :] == 1) & (adjacency_matrix[distance_idx, :, :] < max_edge_distance))
        else:
            edges = np.where(adjacency_matrix[idx, :, :] == 1)

        all_edges[edge_type] = torch.tensor(edges).long()

        proximity = adjacency_matrix[proximity_idx, edges[0], edges[1]]
        edge_attributes[edge_type] = torch.tensor(proximity)

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

    interface_distance = distance # 1 / (distance + 1)

    all_edges[("node", "interface", "node")] = torch.tensor(interface_edges).long()
    edge_attributes[("node", "interface", "node")] = torch.tensor(interface_distance)

    full_edges = np.where(adjacency_matrix[0, :, :] > 0.001)
    full_edge_featutes = np.vstack([adjacency_matrix[0, full_edges[0], full_edges[1]],
                               adjacency_matrix[1, full_edges[0], full_edges[1]],
                               adjacency_matrix[2, full_edges[0], full_edges[1]]]).T

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
            assert chain in "LHABCDEF", f"Chain {chain} not in 'LHABCDEF'"
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

def get_residue_of_embeddings(residue_infos: list, of_emb_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get embeddings calculated for each residue using OpenFold from an external file.
    1. Load embedding file
    2. Inject embeddings at correct position by matching chain ID and residue number between residue infos and embeddings,
        because order of residues is different between the two structures

    Note: Here we can use several fields including "single", "sm" -> "single" and all the intermediary states "sm" "states". I would probably start with "sm" -> "single".

    Args:
        residue_infos: List of residue infos
        of_emb_path: Path to file containing OpenFold embeddings

    Returns:
        np.ndarray: Array with the OpenFold embeddings at the appropriate positions - shape (n, 384)
    """

    of_embs = torch.load(of_emb_path, map_location='cpu')
    # warned = False

    assert torch.all(of_embs['input_data']['context_chain_type'] != 0)  # like this, incompatible with dockground dataset
    matched_of_embs = torch.zeros(len(residue_infos), of_embs['single'].shape[-1])

    matched_positions = torch.zeros(len(residue_infos), 3)
    matched_orientations = torch.zeros(len(residue_infos), 4)
    matched_residue_index = torch.full((len(residue_infos),), -1, dtype=torch.long)

    for i, res in enumerate(residue_infos):
        chain_id = ord(res['chain_id'])
        chain_res_id = torch.nonzero(torch.logical_and(of_embs['input_data']['chain_id_pdb'] == chain_id,
                                      of_embs['input_data']['residue_index_pdb'] == res['residue_id']), as_tuple=True)

        try:
            n_elem = chain_res_id[0].nelement()
            if n_elem != 1:
                raise ValueError(f'Expected 1 matching residue, but go {n_elem}')

            aatype = of_embs["input_data"]["aatype"][chain_res_id][0]
            aatype = list(restype_1to3.values())[aatype]
            if aatype != res['residue_type'].upper():
                raise ValueError(f"OF residue type mismatch. AffGNN: {res['residue_type'].upper()}, OF/Diffusion: {aatype}")

            # All checks passed? Then
            matched_of_embs[i, :] = of_embs['single'][chain_res_id]
            matched_positions[i, :] = of_embs['input_data']['positions'][chain_res_id]
            matched_orientations[i, :] = of_embs['input_data']['orientations'][chain_res_id]
            assert of_embs['input_data']['residue_index'][chain_res_id] >= 0
            matched_residue_index[i] = of_embs['input_data']['residue_index'][chain_res_id]

        except ValueError as e:
            # if not warned:
            #     warned = True
            logger.warning(f'{e}: PDBID: {of_emb_path[-9:-3]}, chain ID: {ord(res["chain_id"])}, residue ID: {res["residue_id"]}.')  # (Won\'t warn again.)

    matched_of_embs = matched_of_embs.numpy()
    return matched_of_embs, matched_positions, matched_orientations, matched_residue_index
