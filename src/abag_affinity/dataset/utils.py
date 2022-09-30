from typing import Dict
import os
import pandas as pd
import string
from ast import literal_eval
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from abag_affinity.utils.pdb_processing import get_residue_encodings, \
    get_residue_edge_encodings, get_atom_edge_encodings, get_atom_encodings, get_residue_infos, get_distances, clean_and_tidy_pdb
from abag_affinity.utils.pdb_reader import read_file


alphabet_letters = set(string.ascii_lowercase)


def get_graph_dict(pdb_id: str, pdb_file_path: str, affinity: float, chain_id2protein: Dict,
                   node_type: str, distance_cutoff: int = 5, interface_hull_size: int = 10,
                   ca_alpha_contact: bool = False) -> Dict:
    """ Generate a dictionary with node, edge and meta-information for a PDB File

    1. Get residue information
    2. Get distances between nodes (residue or atoms)
    3. Get edge encodings
    4. (Optional) Get subset of nodes that are in interface hull

    Args:
        pdb_id: ID of PDB
        pdb_file_path: Path to PDB File
        affinity: Binding affinity of the complex
        chain_id2protein: Dict with protein-information for each chain
        node_type: Type of nodes (residue, atom)
        distance_cutoff: Max. interface distance
        interface_hull_size: Hull size expanding interface
        ca_alpha_contact: Indicator if only C-alpha atoms should be used for residue distance calculation

    Returns:
        Dict: Node, edge and meta-information for PDB (node_features, residue_infos, residue_atom_coordinates,
            adjacency_tensor, affinity, closest_residues, atom_names)
    """
    structure, _ = read_file(pdb_id, pdb_file_path)
    structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure, chain_id2protein)

    if node_type == "residue":
        distances, closest_nodes = get_distances(residue_infos, residue_distance=True, ca_distance=ca_alpha_contact)

        node_features = get_residue_encodings(residue_infos, structure_info, chain_id2protein)
        adj_tensor = get_residue_edge_encodings(distances, residue_infos, chain_id2protein, distance_cutoff=distance_cutoff)
        atom_names = []
        # TODO: Implement residue interface hull graph --> See interfacegraph dataset --> Remove this dataset
    elif node_type == "atom":
        distances, closest_nodes = get_distances(residue_infos, residue_distance=False)

        node_features, atom_names = get_atom_encodings(residue_infos, structure_info, chain_id2protein)
        adj_tensor = get_atom_edge_encodings(distances, node_features, distance_cutoff=distance_cutoff)

        if interface_hull_size is not None:
            # limit graph to interface + hull
            interface_nodes = np.where(adj_tensor[0, :, :] - adj_tensor[2, :, :] > 0.001)[0]
            interface_nodes = np.unique(interface_nodes)
            interface_hull = np.where(adj_tensor[3, interface_nodes, :] < interface_hull_size)
            interface_hull_nodes = np.unique(interface_hull[1])
            interface_hull_residue_expanded = np.where(adj_tensor[1, interface_hull_nodes, :] == 1)
            interface_hull_residue_expanded_nodes = np.unique(interface_hull_residue_expanded[1])

            adj_tensor = adj_tensor[:, interface_hull_residue_expanded_nodes, :][:,:,interface_hull_residue_expanded_nodes]
            node_features = node_features[interface_hull_residue_expanded_nodes]

            closest_nodes = np.where(closest_nodes[:, None] == interface_hull_residue_expanded_nodes[None, :])[1]
    else:
        raise ValueError("Invalid graph_type: Either 'residue' or 'atom'")


    assert len(residue_infos) > 0

    return {
        "node_features":node_features,
        "residue_infos": residue_infos,
        "residue_atom_coordinates": residue_atom_coordinates,
        "adjacency_tensor": adj_tensor,
        "affinity": affinity,
        "closest_residues":closest_nodes,
        "atom_names": atom_names
    }


def load_graph(row: pd.Series, dataset_name: str, config: Dict, node_type: str = "residue", distance_cutoff: int = 5,
               interface_hull_size: int = None, force_recomputation: bool = False) -> Dict:
    """ Load and process a data points and generate a graph and meta-information for that data point

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
    pdb_id = row["pdb"]

    data_location = row["data_location"]
    if "mutation_code" in row and row["mutation_code"] != "":
        pdb_file_path = os.path.join(config[data_location]["path"],
                                     config[data_location][dataset_name]["folder_path"],
                                     config[data_location][dataset_name]["mutated_pdb_path"])
    else:
        pdb_file_path = os.path.join(config[data_location]["path"],
                                     config[data_location][dataset_name]["folder_path"],
                                     config[data_location][dataset_name]["pdb_path"])
    pdb_file_path = os.path.join(pdb_file_path, row["filename"])

    affinity = row["-log(Kd)"]
    chain_id2protein = literal_eval(row["chain_infos"])

    # clean pdb
    cleaned_pdb_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                                      "cleaned_pdbs")
    cleaned_path = os.path.join(cleaned_pdb_folder, pdb_id + ".clean.pdb")
    if not os.path.exists(cleaned_path) or force_recomputation:
        clean_and_tidy_pdb(pdb_id, pdb_file_path, cleaned_path)

    return get_graph_dict(pdb_id, cleaned_path, affinity, chain_id2protein, distance_cutoff=distance_cutoff,
                          interface_hull_size=interface_hull_size ,node_type=node_type)


def scale_affinity(affinity: float, min: float = 0, max: float = 16) -> float:
    """ Scale affinity between 0 and 1 using the given min and max values

    Args:
        affinity: affinity to scale
        min: minimum value
        max: maximum value

    Returns:
        float: scaled affinity
    """
    assert min < affinity < max, "Affinity value out of scaling range"

    return (affinity - min) / (max - min)