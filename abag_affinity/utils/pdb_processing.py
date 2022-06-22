import numpy as np
import scipy.spatial as sp
import torch
from Bio.SeqUtils import seq1
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from typing import Tuple, List, Dict
import os
from collections import defaultdict

from abag_affinity.binding_ddg_predictor.utils.protein import get_residue_pos14, ATOM_CA, augmented_three_to_index

from abag_affinity.utils.pdb_reader import read_file


AMINO_ACIDS = ["ala","cys","asp","glu","phe","gly","his","ile","lys","leu","met","asn","pro","gln","arg","ser","thr","val","trp","tyr"]

AAA2ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID2AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}


def get_distances_and_info(structure: Structure, header: Dict, chain_id2protein: Dict) -> Tuple[np.ndarray, List[Dict], np.ndarray, Dict, np.ndarray]:
    """ Calculate all distances of the amino acids in this structure
    1. Remove duplicate chains
    2. Get position of amino acid C-alpha atoms + Chain Id, AA type, AA number
    3. Calculate pairwise distances
    Args:
        structure: PDB Structure Object to be used for distance calculation
        header:
        chain_id2protein:
    Returns:
        np.ndarray: Matrix of all distances
        List: information about every amino acid (chain_id, aa_type, aa_number on chain)
    """

    compound_info = {}
    if header.get("compound") is not None:
        for _, chain_info in header["compound"].items():
            if "chain" in chain_info:
                compound_info[chain_info["chain"]] = {
                    "fragment": chain_info.get("fragment"),
                    "molecule": chain_info.get("molecule")
                }

    remove_redundant_chains(structure)

    structure_info = {
        "chain_length": defaultdict(int)
    }
    antibody_ca_atom_coordinates = []
    antibody_residues = []
    antigen_ca_atom_coordinates = []
    antigen_residues = []

    residue_atom_coordinates = []
    residue_infos = [] # (chain id, residue type, reside id)
    residue_idx = 0
    chain_idx = 1
    on_chain_residue_idx = 1
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            try:
                all_atom_coordinates = get_residue_pos14(residue).numpy()
            except:
                continue # filter HETATOMs
            residue_atom_coordinates.append(all_atom_coordinates)
            residue_info = {
                "chain_id": chain.id,
                "residue_type": residue.resname,
                "residue_id": residue.get_id()[1],
                "residue_pdb_index": augmented_three_to_index(residue.resname),
                "chain_idx": chain_idx,
                "on_chain_residue_idx": on_chain_residue_idx
            }
            if chain_id2protein[chain.id.lower()] == "antibody":
                antibody_ca_atom_coordinates.append(all_atom_coordinates[1])
                antibody_residues.append(residue_idx)
            else:
                antigen_ca_atom_coordinates.append(all_atom_coordinates[1])
                antigen_residues.append(residue_idx)

            residue_infos.append(residue_info)
            structure_info["chain_length"][chain.id.lower()] += 1
            residue_idx += 1
            on_chain_residue_idx += 1
        on_chain_residue_idx = 1
        chain_idx += 1

    residue_atom_coordinates = np.array(residue_atom_coordinates)
    antibody_residues = np.array(antibody_residues)
    antigen_residues = np.array(antigen_residues)

    distances = sp.distance_matrix(residue_atom_coordinates[:, ATOM_CA, :], residue_atom_coordinates[:, ATOM_CA, :])

    abag_distance = sp.distance_matrix(antibody_ca_atom_coordinates, antigen_ca_atom_coordinates)
    abab_indices = np.unravel_index(np.argsort(abag_distance.ravel()), abag_distance.shape)
    # convert to strucutre residue ids
    antibody_indices = antibody_residues[abab_indices[0]]
    antigen_indices = antigen_residues[abab_indices[1]]
    closest_residue_indices = np.empty((antibody_indices.size + antigen_indices.size), dtype=antigen_indices.dtype)
    closest_residue_indices[0::2] = antibody_indices
    closest_residue_indices[1::2] = antigen_indices
    _, idx = np.unique(closest_residue_indices, return_index=True)
    closest_residue_indices = closest_residue_indices[np.sort(idx)]

    return distances, residue_infos, residue_atom_coordinates, structure_info, closest_residue_indices


def remove_redundant_chains(structure: Structure):
    """ Remove redundant chains from a structure (inplace)
    Redundant chains have exact amio-acid sequence identity
    Remove second chain found
    Args:
        structure: PDB Structure Object to be cleaned
    Returns:
        None
    """

    #TODO: Remove chains of one protein and not of different
    redundant_chains = []

    all_chains = list(structure.get_chains())
    for i, chain in enumerate(all_chains):
        chain_seq = seq1(''.join(residue.resname for residue in chain))
        for o_chain in all_chains[i + 1:]:
            if chain_seq == seq1(''.join(residue.resname for residue in o_chain)):
                redundant_chains.append(o_chain.id)

    for redundant_chain in redundant_chains:
        structure[0].detach_child(redundant_chain)


def load_pdb_infos(pdb_id: str, path: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """ Generate the distance matrix for a defined pdb structure
    1. Load PDB File
    2. Remove redundant chains
    3. calculate distances and features
    Args:
        pdb_id: String with PDB ID
        path: Path to folder with PDB File
    Returns:
        np.ndarray: Matrix of all distances
        List: information about every amino acid (chain_id, aa_type, aa_number on chain)
    """

    path = os.path.join(path, pdb_id + ".pdb")
    structure, header =  read_file(pdb_id, path)

    return get_distances_and_info(structure, header)


def convert_aa_info(info: Dict, structure_info: Dict, chain_id2protein: Dict):
    """ Convert the information about amino acids to a feature matrix
    1. One-Hot Encode the Amino Acid Type
    2. Encode Chain ID as integer
    3. Encode Protein ID as integer
    4. Encode additional information about amio acid (charge, ... )
    Args:
        info:
        structure_info:
    Returns:
    """
    # TODO: Search for amino acid information (charge, ... )

    type_encoding = np.zeros(len(AMINO_ACIDS))
    type_encoding[AAA2ID[info["residue_type"].lower()]] = 1

    protein_encoding = np.zeros(2)
    if chain_id2protein[info["chain_id"].lower()] == "antibody":
        protein_encoding[0] = 1
    elif chain_id2protein[info["chain_id"].lower()] == "antigen":
        protein_encoding[1] = 1

    relative_residue_position = info["residue_id"] / structure_info["chain_length"][info["chain_id"].lower()]

    return np.concatenate([type_encoding, protein_encoding, np.array([relative_residue_position])])


def get_residue_encodings(residue_infos: List, structure_info: Dict, chain_id2protein: Dict) -> np.ndarray:
    """ Convert the residue infos to encodings in a numpy array
    Args:
        residue_infos: List of residue infos
        structure_info: Dict containing information about the structure
    Returns:
        np.ndarray: Array with numerical encodings
    """
    residue_encodings = []
    for res_info in residue_infos:
        residue_encodings.append(convert_aa_info(res_info, structure_info, chain_id2protein))
    return np.array(residue_encodings)


def get_edge_encodings(distance_matrix: np.ndarray, residue_infos: List, chain_id2protein: Dict, distance_cutoff: int = 10) -> np.ndarray:
    """ Convert the distance matrix and residue information to an adjacency tensor
    Information:
        A[0,:,:] = inverse pairwise distances - only below distance cutoff otherwise 0
        A[1,:,:] = neighboring amino acid - 1 if connected by peptide bond
        A[2,:,:] = same protein - 1 if same chain
    Args:
        distance_matrix:
        residue_infos:
        distance_cutoff:
    Returns:
    """
    A = np.zeros((3, len(residue_infos), len(residue_infos)))

    contact_map = distance_matrix < distance_cutoff

    A[0, contact_map] = 1 / (distance_matrix[contact_map] + 1e-9)
    for i, res_info in enumerate(residue_infos):
        for j, other_res_info in enumerate(residue_infos[i:]):
            j += i
            if j - 1 == i and res_info["chain_id"] == other_res_info["chain_id"]:
                A[1, i, j] = 1
                A[1, j, i] = 1

            if "chain_id" in res_info and chain_id2protein[res_info["chain_id"].lower()] == chain_id2protein[other_res_info["chain_id"].lower()]:
                A[2, i, j] = 1
                A[2, j, i] = 1
    return A


def antibody_chain(chain_molecule: str):
    antibody_defining_terms = ["fab", "ige", "antibody", "immunoglobulin", "heavy", "light", "anti", "igg"]
    for antibody_term in antibody_defining_terms:
        if antibody_term in chain_molecule:
            return True
    else:
        return False