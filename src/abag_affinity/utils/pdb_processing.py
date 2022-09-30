"""Process PDB file to get residue and atom encodings, node distances and edge encodings"""
import numpy as np
import scipy.spatial as sp
from Bio.SeqUtils import seq1
from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO
from typing import Tuple, List, Dict
from collections import defaultdict
import os
from pathlib import Path
import shutil
import subprocess
from biopandas.pdb import PandasPdb

from abag_affinity.utils.pdb_reader import read_file

from abag_affinity.binding_ddg_predictor.utils.protein import get_residue_pos14, ATOM_CA, augmented_three_to_index, \
    RESIDUE_SIDECHAIN_POSTFIXES, augmented_three_to_one, NON_STANDARD_SUBSTITUTIONS


# Definition of standard amino acids and objects to quickly access them
AMINO_ACIDS = ["ala","cys","asp","glu","phe","gly","his","ile","lys","leu","met","asn","pro","gln","arg","ser","thr","val","trp","tyr"]
AAA2ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID2AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Definition of Atom names and objects to quickly access them
ATOM_POSTFIXES = ['N', 'CA', 'C', 'O']
for postfixes in RESIDUE_SIDECHAIN_POSTFIXES.values():
    ATOM_POSTFIXES += list(postfixes)
ATOM2ID = {atom: i for i, atom in enumerate(ATOM_POSTFIXES)}
ID2ATOM = {i: atom for i, atom in enumerate(ATOM_POSTFIXES)}


def remove_redundant_chains(structure: Structure):
    """ Remove redundant chains from a structure (inplace)
    Redundant chains have exact amio-acid sequence identity
    Remove second chain found

    Args:
        structure: PDB Structure Object to be cleaned
    Returns:
        None
    """

    #TODO: Remove chains of only one protein and not of different
    redundant_chains = []

    # get all chains and mark redundant ones
    all_chains = list(structure.get_chains())
    for i, chain in enumerate(all_chains):
        chain_seq = seq1(''.join(residue.resname for residue in chain))
        for o_chain in all_chains[i + 1:]:
            if chain_seq == seq1(''.join(residue.resname for residue in o_chain)):
                redundant_chains.append((o_chain.id, o_chain.parent.id))

    for (redundant_chain, model) in redundant_chains:
        if redundant_chain in structure[model]:
            structure[model].detach_child(redundant_chain)


def convert_aa_info(info: Dict, structure_info: Dict, chain_id2protein: Dict) -> np.ndarray:
    """ Convert the information about amino acids to a feature matrix
    1. One-Hot Encode the Amino Acid Type
    2. Encode Chain ID as integer
    3. Encode Protein ID as integer
    4. Encode additional information about amio acid (charge, ... )

    Args:
        info: residue information (from get_residue_info)
        structure_info: structure information (from get_residue_info)
        chain_id2protein: Dict with protein information for each chain
    Returns:
        np.ndarray: numerical encoding of residue - shape: (23,)
    """
    # TODO: Get more amino acid information (charge, ... )

    type_encoding = np.zeros(len(AMINO_ACIDS))
    type_encoding[AAA2ID[info["residue_type"].lower()]] = 1

    protein_encoding = np.zeros(2)
    if chain_id2protein[info["chain_id"].lower()] == 0:
        protein_encoding[0] = 1
    elif chain_id2protein[info["chain_id"].lower()] == 1:
        protein_encoding[1] = 1

    relative_residue_position = info["residue_id"] / structure_info["chain_length"][info["chain_id"].lower()]

    return np.concatenate([type_encoding, protein_encoding, np.array([relative_residue_position])])


def get_residue_infos(structure: Structure, chain_id2protein: Dict) -> Tuple[Dict, List[Dict], np.ndarray]:
    """ Calculate all distances of the amino acids in this structure
    1. Remove duplicate chains
    2. Get position of amino acid C-alpha atoms + Chain Id, AA type, AA number
    3. Calculate pairwise distances

    Args:
        structure: PDB Structure Object to be used for distance calculation
        chain_id2protein: Dict with protein information for each chain
    Returns:
        np.ndarray: Matrix of all distances
        List: information about every amino acid (chain_id, aa_type, aa_number on chain)
        np.ndarray: Atom coordinates for every residue
    """

    remove_redundant_chains(structure)

    structure_info = {
        "chain_length": defaultdict(int)
    }
    residue_infos = [] # (chain id, residue type, reside id)
    residue_atom_coordinates = []
    residue_idx = 0
    chain_idx = 1
    on_chain_residue_idx = 1
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            # get atom coordinates
            try:
                all_atom_coordinates = get_residue_pos14(residue).numpy()
                assert np.isfinite(all_atom_coordinates[0:3,:]).all()
            except Exception as e:
                continue # filter HETATOMs
            residue_atom_coordinates.append(all_atom_coordinates)

            # convert residue type to one of the standard residues
            residue_type = residue.resname
            if residue_type.lower() not in AMINO_ACIDS:
                residue_type = NON_STANDARD_SUBSTITUTIONS[residue.resname.upper()]

            # extract names of atoms (for atom encodings)
            atom_names = [atom.get_name() for atom in residue.get_atoms()]

            residue_info = {
                "chain_id": chain.id,
                "residue_type": residue_type.lower(),
                "residue_id": residue.get_id()[1],
                "residue_pdb_index": augmented_three_to_index(residue.resname),
                "chain_idx": chain_idx,
                "on_chain_residue_idx": on_chain_residue_idx,
                "all_atom_coordinates": all_atom_coordinates,
                "antibody": chain_id2protein[chain.id.lower()],
                "atom_names": atom_names
            }

            residue_infos.append(residue_info)
            structure_info["chain_length"][chain.id.lower()] += 1
            residue_idx += 1
            on_chain_residue_idx += 1
        on_chain_residue_idx = 1
        chain_idx += 1

    residue_atom_coordinates = np.array(residue_atom_coordinates)
    return structure_info, residue_infos, residue_atom_coordinates


def get_distances(residue_info: List[Dict], residue_distance: bool = True, ca_distance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    coordinates = []
    antibody_idx = []
    antigen_idx = []

    node_idx = 0
    for residue in residue_info:
        residue_coordinates = residue["all_atom_coordinates"]

        if residue_distance: # get residues and their coordinates
            if ca_distance: # only use C-alpha atom
                coordinates.append(residue_coordinates[ATOM_CA])
            else: # save all atom coordinates
                coordinates.append(residue_coordinates)
            if residue["antibody"]:
                antibody_idx.append(node_idx)
            else:
                antigen_idx.append(node_idx)
            node_idx += 1
        else: # get atoms and their coordinates
            for atom_idx, atom_coordinates in enumerate(residue_coordinates):
                if np.isinf(atom_coordinates).any():
                    continue
                coordinates.append(atom_coordinates)
                if residue["antibody"]:
                    antibody_idx.append(node_idx)
                else:
                    antigen_idx.append(node_idx)
                node_idx += 1

    coordinates = np.array(coordinates)
    antibody_idx = np.array(antibody_idx).astype(np.int)
    antigen_idx = np.array(antigen_idx).astype(np.int)

    if ca_distance or not residue_distance:
        distances = sp.distance_matrix(coordinates, coordinates)
    else:
        all_distances = []
        for i in range(coordinates.shape[1]):
            for j in range(coordinates.shape[1]):
                all_distances.append(
                    sp.distance_matrix(coordinates[:, i, :], coordinates[:, j, :]))

        all_distances = np.array(all_distances)
        distances = np.nanmin(all_distances, axis=0)

    # select antibody-antigen distances (or protein1-protein2)
    abag_distance = distances[antibody_idx, :][:, antigen_idx]
    abag_indices = np.unravel_index(np.argsort(abag_distance.ravel()), abag_distance.shape)
    # convert to structure node ids
    antibody_indices = antibody_idx[abag_indices[0]]
    antigen_indices = antigen_idx[abag_indices[1]]

    # sort by closeness
    closest_residue_indices = np.empty((antibody_indices.size + antigen_indices.size), dtype=antigen_indices.dtype)
    closest_residue_indices[0::2] = antibody_indices
    closest_residue_indices[1::2] = antigen_indices
    _, idx = np.unique(closest_residue_indices, return_index=True)
    closest_residue_indices = closest_residue_indices[np.sort(idx)]

    return distances, closest_residue_indices


def get_residue_encodings(residue_infos: List, structure_info: Dict, chain_id2protein: Dict) -> np.ndarray:
    """ Convert the residue infos to numerical encodings in a numpy array
    Args:
        residue_infos: List of residue infos
        structure_info: Dict containing information about the structure
        chain_id2protein: Dict with protein information for each chain

    Returns:
        np.ndarray: Array with numerical encodings - shape (n, 23)
    """
    residue_encodings = []
    for res_info in residue_infos:
        residue_encodings.append(convert_aa_info(res_info, structure_info, chain_id2protein))
    return np.array(residue_encodings)


def get_residue_edge_encodings(distance_matrix: np.ndarray, residue_infos: List, chain_id2protein: Dict, distance_cutoff: int = 5) -> np.ndarray:
    """ Convert the distance matrix and residue information to an adjacency tensor
    Information:
        A[0,:,:] = inverse pairwise distances - only below distance cutoff otherwise 0
        A[1,:,:] = neighboring amino acid - 1 if connected by peptide bond
        A[2,:,:] = same protein - 1 if same chain
        A[3,:,:] = distances


    Args:
        distance_matrix: matrix with node distances
        residue_infos: List with residue information dicts
        distance_cutoff: distance cutoff for interface
    Returns:
        np.ndarray: Tensor edge information about interface closeness, chain identity, protein identity, distances
    """
    A = np.zeros((4, len(residue_infos), len(residue_infos)))

    contact_map = distance_matrix < distance_cutoff

    A[0, contact_map] = 1 / (distance_matrix[contact_map] + 1)
    for i, res_info in enumerate(residue_infos):
        for j, other_res_info in enumerate(residue_infos[i:]):
            j += i
            if j - 1 == i and res_info["chain_id"] == other_res_info["chain_id"]:
                A[1, i, j] = 1
                A[1, j, i] = 1

            if "chain_id" in res_info and chain_id2protein[res_info["chain_id"].lower()] == chain_id2protein[other_res_info["chain_id"].lower()]:
                A[2, i, j] = 1
                A[2, j, i] = 1

    A[3, :, :] = distance_matrix

    return A


def get_atom_encodings(residue_infos: List[Dict], structure_info: Dict, chain_id2protein: Dict):
    """ Convert the atom infos to numerical encodings in a numpy array

    Args:
        residue_infos: List of residue infos
        structure_info: Dict containing information about the structure
        chain_id2protein: Dict with protein information for each chain

    Returns:
        np.ndarray: Array with numerical encodings - shape (n, 115)
    """
    atom_encoding = []
    atom_names = []
    for res_idx, res_info in enumerate(residue_infos):
        res_encoding = convert_aa_info(res_info, structure_info, chain_id2protein)
        atom_order = ['N', 'CA', 'C', 'O'] + RESIDUE_SIDECHAIN_POSTFIXES[augmented_three_to_one(res_info["residue_type"].upper())]
        atom_idx = 0
        for atom_coords in res_info["all_atom_coordinates"]:
            if not np.isinf(atom_coords).any():
                atom_type = np.zeros(len(ATOM_POSTFIXES))
                atom_type[ATOM2ID[atom_order[atom_idx]]] = 1
                atom_info = np.concatenate([res_encoding, atom_type, np.array([res_idx])])
                atom_encoding.append(atom_info)
                atom_names.append(res_info["atom_names"][atom_idx])
                atom_idx += 1
    return np.array(atom_encoding), atom_names # (residue_type, protein_type, relative_chain_position, atom_type, residue_index)


def get_atom_edge_encodings(distance_matrix: np.ndarray, atom_encodings: np.ndarray, distance_cutoff: int = 5) -> np.ndarray:
    """ Convert the distance matrix and atom information to an adjacency tensor
    Information:
        A[0,:,:] = inverse pairwise distances - only if distance is below cutoff otherwise 0
        A[1,:,:] = same residue - 1 if belonging to same residue
        A[2,:,:] = same protein - 1 if same chain
        A[3,:,:] = distances


    Args:
        distance_matrix: matrix with node distances
        atom_encodings: matrix with atom encodings
        distance_cutoff: distance cutoff for interface

    Returns:
        np.ndarray: Tensor edge information about interface closeness, chain identity, protein identity, distances
    """
    A = np.zeros((4, len(atom_encodings), len(atom_encodings))) # (contact, same_residue, same_protein, distance)

    contact_map = distance_matrix < distance_cutoff

    A[0, contact_map] = 1 / (distance_matrix[contact_map] + 1)

    # same residue index
    A[1, :, :] = (atom_encodings[:, -1, None] == atom_encodings[:, -1]).astype(float)
    # same protein
    A[2, :, :] = (atom_encodings[:, 20, None] == atom_encodings[:, 20]).astype(float)

    A[3, :, :] = distance_matrix
    return A


def clean_and_tidy_pdb(pdb_id: str, pdb_file_path: str, cleaned_file_path:str) -> str:
    Path(cleaned_file_path).parent.mkdir(exist_ok=True, parents=True)

    tmp_pdb_filepath = f'{pdb_file_path}.tmp'
    shutil.copyfile(pdb_file_path, tmp_pdb_filepath)
    # Clean temporary PDB file and then save its cleaned version as the original PDB file
    args = ['pdb_tidy', tmp_pdb_filepath]
    with open(cleaned_file_path, 'w') as outfile:
        p1 = subprocess.run(args=args, stdout=outfile)

    cleaned_pdb = PandasPdb().read_pdb(cleaned_file_path)
    input_atom_df = cleaned_pdb.df['ATOM']

    # remove all duplicate (alternate location residues)
    filtered_df = input_atom_df.drop_duplicates(subset=["atom_name", "chain_id", "residue_number"])

    # remove all residues that do not have at least N, CA, C atoms
    filtered_df = filtered_df.groupby(["chain_id", "residue_number", "residue_name"]).filter(
        lambda x: x["atom_name"].values[:3].tolist() == ["N", "CA", "C"])

    # drop H atoms
    filtered_df = filtered_df[filtered_df['element_symbol'] != 'H']


    cleaned_pdb.df['ATOM'] = filtered_df.reset_index(drop=True)
    cleaned_pdb.to_pdb(path=cleaned_file_path,
                       records=["ATOM"],
                       gz=False,
                       append_newline=True)

    # Clean up from using temporary PDB file for tidying
    os.remove(tmp_pdb_filepath)
