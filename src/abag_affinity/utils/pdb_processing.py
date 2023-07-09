"""Process PDB file to get residue and atom encodings, node distances and edge encodings"""
import tempfile
from collections import defaultdict, deque
import logging
import os
import re
import shutil
import string
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import scipy.spatial as sp
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
from biopandas.pdb import PandasPdb

from abag_affinity.binding_ddg_predictor.utils.protein import (
    ATOM_CA, NON_STANDARD_SUBSTITUTIONS, RESIDUE_SIDECHAIN_POSTFIXES,
    augmented_three_to_index, augmented_three_to_one, get_residue_pos14)
from .pdb_reader import read_file
from .feature_extraction import residue_features, atom_features

# Definition of standard amino acids and objects to quickly access them (order corresponds to AF2/OF)
AMINO_ACIDS = ["ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his", "ile", "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val"]

AAA2ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID2AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Definition of Atom names and objects to quickly access them
ATOM_POSTFIXES = ['N', 'CA', 'C', 'O']
for postfixes in RESIDUE_SIDECHAIN_POSTFIXES.values():
    ATOM_POSTFIXES += list(postfixes)
ATOM_POSTFIXES = set(ATOM_POSTFIXES)

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


def convert_aa_info(info: Dict, structure_info: Dict) -> np.ndarray:
    """ Convert the information about amino acids to a feature matrix
    1. One-Hot Encode the Amino Acid Type
    2. Encode Chain ID as integer
    3. Encode Protein ID as integer
    4. Encode additional information about amio acid (charge, ... )

    Args:
        info: residue information (from get_residue_info)
        structure_info: structure information (from get_residue_info)
    Returns:
        np.ndarray: numerical encoding of residue - shape: (23,)
    """

    type_encoding = np.zeros(len(AMINO_ACIDS))
    type_encoding[AAA2ID[info["residue_type"].lower()]] = 1

    protein_encoding = np.zeros(2)
    if info["chain_id"].upper() in "HL":
        protein_encoding[0] = 1
    else:  # info["chain_id"].upper() not in "HL"
        protein_encoding[1] = 1

    relative_residue_position = info["residue_id"] / structure_info["chain_length"][info["chain_id"].lower()]

    manual_features = residue_features(augmented_three_to_one(info["residue_type"].upper()))
    manual_features = np.array(manual_features)

    return np.concatenate([type_encoding, protein_encoding, manual_features, np.array([relative_residue_position])])


def get_residue_infos(structure: Structure) -> Tuple[Dict, List[Dict], np.ndarray]:
    """ Calculate all distances of the amino acids in this structure
    1. Remove duplicate chains
    2. Get position of amino acid C-alpha atoms + Chain Id, AA type, AA number
    3. Calculate pairwise distances

    Args:
        structure: PDB Structure Object to be used for distance calculation
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
                all_atom_coordinates, atom_names = get_residue_pos14(residue)
                all_atom_coordinates = all_atom_coordinates.numpy()
                assert np.isfinite(all_atom_coordinates[0:3,:]).all()
            except Exception as e:
                continue # filter HETATOMs
            residue_atom_coordinates.append(all_atom_coordinates)

            # convert residue type to one of the standard residues
            residue_type = residue.resname
            if residue_type.lower() not in AMINO_ACIDS:
                residue_type = NON_STANDARD_SUBSTITUTIONS[residue.resname.upper()]

            # extract names of atoms (for atom encodings)
            if atom_names != [atom.get_name() for atom in residue.get_atoms()]:
                logging.info("Atom names are not in the expected order (in provided PDB)!")

            antibody = (chain.id.upper() in 'LH')

            residue_info = {
                "chain_id": chain.id,
                "residue_type": residue_type.lower(),
                "residue_id": residue.get_id()[1],
                "residue_pdb_index": augmented_three_to_index(residue.resname),
                "chain_idx": chain_idx,
                "on_chain_residue_idx": on_chain_residue_idx,
                "all_atom_coordinates": all_atom_coordinates,
                "antibody": antibody,
                "atom_names": atom_names  # TODO this might be wrong in some cases (probably with broken PDB)! Here, CB is BEFORE 'O': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'ND2', 'OD1']. In contrast, get_residue_pos14 returns order: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND2', 'OD1']
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
        else: # get all atoms and their coordinates
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
    antibody_idx = np.array(antibody_idx).astype(int)
    antigen_idx = np.array(antigen_idx).astype(int)

    if ca_distance or not residue_distance:
        distances = sp.distance_matrix(coordinates, coordinates)
    else:
        atom_per_residue = coordinates.shape[1]
        coordinates_flattend = coordinates.reshape((len(coordinates) * atom_per_residue, 3))

        distances = sp.distance_matrix(coordinates_flattend, coordinates_flattend)

        new_shape = (len(coordinates), atom_per_residue, len(coordinates), atom_per_residue)
        distances = distances.reshape(new_shape)
        distances = np.nanmin(distances, axis=(1, 3))

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


def get_residue_encodings(residue_infos: List, structure_info: Dict) -> np.ndarray:
    """ Convert the residue infos to numerical encodings in a numpy array
    Args:
        residue_infos: List of residue infos
        structure_info: Dict containing information about the structure

    Returns:
        np.ndarray: Array with numerical encodings - shape (n, 23)
    """
    residue_encodings = []
    for res_info in residue_infos:
        aa_protein_chain_encoding = convert_aa_info(res_info, structure_info)
        residue_encodings.append(aa_protein_chain_encoding)
    return np.array(residue_encodings)


def get_residue_edge_encodings(distance_matrix: np.ndarray, residue_infos: List, distance_cutoff: int = 5) -> np.ndarray:
    """
    Convert the distance matrix and residue information to an adjacency tensor
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

    # scale distances
    A[0, contact_map] = distance_matrix[contact_map] / distance_cutoff

    # Test whether "L" or "H" in residue_infos["chain_id"]
    lh_present = np.any(["chain_id" in res_info and res_info["chain_id"].upper() in "LH" for res_info in residue_infos])

    for i, res_info in enumerate(residue_infos):
        for j, other_res_info in enumerate(residue_infos[i:]):
            j += i
            if j - 1 == i and res_info["chain_id"] == other_res_info["chain_id"]:
                A[1, i, j] = 1
                A[1, j, i] = 1

            if lh_present:
                if "chain_id" in res_info and (res_info["chain_id"].upper() in "LH") == (other_res_info["chain_id"].upper() in "LH"):
                    A[2, i, j] = 1
                    A[2, j, i] = 1
            else:
                if "chain_id" in res_info and res_info["chain_id"].upper() == other_res_info["chain_id"].upper():
                    A[2, i, j] = 1
                    A[2, j, i] = 1

    A[3, :, :] = distance_matrix

    return A


def get_atom_encodings(residue_infos: List[Dict], structure_info: Dict):
    """ Convert the atom infos to numerical encodings in a numpy array

    Args:
        residue_infos: List of residue infos
        structure_info: Dict containing information about the structure

    Returns:
        np.ndarray: Array with numerical encodings - shape (n, 115)
    """
    from rdkit import Chem
    atom_encoding = []
    atom_names = []
    for res_idx, res_info in enumerate(residue_infos):
        res_encoding = convert_aa_info(res_info, structure_info)

        atom_order = ['N', 'CA', 'C', 'O'] + RESIDUE_SIDECHAIN_POSTFIXES[augmented_three_to_one(res_info["residue_type"].upper())]

        # get rdkit features for sequence
        seq = ""
        has_aa_before = False
        has_aa_after = False
        if res_idx > 0:
            seq += augmented_three_to_one(residue_infos[res_idx - 1]["residue_type"].upper())
            has_aa_before = True
        seq += augmented_three_to_one(res_info["residue_type"].upper())
        if res_idx < len(residue_infos) - 1:
            seq += augmented_three_to_one(residue_infos[res_idx + 1]["residue_type"].upper())
            has_aa_after = True
        mol = Chem.MolFromSequence(seq)
        atoms = [ atom for atom in mol.GetAtoms()]
        atom_types = "".join([ atom.GetSymbol() for atom in atoms ])
        res_start_atoms = [ x.start() for x in re.finditer("NCCO", atom_types)]
        if has_aa_before:
            atoms = atoms[res_start_atoms[1]:]
            atom_types = atom_types[res_start_atoms[1]:]
        if has_aa_after:
            atoms = atoms[:res_start_atoms[-1]]
            atom_types = atom_types[:res_start_atoms[-1]]

        atom_idx = 0
        for atom_coords in res_info["all_atom_coordinates"]:
            if not np.isinf(atom_coords).any():
                atom_type = np.zeros(len(ATOM_POSTFIXES))
                atom_type[ATOM2ID[atom_order[atom_idx]]] = 1
                if atom_types[atom_idx] == res_info["atom_names"][atom_idx][0]:
                    atom_feats = atom_features(atoms[atom_idx]).astype(np.float)
                else:
                    atom_feats = np.zeros(13)
                    #print("Atom not found")
                atom_info = np.concatenate([atom_feats, atom_type, res_encoding])
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

    # scale distances
    A[0, contact_map] = distance_matrix[contact_map] / distance_cutoff # 1 / (distance_matrix[contact_map] + 1)

    # same residue index
    A[1, :, :] = (atom_encodings[:, -1, None] == atom_encodings[:, -1]).astype(float)
    # same protein
    A[2, :, :] = (atom_encodings[:, 54, None] == atom_encodings[:, 54]).astype(float)

    A[3, :, :] = distance_matrix
    return A



def order_substitutions(substitutions):
    """
    Order substiutions to avoid chain overlaps (and thereby loss of chain information)
    """
    # Create a dependency graph with nodes as keys and values
    graph = defaultdict(list)
    for src, dest in substitutions.items():
        graph[src].append(dest)

    # Perform a topological sorting on the graph
    sorted_nodes = []
    visited = set()
    stack = deque()

    def visit(node):
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                visit(neighbor)
            stack.appendleft(node)

    for node in list(graph.keys()):
        visit(node)

    # Apply substitutions in the sorted order
    result = {}
    for node in reversed(stack):
        if node in substitutions:
            result[node] = substitutions[node]

    return result


def clean_and_tidy_pdb(pdb_id: str, pdb_file_path: Union[str, Path], cleaned_file_path: Union[str, Path], fix_insert=True, rename_chains: Optional[Dict] = None):
    Path(cleaned_file_path).parent.mkdir(exist_ok=True, parents=True)

    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
        tmp_pdb_filepath = tmp_file.name

    shutil.copyfile(pdb_file_path, tmp_pdb_filepath)

    # remove additional models - only keep first model
    structure, _ = read_file(pdb_id, tmp_pdb_filepath)
    model = structure[0]
    io = PDBIO()
    io.set_structure(model)
    io.save(tmp_pdb_filepath)

    # identify antigen chain
    if rename_chains is None:
        # check whether the PDB file header includes chain mapping
        df_map = pdb_chain_mapping(pdb_file_path)
        if df_map is not None:
            antigen_chains = sorted(df_map[df_map["type"] == "A"]["abdb_label"].values)
            rename_chains = dict(zip(antigen_chains, string.ascii_uppercase))

    rename_fixinsert_command = ""
    if rename_chains is not None:
        # order chain ids to avoid overlaps
        rename_chains = order_substitutions(rename_chains)

        # assign chains ids from A to Z. I checked that this will not lead to conflicts with the original chain ids (in case of 2 or more antigen chains)
        for chain, new_id in rename_chains.items():
            rename_fixinsert_command += f" | pdb_rplchain -{chain}:{new_id}"

    if fix_insert:
        rename_fixinsert_command += " | pdb_fixinsert "

    # Clean temporary PDB file and then save its cleaned version as the original PDB file
    # retry 3 times because these commands sometimes do not properly write to disc
    retries = 0
    while not os.path.exists(cleaned_file_path):
        command = f'pdb_sort "{tmp_pdb_filepath}" | pdb_tidy {rename_fixinsert_command} | pdb_delhetatm  > "{cleaned_file_path}"'
        subprocess.run(command, shell=True)
        retries += 1
        if retries >= 3:
            raise RuntimeError(f"Error in PDB Utils commands to clean PDB {tmp_pdb_filepath}")

    cleaned_pdb = PandasPdb().read_pdb(cleaned_file_path)
    input_atom_df = cleaned_pdb.df['ATOM']

    # remove all duplicate (alternate location residues)
    filtered_df = input_atom_df.drop_duplicates(subset=["atom_name", "chain_id", "residue_number", "insertion"])

    # remove all residues that do not have at least N, CA, C atoms
    filtered_df = filtered_df.groupby(["chain_id", "residue_number", "residue_name"]).filter(
        lambda x: x["atom_name"].values[:3].tolist() == ["N", "CA", "C"])

    # drop H atoms
    filtered_df = filtered_df[filtered_df['element_symbol'] != 'H']

    # remove all non-standard atoms - used in Binding_DDG preprocessing
    all_postfixes = [ "" ]
    for postfixes in RESIDUE_SIDECHAIN_POSTFIXES.values():
        all_postfixes += postfixes
    atom_name_postfix = filtered_df['atom_name'].apply(get_atom_postfixes)
    filtered_df = filtered_df[atom_name_postfix.isin(all_postfixes)]

    assert len(filtered_df) > 0, f"No atoms in pdb file after cleaning: {pdb_file_path}"

    cleaned_pdb.df['ATOM'] = filtered_df.reset_index(drop=True)

    cleaned_pdb.to_pdb(path=str(cleaned_file_path),
                       records=["ATOM"],
                       gz=False,
                       append_newline=True)

    # Clean up from using temporary PDB file for tidying
    if os.path.exists(tmp_pdb_filepath):
        try:
            os.remove(tmp_pdb_filepath)
        except FileNotFoundError:
            pass # file has already been removed


def get_atom_postfixes(atom_name: str):
    # very similar to binding_ddg preprocessing
    if atom_name in ('N', 'CA', 'C', 'O'):
        return ""
    if atom_name[-1].isnumeric():
        return atom_name[-2:]
    else:
        return atom_name[-1:]

def pdb_chain_mapping(pdb_file: Union[str, Path]) -> pd.DataFrame:
    """
    Return the chain mapping as provided by AbDb
    """
    mapping = []

    with open(pdb_file) as f:
        for l in f:
            if l.startswith("REMARK 950 CHAIN "):
                mapping.append(l.replace("REMARK 950 CHAIN ", "").split())
            elif len(mapping) > 0:
                break
        else:
            logging.debug("pdb_file did not contain chain mapping.")
            return  None
            # mapping = [["L", "L", "L"], ["H", "H", "H"], ["A", "A", "A"]]
    df = pd.DataFrame(data=mapping, columns=("type", "abdb_label", "original_label"))
    if "1ZV5" in str(pdb_file):  # fix error in dataset
        df.loc[df["abdb_label"] == "L", "abdb_label"] = "l"
    return df

