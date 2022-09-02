from typing import Dict
import os
import pandas as pd
import string
from ast import literal_eval
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from abag_affinity.utils.pdb_processing import get_residue_encodings, \
    get_residue_edge_encodings, get_atom_edge_encodings, get_atom_encodings, get_residue_infos, get_distances
from abag_affinity.utils.pdb_reader import read_file

from abag_affinity.utils.affinity_conversion import clean_temp, calc_delta_g

alphabet_letters = set(string.ascii_lowercase)


def get_graph_dict(pdb_id: str, pdb_file_path: str, affinity: float, chain_id2protein: Dict,
                   node_type: str, distance_cutoff: int, ca_alpha_contact: bool = False) -> Dict:
    structure, header = read_file(pdb_id, pdb_file_path)

    if node_type == "residue":
        structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_residues = get_distances(residue_infos, residue_distance=True, ca_distance=ca_alpha_contact)

        node_features = get_residue_encodings(residue_infos, structure_info, chain_id2protein)
        adj_tensor = get_residue_edge_encodings(distances, residue_infos, chain_id2protein, distance_cutoff=distance_cutoff)
        atom_names = []
    elif node_type == "atom":
        structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_residues = get_distances(residue_infos, residue_distance=False)

        node_features, atom_names = get_atom_encodings(residue_infos, structure_info, chain_id2protein)
        adj_tensor = get_atom_edge_encodings(distances, node_features, distance_cutoff=distance_cutoff)
    else:
        raise ValueError("Invalid graph_type: Either 'residue' or 'atom'")


    assert len(residue_infos) > 0
    #assert node_features.shape[0] == len(residue_infos)
    #assert adj_tensor[0, :, :].shape == (len(residue_infos), len(residue_infos))

    return {
        "node_features":node_features,
        "residue_infos": residue_infos,
        "residue_atom_coordinates": residue_atom_coordinates,
        "adjacency_tensor": adj_tensor,
        "affinity": affinity,
        "closest_residues":closest_residues,
        "atom_names": atom_names
    }


def load_graph(row: pd.Series, dataset_type: str, config: Dict, node_type: str = "residue", distance_cutoff: int = 10,
               mutated_complex: str = "mutated_relaxed") -> Dict:
    if dataset_type == "Dataset_v1":
        pdb_id = row["pdb"]
        affinity = row["-log(Kd)"]
        resource_path = os.path.join(config["RESOURCES"]["path"], config["RESOURCES"]["Dataset_v1"]["folder_path"])
        pdb_file_path = os.path.join(resource_path, config["RESOURCES"]["Dataset_v1"]["pdb_path"], row["abdb_file"])
        chain_id2protein = {"l": "antibody",
                            "h": "antibody"}  # heavy and light chains of antibodies are often called l and h
        for chain_id in row["antibody_chains"]:
            chain_id2protein[chain_id.lower()] = "antibody"
        for chain_id in row["antigen_chains"]:
            chain_id2protein[chain_id.lower()] = "antigen"
        for letter in alphabet_letters - set(chain_id2protein.keys()):
            chain_id2protein[letter] = "antigen"

    elif dataset_type == "PDBBind":
        pdb_id = row["pdb"]
        affinity = row["-log(Kd)"]
        resource_path = os.path.join(config["RESOURCES"]["path"], config["RESOURCES"]["PDBBind"]["folder_path"])
        pdb_file_path = os.path.join(resource_path, config["RESOURCES"]["PDBBind"]["pdb_path"], row["pdb"] + ".ent.pdb")
        chain_id2protein = row["chain_infos"]
        if chain_id2protein[0] != "{":
            raise ValueError( "Not enough chain information available {}".format(pdb_id))
        chain_id2protein = literal_eval(chain_id2protein)
        # set first protein as "antibody"
        chain_id2protein = {chain: "antibody" if protein == 0 else "antigen" for chain, protein in chain_id2protein.items()}
    elif dataset_type == "SKEMPI.v2":
        pdb_id, ab_chains, ag_chains = row["#Pdb"].split("_")
        data_path = os.path.join(config["DATA"]["path"], config["DATA"]["SKEMPI.v2"]["folder_path"])
        #temperature = clean_temp(row["Temperature"])
        #if np.isnan(temperature):
        #    raise ValueError("Temperature not available for {}".format(pdb_id))
        if mutated_complex != "wildtype":
            mutation_code = row["Mutation(s)_cleaned"]
            pdb_file_path = os.path.join(data_path, mutated_complex, pdb_id, mutation_code + ".pdb")
            #affinity = calc_delta_g(temperature, row["Affinity_mut_parsed"])
            affinity = row["-log(Kd)_mut"]
        else:
            pdb_file_path = os.path.join(data_path, mutated_complex, pdb_id + ".pdb")
            #affinity = calc_delta_g(temperature, row["Affinity_wt_parsed"])
            affinity = row["-log(Kd)_wt"]


        chain_id2protein = {}
        for chain in ab_chains:
            chain_id2protein[chain.lower()] = "antibody"
        for chain in ag_chains:
            chain_id2protein[chain.lower()] = "antigen"

        if np.isnan(affinity):
            raise ValueError("Affinity not available for {}".format(pdb_id))
    else:
        raise ValueError("Invalid Dataset Type given (Dataset_v1, PDBBind, SKEMPI.v2")

    return get_graph_dict(pdb_id, pdb_file_path, affinity, chain_id2protein, distance_cutoff=distance_cutoff, node_type=node_type)


if __name__ == "__main__":
    residue_graph = get_graph_dict("4hwb", "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/resources/AbDb/NR_LH_Protein_Martin/4HWB_1.pdb", 0.1,
                   {'h':"antibody", 'l':"antibody", "a": "antigen"},  node_type="residue", distance_cutoff=10)
    residue_ca_graph = get_graph_dict("4hwb", "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/resources/AbDb/NR_LH_Protein_Martin/4HWB_1.pdb", 0.1,
                   {'h':"antibody", 'l':"antibody", "a": "antigen"},  node_type="residue", distance_cutoff=10, ca_alpha_contact=True)
    atom_graph = get_graph_dict("4hwb", "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/resources/AbDb/NR_LH_Protein_Martin/4HWB_1.pdb", 0.1,
                   {'h':"antibody", 'l':"antibody", "a": "antigen"}, node_type="atom", distance_cutoff=10)
    a = 0