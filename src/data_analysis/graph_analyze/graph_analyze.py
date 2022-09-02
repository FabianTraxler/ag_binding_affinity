import os.path
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from abag_affinity.dataset.data_loader import AffinityDataset
from abag_affinity.utils.config import read_yaml, get_resources_paths
from abag_affinity.utils.pdb_processing import get_residue_infos, get_distances, get_residue_edge_encodings, \
    get_atom_edge_encodings, get_atom_encodings
from abag_affinity.utils.pdb_reader import read_file

def write_selection_for_interface(pdb_id: str, cutoff: int = None, n_closest: int = None, use_dataloader: bool = False,
                           pdb_path: str = None, chain_id2protein: Dict = None):

    if use_dataloader:
        config = read_yaml("../../abag_affinity/config.yaml")
        dataset = AffinityDataset(config, "Dataset_v1", pdb_ids=[pdb_id], max_nodes=n_closest, node_type="atom", load_from_disc=False, save_graphs=False)

        if cutoff is None:
            graph_dict = dataset.get_graph_dict(pdb_id)
        else:
            graph_dict = dataset.get_graph_dict(pdb_id, distance_cutoff=cutoff)

        if n_closest is None:
            interface = graph_dict["adjacency_tensor"][0,:,:] - graph_dict["adjacency_tensor"][2,:,:]
            atoms = np.where(interface > 0.001)[0]
            atoms = np.unique(atoms)

        else:
            atoms = graph_dict["closest_residues"][:n_closest]

        node_infos = graph_dict["residue_infos"]

    elif pdb_path is not None and chain_id2protein is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        atom_encoding, atom_names = get_atom_encodings(node_infos, structure_info, chain_id2protein)
        A = get_atom_edge_encodings(distances, atom_encoding, cutoff)
        if n_closest is None:
            interface = A[0, :, :] - A[2, :, :]
            atoms = np.where(interface > 0.001)[0]
            atoms = np.unique(atoms)

        else:
            atoms = closest_atoms[:n_closest]

        graph_dict = None
    else:
        print("Please select 'use_dataloader' or provide path and info_dict")
        return "", None
    selection_string = f"select {pdb_id}_interface, "
    for atom in atoms:
        res_index = int(atom_encoding[atom][-1])
        atom_name = atom_names[atom]
        node_info = node_infos[res_index]
        chain_id = node_info["chain_id"]
        residue_id = node_info["residue_id"]

        selection_string += f"({pdb_id} and chain {chain_id} and resi {residue_id} and name {atom_name}) OR "

    selection_string = selection_string[:-4]

    #print(selection_string)
    return selection_string, graph_dict


def write_connection_lines(pdb_id: str, cutoff: int = 5, graph_dict = None, use_dataloader: bool = False,
                           pdb_path: str = None, chain_id2protein: Dict = None):
    if graph_dict is None and use_dataloader:
        config = read_yaml("../../abag_affinity/config.yaml")
        dataset = AffinityDataset(config, "Dataset_v1", pdb_ids=[pdb_id], load_from_disc=False, save_graphs=False)

        graph_dict = dataset.get_graph_dict(pdb_id, distance_cutoff=cutoff)
        interface = graph_dict["adjacency_tensor"][0, :, :] - graph_dict["adjacency_tensor"][2, :, :]

    elif graph_dict is None and pdb_path is not None and chain_id2protein is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_residues = get_distances(residue_infos, residue_distance=False)
        A = get_residue_edge_encodings(distances, residue_infos, chain_id2protein, cutoff)
        interface = A[0, :, :]  - A[2, :, :]
    else:
        interface = graph_dict["adjacency_tensor"][0, :, :] - graph_dict["adjacency_tensor"][2, :, :]


    interface_residues = np.where(interface > 0.001)
    interface_distances = interface[interface_residues[0], interface_residues[1]]
    closest_residues = np.argsort(interface_distances )[::-1]
    node_infos = graph_dict["residue_infos"]
    selection_string = ""
    for i, closest in enumerate(closest_residues):
        residue1 = interface_residues[0][closest]
        residue2 = interface_residues[1][closest]
        chain_id_1 = node_infos[residue1]["chain_id"]
        residue_id_1 = node_infos[residue1]["residue_id"]

        chain_id_2 = node_infos[residue2]["chain_id"]
        residue_id_2 = node_infos[residue2]["residue_id"]

        selection_string += f"distance {pdb_id}_interaction, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1} " \
                            f"and name CA) , ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name CA),100, 4 \n "

    #print(selection_string)
    return selection_string


def write_atom_connection_lines(pdb_id: str, cutoff: int = 5):
    config = read_yaml("../../abag_affinity/config.yaml")
    dataset = AffinityDataset(config, "Dataset_v1", pdb_ids=[pdb_id], load_from_disc=False, save_graphs=False, node_type="atom")

    graph_dict = dataset.get_graph_dict(pdb_id, distance_cutoff=cutoff)

    interface = graph_dict["adjacency_tensor"][0, :, :] - graph_dict["adjacency_tensor"][2, :, :]
    interface_atoms = np.where(interface > 0.001)
    interface_distances = interface[interface_atoms[0], interface_atoms[1]]
    closest_residues = np.argsort(interface_distances )[::-1]
    node_infos = graph_dict["residue_infos"]
    selection_string = ""
    for i, closest in enumerate(closest_residues):
        atom1 = interface_atoms[0][closest]
        atom2 = interface_atoms[1][closest]

        #atom1_type = ID2ATOM[int(np.where(graph_dict["node_features"][atom1][23:-1] == 1)[0])]
        atom1_type = graph_dict["atom_names"][atom1]
        atom1_residue = int(graph_dict["node_features"][atom1][-1])
        chain_id_1 = node_infos[atom1_residue]["chain_id"]
        residue_id_1 = node_infos[atom1_residue]["residue_id"]

        atom2_type = graph_dict["atom_names"][atom2]
        atom2_residue = int(graph_dict["node_features"][atom2][-1])

        chain_id_2 = node_infos[atom2_residue]["chain_id"]
        residue_id_2 = node_infos[atom2_residue]["residue_id"]

        selection_string += f"distance {pdb_id}_interaction, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1} " \
                            f"and name {atom1_type}), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name {atom2_type}),100, 4 \n "

    #print(selection_string)
    return selection_string


def load_pdb(pdb_id: str, file_path: str = None):
    if file_path is None:
        config = read_yaml("../../abag_affinity/config.yaml")
        dataset = AffinityDataset(config, "Dataset_v1", pdb_ids=[pdb_id], load_from_disc=False, save_graphs=False)
        _, pdb_path = get_resources_paths(config, "Dataset_v1")
        file_name = dataset.data_df[dataset.data_df["pdb"] == pdb_id]["abdb_file"].values[0]
        file_path = os.path.join(pdb_path, file_name)


    pymol_commands = 'hide all\n'
    pymol_commands += f'load {file_path}, {pdb_id}\n'
    pymol_commands += "spectrum chain, blue red green\n"

    return pymol_commands


def start_pymol(pymol_path: str):
    from utils import PyMol
    pm = PyMol(pymol_path)

    return pm


def load_pdb_into_pymol(pm, pdb_id: str, cutoff: int = 5):
    print("Load PDB ...")
    command = load_pdb(pdb_id)
    pm(command)

    print("Select Interface ...")
    selection, graph_dict = write_selection_for_interface(pdb_id, cutoff=cutoff)
    selection += f"\ncolor magenta, {pdb_id}_interface\n"
    pm(selection)

    print("Load interaction ...")
    #lines = write_connection_lines(pdb_id, cutoff, graph_dict)
    lines = write_atom_connection_lines(pdb_id, cutoff)
    pm(lines)
    print("All done!")

    return pm


def select_residues(pdb_id:str, residues: List, name: str = "selected_Residues"):
    selection_string = f"select {name},"

    for (chain, res_idx) in residues:
        selection_string += f" ({pdb_id} AND chain {chain} AND resi {res_idx}) OR"

    selection_string = selection_string[:-3]

    return selection_string

def main():
    input_string = ""
    pm = start_pymol("/home/fabian/Downloads/pymol/bin")

    print("Welcome to the graph visualizer")
    while input_string != "stop":
        input_string = input("Please provide a PDB ID or type 'stop' to end the program: ")
        if input_string == "stop":
            break
        print(f"Loading graph for {input_string}")
        try:
            pm = load_pdb_into_pymol(pm, input_string.lower())
        except:
            print("Invalid PDB Id or not in dataset")



def dms_analysis():
    pdb_id = "cr9114" # "3sdy"
    file_path = "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/resources/DMS/pdbs/cr9114_newcal1999_processed_relaxed.pdb"
    load_command = load_pdb(pdb_id, file_path)
    chain2protein = {"a": "antigen", "b": "antigen", "c": "antibody"}
    selection, graph_dict = write_selection_for_interface(pdb_id, cutoff=5, pdb_path=file_path, chain_id2protein=chain2protein)

    #atom_connection = write_atom_connection_lines(pdb_id)
    #residue_connection = write_connection_lines(pdb_id)
    mutations = [('C', '209'), ('C', '210'), ('C', '165'), ('C', '187'), ('C', '241'), ('C', '194'), ('C', '212'),
                 ('C', '193'), ('C', '164'), ('C', '211'), ('C', '222'), ('C', '166'), ('C', '230'), ('C', '219'),
                 ('C', '192'), ('C', '206')]

    mutation_select = select_residues(pdb_id, mutations, "Mutations")

    expanded_interface_5 = "select expanded_5"  + selection[selection.find("_interface"):] + " expand 5"
    expaned_interface_10 = "select expanded_10"  + selection[selection.find("_interface"):] + " expand 10"
    return


if __name__ == "__main__":
    dms_analysis()
    #main()