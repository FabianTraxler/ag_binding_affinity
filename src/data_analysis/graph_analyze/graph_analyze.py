import os.path
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")
import numpy as np
from abag_affinity.dataset import BoundComplexGraphs
from abag_affinity.utils.config import get_resources_paths, read_config
from abag_affinity.utils.pdb_processing import (get_atom_edge_encodings,
                                                get_atom_encodings,
                                                get_distances,
                                                get_residue_edge_encodings,
                                                get_residue_encodings,
                                                get_residue_infos)
from abag_affinity.utils.pdb_reader import read_file


def write_selection_for_residue_interface_hull(pdb_id: str, interface_cutoff: int = 5, hull_size:int = 7, use_dataloader: bool = False,
                                  pdb_path: str = None, chain_id2protein: Dict = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = BoundComplexGraphs(config, "Dataset_v1", pdb_ids=[pdb_id],
                                     interface_distance_cutoff=interface_cutoff, node_type="residue", load_from_disc=False,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)

        adjacency_matrix = graph_dict["adjacency_tensor"]
        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])


        node_infos = graph_dict["residue_infos"]

    elif pdb_path is not None and chain_id2protein is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)


        adjacency_matrix = get_residue_edge_encodings(distances, node_infos, chain_id2protein, interface_cutoff)

        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])

        graph_dict = None
    else:
        print("Please select 'use_dataloader' or provide path and info_dict")
        return "", None

    selection_string = f"select i_hull_{pdb_id}, "
    for res_index in interface_hull_nodes:
        node_info = node_infos[res_index]
        chain_id = node_info["chain_id"]
        residue_id = node_info["residue_id"]

        selection_string += f"({pdb_id} and chain {chain_id} and resi {residue_id}) OR "

    selection_string = selection_string[:-4]

    # print(selection_string)
    return selection_string, graph_dict


def write_selection_for_atom_interface_hull(pdb_id: str, interface_cutoff: int = 5, hull_size:int = 7, use_dataloader: bool = False,
                                  pdb_path: str = None, chain_id2protein: Dict = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = BoundComplexGraphs(config, "Dataset_v1", pdb_ids=[pdb_id],
                                     interface_distance_cutoff=interface_cutoff, node_type="atom", load_from_disc=False,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)

        adjacency_matrix = graph_dict["adjacency_tensor"]
        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])



        node_infos = graph_dict["residue_infos"]
        atom_encoding = graph_dict["node_features"]
        atom_names = graph_dict["atom_names"]

    elif pdb_path is not None and chain_id2protein is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        atom_encoding, atom_names = get_atom_encodings(node_infos, structure_info, chain_id2protein)
        adjacency_matrix = get_atom_edge_encodings(distances, atom_encoding, interface_cutoff)

        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])

        graph_dict = None
    else:
        print("Please select 'use_dataloader' or provide path and info_dict")
        return "", None

    selection_string = f"select i_hull_{pdb_id}, "
    for atom in interface_hull_nodes:
        res_index = int(atom_encoding[atom][-1])
        atom_name = atom_names[atom]
        node_info = node_infos[res_index]
        chain_id = node_info["chain_id"]
        residue_id = node_info["residue_id"]

        selection_string += f"({pdb_id} and chain {chain_id} and resi {residue_id} and name {atom_name}) OR "

    selection_string = selection_string[:-4]

    # print(selection_string)
    return selection_string, graph_dict


def write_selection_for_interface(pdb_id: str, cutoff: int = None, n_closest: int = None, use_dataloader: bool = False,
                           pdb_path: str = None, chain_id2protein: Dict = None):

    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = BoundComplexGraphs(config, "Dataset_v1", pdb_ids=[pdb_id], max_nodes=n_closest,
                                     interface_distance_cutoff=cutoff, node_type="atom", load_from_disc=False,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)


        if n_closest is None:
            interface = graph_dict["adjacency_tensor"][0,:,:] - graph_dict["adjacency_tensor"][2,:,:]
            atoms = np.where(interface > 0.001)[0]
            atoms = np.unique(atoms)

        else:
            atoms = graph_dict["closest_residues"][:n_closest]

        node_infos = graph_dict["residue_infos"]
        atom_encoding = graph_dict["node_features"]
        atom_names = graph_dict["atom_names"]

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
        config = read_config("../../config.yaml")
        dataset = BoundComplexGraphs(config, "Dataset_v1",  pdb_ids=[pdb_id],
                                     interface_distance_cutoff=cutoff, node_type="residue", load_from_disc=False,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)
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


def write_atom_connection_lines(pdb_id: str, cutoff: int = 5, use_dataloader: bool = False,
                           pdb_path: str = None, chain_id2protein: Dict = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = BoundComplexGraphs(config, "Dataset_v1",  pdb_ids=[pdb_id],
                                     interface_distance_cutoff=cutoff, node_type="atom", load_from_disc=False,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)
        A = graph_dict["adjacency_tensor"]
        node_infos = graph_dict["residue_infos"]
        atom_names = graph_dict["atom_names"]
        node_features = graph_dict["node_features"]

    elif pdb_path is not None and chain_id2protein is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure, header, chain_id2protein)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        node_features, atom_names = get_atom_encodings(node_infos, structure_info, chain_id2protein)
        A = get_atom_edge_encodings(distances, node_features, cutoff)
    else:
        return ""
    interface = A[0, :, :] - A[2, :, :]
    interface_atoms = np.where(interface > 0.001)

    interface_distances = interface[interface_atoms[0], interface_atoms[1]]
    closest_residues = np.argsort(interface_distances )[::-1]
    selection_string = ""
    for i, closest in enumerate(closest_residues):
        atom1 = interface_atoms[0][closest]
        atom2 = interface_atoms[1][closest]

        #atom1_type = ID2ATOM[int(np.where(graph_dict["node_features"][atom1][23:-1] == 1)[0])]
        atom1_type = atom_names[atom1]
        atom1_residue = int(node_features[atom1][-1])
        chain_id_1 = node_infos[atom1_residue]["chain_id"]
        residue_id_1 = node_infos[atom1_residue]["residue_id"]

        atom2_type = atom_names[atom2]
        atom2_residue = int(node_features[atom2][-1])

        chain_id_2 = node_infos[atom2_residue]["chain_id"]
        residue_id_2 = node_infos[atom2_residue]["residue_id"]

        selection_string += f"distance {pdb_id}_atom_interaction, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1} " \
                            f"and name {atom1_type}), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name {atom2_type}),100, 4 \n "

    #print(selection_string)
    return selection_string


def load_pdb(pdb_id: str, file_path: str = None):
    if file_path is None:
        config = read_config("../../config.yaml")
        dataset = BoundComplexGraphs(config, "Dataset_v1", pdb_ids=[pdb_id], load_from_disc=False, save_graphs=False)
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


def pdb_analysis():
    pdb_id = "cr9114_newcal1999" # "3sdy"
    file_path = "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/resources/dms_phillips21/pdbs/cr9114_newcal1999.pdb"
    load_command = load_pdb(pdb_id, file_path)
    chain2protein = {"b": 1, "c": 1, "d": 0}
    selection, graph_dict = write_selection_for_interface(pdb_id, cutoff=5, pdb_path=file_path, chain_id2protein=chain2protein)

    interface_graph_connections = write_atom_connection_lines(pdb_id, 5, pdb_path=file_path, chain_id2protein=chain2protein)
    #atom_connection = write_atom_connection_lines(pdb_id)
    #residue_connection = write_connection_lines(pdb_id)
    mutations = [('D', '209'), ('D', '210'), ('D', '165'), ('D', '187'), ('D', '241'), ('D', '194'), ('D', '212'),
                 ('D', '193'), ('D', '164'), ('D', '211'), ('D', '222'), ('D', '166'), ('D', '230'), ('D', '219'),
                 ('D', '192'), ('D', '206')]

    #mutations = [('D', '230'), ('D', '219'), ('D', '222')]
    mutations = [('D', '187'), ('D', '241'), ('D', '166'), ('D', '230')]

    mutation_select = select_residues(pdb_id, mutations, "Mutations")

    expanded_interface_5 = "select expanded_5"  + selection[selection.find("_interface"):] + " expand 5"
    expaned_interface_10 = "select expanded_10"  + selection[selection.find("_interface"):] + " expand 10"
    return


if __name__ == "__main__":
    #pdb_analysis()
    #main()

    pdb_id = "1bj1"

    sel, graph_dict = write_selection_for_interface(pdb_id, cutoff=5, use_dataloader=True)
    conn = write_connection_lines(pdb_id, cutoff=5, use_dataloader=True)
    atom_conn = write_atom_connection_lines(pdb_id, cutoff=5, use_dataloader=True)
    interace_hull = write_selection_for_atom_interface_hull(pdb_id, interface_cutoff = 5, hull_size = 7, use_dataloader = True)
    res_hull = write_selection_for_residue_interface_hull(pdb_id, interface_cutoff = 5, hull_size = 7, use_dataloader = True)
    a = 0
