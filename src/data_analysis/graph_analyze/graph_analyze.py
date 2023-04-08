import os.path
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path

from abag_affinity.dataset import AffinityDataset
from abag_affinity.utils.config import read_config, get_resources_paths
from abag_affinity.utils.pdb_processing import get_residue_infos, get_distances, get_residue_edge_encodings, \
    get_atom_edge_encodings, get_atom_encodings, get_residue_encodings
from abag_affinity.utils.pdb_reader import read_file


dataset_name = "abag_affinity"

def write_selection_for_residue_interface_hull(pdb_id: str, interface_cutoff: int = 5, hull_size:int = 7, use_dataloader: bool = False,
                                  pdb_path: str = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name, pdb_ids=[pdb_id],
                                     node_type="residue",interface_hull_size=hull_size,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)

        adjacency_matrix = graph_dict["adjacency_tensor"]
        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])


        node_infos = graph_dict["residue_infos"]

    elif pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure)
        distances, closest_atoms = get_distances(node_infos, residue_distance=True, ca_distance=False)


        adjacency_matrix = get_residue_edge_encodings(distances, node_infos, interface_cutoff)

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
                                  pdb_path: str = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name, pdb_ids=[pdb_id],
                                     node_type="atom", interface_hull_size=hull_size,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)

        adjacency_matrix = graph_dict["adjacency_tensor"]
        interface_nodes = np.where((adjacency_matrix[3, :, :] < 5) & (adjacency_matrix[2, :, :] == 0))[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] <= hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])

        node_infos = graph_dict["residue_infos"]
        atom_encoding = graph_dict["node_features"]
        atom_names = graph_dict["atom_names"]

    elif pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        atom_encoding, atom_names = get_atom_encodings(node_infos, structure_info)
        adjacency_matrix = get_atom_edge_encodings(distances, atom_encoding, interface_cutoff)

        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])

        graph_dict = None
    else:
        print("Please select 'use_dataloader' or provide path and info_dict")
        return "", None

    interface_hull_residue_expanded = np.where(adjacency_matrix[1, interface_hull_nodes, :] == 1)
    interface_hull_residue_expanded_nodes = np.unique(interface_hull_residue_expanded[1])

    selection_string = f"select i_hull_{pdb_id}, "
    for atom in interface_hull_residue_expanded_nodes:
        res_index = int(atom_encoding[atom][-1])
        atom_name = atom_names[atom]
        node_info = node_infos[res_index]
        chain_id = node_info["chain_id"]
        residue_id = node_info["residue_id"]

        selection_string += f"({pdb_id} and chain {chain_id} and resi {residue_id} and name {atom_name}) OR "

    selection_string = selection_string[:-4]

    # print(selection_string)
    return selection_string, graph_dict


def write_selection_for_interface(pdb_id: str, hull_size: int = None, n_closest: int = None, use_dataloader: bool = False,
                           pdb_path: str = None):

    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name, pdb_ids=[pdb_id], max_nodes=n_closest,
                                     node_type="atom", interface_hull_size=hull_size,
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

    elif pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        atom_encoding, atom_names = get_atom_encodings(node_infos, structure_info)
        A = get_atom_edge_encodings(distances, atom_encoding, 5)
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


def write_connection_lines(pdb_id: str, hull_size: int = 5, graph_dict = None, use_dataloader: bool = False,
                           pdb_path: str = None):
    if graph_dict is None and use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name,  pdb_ids=[pdb_id],
                                     node_type="residue", interface_hull_size=hull_size,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)
        interface = graph_dict["adjacency_tensor"][0, :, :] - graph_dict["adjacency_tensor"][2, :, :]

    elif graph_dict is None and pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, residue_infos, residue_atom_coordinates = get_residue_infos(structure)
        distances, closest_residues = get_distances(residue_infos, residue_distance=False)
        A = get_residue_edge_encodings(distances, residue_infos, 5)
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
        a = 0

    #print(selection_string)
    return selection_string


def write_atom_connection_lines(pdb_id: str, hull_size: int = 5, use_dataloader: bool = False,
                           pdb_path: str = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name,  pdb_ids=[pdb_id],
                                     node_type="atom",interface_hull_size=hull_size,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)
        A = graph_dict["adjacency_tensor"]
        node_infos = graph_dict["residue_infos"]
        atom_names = graph_dict["atom_names"]
        node_features = graph_dict["node_features"]

    elif pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        node_features, atom_names = get_atom_encodings(node_infos, structure_info)
        A = get_atom_edge_encodings(distances, node_features, 5)
    else:
        return ""

    interface_edges = np.where((A[2, :, :] != 1) & (A[0, :, :] > 0.001))

    if True:
        interface_edges = np.array(interface_edges)

        interface_edges = interface_edges[:, node_features[interface_edges[0], 20] == 1]

    distance = A[0, interface_edges[0], interface_edges[1]]

    max_interface_edges = None #300
    if max_interface_edges is not None:
        interface_edges = np.array(interface_edges)
        sorted_edge_idx = np.argsort(-distance)[:max_interface_edges] # use negtive values to sort descending
        interface_atoms = interface_edges[:, sorted_edge_idx]
        interface_distances = A[0, interface_atoms[0], interface_atoms[1]]
        closest_residues = np.arange(0, len(interface_distances))
    else:
        interface = A[0, :, :] - A[2, :, :]
        interface_atoms = np.where(interface > 0.001)

        interface_distances = interface[interface_atoms[0], interface_atoms[1]]
        interface_atoms = np.array(interface_atoms)
        closest_residues = np.argsort(interface_distances )[::-1]
    selection_string = ""
    edges = []
    for i, closest in enumerate(closest_residues):
        atom1 = interface_atoms[0, closest]
        atom2 = interface_atoms[1, closest]

        edges.append((atom1, atom2))

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


def write_graph_connection_lines(pdb_id: str, hull_size: int = 7, node_distance: int = None, node_type: str = "residue", use_dataloader: bool = False,
                           pdb_path: str = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name,  pdb_ids=[pdb_id],
                                    node_type=node_type, interface_hull_size=hull_size,
                                    save_graphs=False, force_recomputation=True,
                                  max_edge_distance=5)

        graph_dict = dataset.get_graph_dict(pdb_id)
        A = graph_dict["adjacency_tensor"]
        node_infos = graph_dict["residue_infos"]
        atom_names = graph_dict["atom_names"]
        node_features = graph_dict["node_features"]

    elif pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure)
        residue_distances = node_type == "residue"
        distances, closest_atoms = get_distances(node_infos, residue_distance=residue_distances, ca_distance=residue_distances)
        if node_type == "residue":
            node_features = get_residue_encodings(node_infos, structure_info)
            A = get_residue_edge_encodings(distances, node_infos, distance_cutoff=5)
        else:
            node_features, atom_names = get_atom_encodings(node_infos, structure_info)
            A = get_atom_edge_encodings(distances, node_features, distance_cutoff=5)
    else:
        return ""

    if node_distance is None:
        edges = np.where(A[0, :, :] > 0.001)
    else:
        edges = np.where(A[3, :, :] < node_distance)


    distance = A[3, edges[0], edges[1]]
    edges = np.array(edges)
    closest_atoms = np.argsort(distance)

    max_edges = None #300
    if max_edges is not None:
        closest_atoms = closest_atoms[:max_edges]

    selection_string = ""
    all_edges = set()
    for i, closest in enumerate(closest_atoms):
        atom1 = edges[0, closest]
        atom2 = edges[1, closest]

        if distance[closest] == 0 or (atom1, atom2) in all_edges:
            continue
        all_edges.add((atom1, atom2))
        all_edges.add((atom2, atom1))

        #atom1_type = ID2ATOM[int(np.where(graph_dict["node_features"][atom1][23:-1] == 1)[0])]
        if node_type == "atom":
            atom1_residue = int(node_features[atom1][-1])
        else:
            atom1_residue = atom1
        chain_id_1 = node_infos[atom1_residue]["chain_id"]
        residue_id_1 = node_infos[atom1_residue]["residue_id"]

        if node_type == "atom":
            atom2_residue = int(node_features[atom2][-1])
        else:
            atom2_residue = atom2
        chain_id_2 = node_infos[atom2_residue]["chain_id"]
        residue_id_2 = node_infos[atom2_residue]["residue_id"]

        if node_type == "atom":
            atom1_type = atom_names[atom1]
            atom2_type = atom_names[atom2]
            selection_string += f"distance {node_distance}A_{pdb_id}_{node_type}_graph, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1} " \
                            f"and name {atom1_type}), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name {atom2_type}),100, 4 \n "
        else:
            selection_string += f"distance {node_distance}A_{pdb_id}_{node_type}_graph, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1}" \
                            f" and name CA), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name CA),100, 4 \n "

        a = 0
    #print(selection_string)
    return selection_string


def write_peptide_bond_connection_lines(pdb_id: str, hull_size: int = 7, node_distance: int = None, node_type: str = "residue", use_dataloader: bool = False,
                           pdb_path: str = None):
    if use_dataloader:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name,  pdb_ids=[pdb_id],
                                     node_type=node_type, interface_hull_size=hull_size,
                                     save_graphs=False, force_recomputation=True)

        graph_dict = dataset.get_graph_dict(pdb_id)
        A = graph_dict["adjacency_tensor"]
        node_infos = graph_dict["residue_infos"]
        atom_names = graph_dict["atom_names"]
        node_features = graph_dict["node_features"]

    elif pdb_path is not None:
        structure, header = read_file(pdb_id, pdb_path)
        structure_info, node_infos, residue_atom_coordinates = get_residue_infos(structure)
        distances, closest_atoms = get_distances(node_infos, residue_distance=False, ca_distance=False)
        node_features, atom_names = get_atom_encodings(node_infos, structure_info)
        A = get_atom_edge_encodings(distances, node_features, distance_cutoff=5)
    else:
        return ""


    edges = np.where(A[1, :, :]  == 1)

    distance = A[3, edges[0], edges[1]]
    edges = np.array(edges)
    closest_atoms = np.argsort(distance)

    max_edges = None #300
    if max_edges is not None:
        closest_atoms = closest_atoms[:max_edges]

    selection_string = ""
    all_edges = set()
    for i, closest in enumerate(closest_atoms):
        atom1 = edges[0, closest]
        atom2 = edges[1, closest]

        if distance[closest] == 0 or (atom1, atom2) in all_edges:
            continue
        all_edges.add((atom1, atom2))
        all_edges.add((atom2, atom1))

        #atom1_type = ID2ATOM[int(np.where(graph_dict["node_features"][atom1][23:-1] == 1)[0])]
        if node_type == "atom":
            atom1_residue = int(node_features[atom1][-1])
        else:
            atom1_residue = atom1
        chain_id_1 = node_infos[atom1_residue]["chain_id"]
        residue_id_1 = node_infos[atom1_residue]["residue_id"]

        if node_type == "atom":
            atom2_residue = int(node_features[atom2][-1])
        else:
            atom2_residue = atom2
        chain_id_2 = node_infos[atom2_residue]["chain_id"]
        residue_id_2 = node_infos[atom2_residue]["residue_id"]

        if node_type == "atom":
            atom1_type = atom_names[atom1]
            atom2_type = atom_names[atom2]
            selection_string += f"distance {node_distance}A_{pdb_id}_{node_type}_graph, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1} " \
                            f"and name {atom1_type}), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name {atom2_type}),100, 4 \n "
        else:
            selection_string += f"distance {pdb_id}_peptide_bond_graph, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1}" \
                            f" and name CA), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name CA),100, 4 \n "

        a = 0
    #print(selection_string)
    return selection_string


def load_pdb(pdb_id: str, file_path: str = None):
    if file_path is None:
        config = read_config("../../config.yaml")
        dataset = AffinityDataset(config, dataset_name, pdb_ids=[pdb_id], save_graphs=False)
        _, pdb_path = get_resources_paths(config, dataset_name)
        file_name = dataset.data_df[dataset.data_df["pdb"] == pdb_id]["abdb_file"].values[0]
        file_path = os.path.join(pdb_path, file_name)


    pymol_commands = 'hide all\n'
    pymol_commands += f'load {file_path}, {pdb_id}\n'
    pymol_commands += "spectrum chain, blue red green\n"

    return pymol_commands


def start_pymol(pymol_path: str):
    from utils import PyMol
    pm = PyMol(pymol_path, mode=1)

    return pm


def get_deeprefine_edges(pdb_id: str):
    config = read_config("../../config.yaml")
    dataset = AffinityDataset(config, dataset_name, pdb_ids=[pdb_id], interface_hull_size= 7, node_type="atom",
                                 save_graphs=False, force_recomputation=True, pretrained_model="DeepRefine")

    datapoint = dataset.load_data_point(pdb_id)

    graph_dict = dataset.get_graph_dict(pdb_id)
    A = graph_dict["adjacency_tensor"]
    node_infos = graph_dict["residue_infos"]
    atom_names = graph_dict["atom_names"]
    node_features = graph_dict["node_features"]

    edges = datapoint["deeprefine_graph"].edges()
    selection_string = ""
    for edge in zip(edges[0].tolist(), edges[1].tolist()):
        atom1, atom2 = edge
        #atom1_type = ID2ATOM[int(np.where(graph_dict["node_features"][atom1][23:-1] == 1)[0])]
        atom1_type = atom_names[atom1]
        atom1_residue = int(node_features[atom1][-1])
        chain_id_1 = node_infos[atom1_residue]["chain_id"]
        residue_id_1 = node_infos[atom1_residue]["residue_id"]

        atom2_type = atom_names[atom2]
        atom2_residue = int(node_features[atom2][-1])

        chain_id_2 = node_infos[atom2_residue]["chain_id"]
        residue_id_2 = node_infos[atom2_residue]["residue_id"]

        selection_string += f"distance {pdb_id}_deeprefine_edges, ({pdb_id} and chain {chain_id_1} and resi {residue_id_1} " \
                            f"and name {atom1_type}), ({pdb_id} and chain {chain_id_2} and resi {residue_id_2} and name {atom2_type}),100, 4 \n "

    return selection_string


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


def generate_graph_reasons(pdb_id: str, file_path: str, image_path: str, chain2protein: Dict, rotations: List, zoom:int,
                           node_distance=3 ):
    Path(image_path).mkdir(parents=True, exist_ok=True)

    load_command = load_pdb(pdb_id,  os.path.join(file_path, f'{pdb_id}.pdb'))

    use_dataloader = True

    pm = start_pymol("/home/fabian/Downloads/pymol/bin")
    pm("bg_color white")
    pm("set dash_width, 2")
    pm("set dash_gap, 0")
    pm("set dash_color, grey")
    pm(load_command)
    for axis, angle in rotations:
        pm(f"rotate {axis}, {angle}")

    pm(f"zoom ({pdb_id}), buffer={zoom}")


    pm("spectrum chain, tv_blue marine teal")

    pm(f"png {os.path.join(image_path, 'full_cartoon.png')}, dpi=300")
    pm("show sticks")
    pm(f"png {os.path.join(image_path, 'full_sticks_cartoon.png')}, dpi=300")
    pm("hide cartoon")
    pm(f"png {os.path.join(image_path, 'full_sticks.png')}, dpi=300")


    residue_graph_edges = write_graph_connection_lines(pdb_id, hull_size=None, pdb_path=file_path,
                                                       use_dataloader=use_dataloader, node_distance=node_distance,
                                                       node_type="residue")
    pm("hide cartoon")
    for edge in residue_graph_edges.split("\n"):
        pm(edge)
    pm("hide label")
    pm("hide sticks")
    pm("select all_ca_atoms, name ca")
    pm("show spheres, all_ca_atoms")
    pm("set sphere_scale, 0.4, (all)")
    pm("deselect")
    pm(f"png {os.path.join(image_path, 'full_graph.png')}, dpi=300")

    return


def antibody_antigen_binding(pdb_id: str, file_path: str, image_path: str, chain2protein: Dict, rotations: List,
                             zoom: int):
    Path(image_path).mkdir(parents=True, exist_ok=True)

    load_command = load_pdb(pdb_id,  os.path.join(file_path, f'{pdb_id}.pdb'))

    ab_chains = [ chain.upper() for chain, ag in chain2protein.items() if ag == 0]
    ag_chains = [ chain.upper() for chain, ag in chain2protein.items() if ag == 1]

    pm = start_pymol("/home/fabian/Downloads/pymol/bin")
    pm("bg_color white")
    pm(load_command)
    for axis, angle in rotations:
        pm(f"rotate {axis}, {angle}")

    pm(f"zoom ({pdb_id}),  buffer={zoom}")


    pm("spectrum chain, tv_blue marine teal")
    pm(f"png {os.path.join(image_path, 'full_cartoon.png')}, dpi=300")

    for chain in ab_chains:
        pm(f"hide (chain {chain.upper()})")
    pm(f"png {os.path.join(image_path, 'ag_only_cartoon.png')}, dpi=300")

    pm("show cartoon, all")
    for chain in ag_chains:
        pm(f"hide (chain {chain.upper()})")
    pm(f"png {os.path.join(image_path, 'ab_only_cartoon.png')}, dpi=300")

    return


def show_mutation(pdb_id: str, file_path: str, image_path: str, chain2protein: Dict, mutation_code: str, rotations: List,
                  zoom: int):
    Path(image_path).mkdir(parents=True, exist_ok=True)

    load_command = load_pdb(pdb_id,  os.path.join(file_path, f'{pdb_id}.pdb'))
    load_command_mutant = load_pdb("mutant",  os.path.join(file_path, f'{pdb_id}-{mutation_code}.pdb'))


    pm = start_pymol("/home/fabian/Downloads/pymol/bin")
    pm("bg_color white")
    pm(load_command)
    pm(load_command_mutant)
    pm(f"hide mutant")

    for axis, angle in rotations:
        pm(f"rotate {axis}, {angle}")

    pm(f"zoom ({pdb_id}), buffer= buffer={zoom}")
    pm("spectrum chain, tv_blue marine teal")


    pm(f"png {os.path.join(image_path, 'wt_cartoon.png')}, dpi=300")

    pm(f"select mutation_side, ({pdb_id} and chain {mutation_code[1].upper()} and resi {mutation_code[2:-1]})")
    pm(f"zoom (mutation_side)")
    pm(f"show sticks, (mutation_side)")
    pm(f"spectrum elem, selection=mutation_side")
    pm("deselect")
    pm(f"png {os.path.join(image_path, 'wt_zoom.png')}, dpi=300")

    pm(f"hide {pdb_id}")
    pm(f"show mutant")

    pm(f"zoom (mutant),  buffer={zoom}")
    pm("spectrum chain, tv_blue marine teal")

    pm(f"png {os.path.join(image_path, 'mut_cartoon.png')}, dpi=300")

    pm(f"select mutation_side, (mutant and chain {mutation_code[1].upper()} and resi {mutation_code[2:-1]})")
    pm(f"zoom (mutation_side)")
    pm(f"show sticks, (mutation_side)")
    pm(f"spectrum elem, selection=mutation_side")
    pm("deselect")
    pm(f"png {os.path.join(image_path, 'mut_zoom.png')}, dpi=300")

    return


def generate_graph_story(pdb_id: str, file_path: str, image_path: str, chain2protein: Dict, rotations: List, zoom: int):

    Path(image_path).mkdir(parents=True, exist_ok=True)

    hull_file_path = os.path.join(file_path, f'interface_hull_7/{pdb_id}.pdb')
    load_command = load_pdb(pdb_id, hull_file_path)
    load_full = load_pdb(pdb_id + "_full", os.path.join(file_path, f'{pdb_id}.pdb'))

    use_dataloader = True
    hull_size = 7

    pm = start_pymol("/home/fabian/Downloads/pymol/bin")
    pm("bg_color white")
    pm("set dash_width, 2")
    pm("set dash_gap, 0")
    pm(load_command)
    pm(load_full)
    for axis, angle in rotations:
        pm(f"rotate {axis}, {angle}")

    pm(f"zoom ({pdb_id}_full), buffer={zoom}")
    pm("spectrum chain, tv_blue marine teal")

    pm(f"png {os.path.join(image_path, 'full_cartoon.png')}, dpi=500")
    pm("show sticks")
    pm("show cartoon")
    pm(f"png {os.path.join(image_path, 'full_cartoon_sticks.png')}, dpi=500")

    pm(f"disable ({pdb_id}_full)")
    pm(f"zoom ({pdb_id}), complete=1")

    pm(f"png {os.path.join(image_path, 'hull_cartoon_sticks.png')}, dpi=300")

    pm("hide cartoon")
    pm("hide sticks")

    pm(f"select all_ca_atoms, ({pdb_id} and name ca)")
    pm("show spheres, all_ca_atoms")
    pm("set sphere_scale, 0.4, (all)")
    pm("deselect")
    pm(f"png {os.path.join(image_path, 'hull_nodes.png')}, dpi=300")

    residue_graph_edges = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path,
                                                       use_dataloader=use_dataloader, node_distance=5,
                                                       node_type="residue")

    for edge in residue_graph_edges.split("\n"):
        pm(edge)
    pm("hide label")
    pm(f"set dash_color, grey, 5A_{pdb_id}_residue_graph")
    pm(f"png {os.path.join(image_path, 'hull_graph.png')}, dpi=300")

    residue_interface_graph_connections = write_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path,
                                                                 use_dataloader=use_dataloader)

    for edge in residue_interface_graph_connections.split("\n"):
        pm(edge)
    pm(f"set dash_color, red, {pdb_id}_interaction")
    pm("hide label")
    pm(f"disable 5A_{pdb_id}_residue_graph")
    pm(f"enable 5A_{pdb_id}_residue_graph")
    pm(f"set dash_width, 1, 5A_{pdb_id}_residue_graph")
    pm(f"png {os.path.join(image_path, 'hull_graph_interface_highlight.png')}, dpi=300")


    residue_peptide_graph_connections = write_peptide_bond_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path,
                                                                 use_dataloader=use_dataloader)


    for edge in residue_peptide_graph_connections.split("\n"):
        pm(edge)
    pm(f"set dash_color, red, {pdb_id}_peptide_bond_graph")
    pm("hide label")
    pm(f"disable {pdb_id}_interaction")
    pm(f"disable 5A_{pdb_id}_residue_graph")
    pm(f"enable 5A_{pdb_id}_residue_graph")
    pm(f"png {os.path.join(image_path, 'hull_graph_peptide_bond_highlight.png')}, dpi=300")

    pm("spectrum resn")
    pm(f"disable {pdb_id}_interaction")
    pm(f"enable 5A_{pdb_id}_residue_graph")
    pm(f"disable {pdb_id}_peptide_bond_graph")
    pm(f"png {os.path.join(image_path, 'graph_residue_type.png')}, dpi=300")

    pm(f"disable 5A_{pdb_id}_residue_graph")
    pm(f"png {os.path.join(image_path, 'nodes_residue_type.png')}, dpi=300")

    return


def pdb_analysis():
    pdb_id = "1e6j" # "3sdy"
    file_path = f'/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/cleaned_pdb/Dataset_v1/interface_hull_7/{pdb_id}.pdb'
    #load_command = load_pdb(pdb_id, file_path)
    #load_full = load_pdb(pdb_id + "_full", f'/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/cleaned_pdb/Dataset_v1/{pdb_id}.pdb')
    load_command = load_pdb(pdb_id, f'/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/cleaned_pdb/Dataset_v1/{pdb_id}.pdb')
    chain2protein = {"p": 1, "h": 0, "l": 0}

    use_dataloader = True
    hull_size = 7

    selection, graph_dict = write_selection_for_interface(pdb_id, hull_size=hull_size, pdb_path=file_path,
                                                          use_dataloader=use_dataloader)

    residue_interface_graph_connections = write_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path,
                                                              use_dataloader=use_dataloader)

    interface_graph_connections = write_atom_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader)

    atom_graph_edges = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader, node_distance=5, node_type="atom")

    atom_graph_edges_3 = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader, node_distance=3, node_type="atom")

    atom_graph_edges_2 = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader, node_distance=2, node_type="atom")

    residue_graph_edges = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader, node_distance=5, node_type="residue")

    residue_graph_edges_3 = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader, node_distance=3, node_type="residue")

    residue_graph_edges_2 = write_graph_connection_lines(pdb_id, hull_size=hull_size, pdb_path=file_path, use_dataloader=use_dataloader, node_distance=2, node_type="residue")

    #mutations = [('D', '187'), ('D', '241'), ('D', '166'), ('D', '230')]
    #mutation_select = select_residues(pdb_id, mutations, "Mutations")

    deep_refine_edges = get_deeprefine_edges(pdb_id)

    res_hull,_ = write_selection_for_atom_interface_hull(pdb_id, interface_cutoff=5, hull_size=hull_size, pdb_path=file_path,
                                                         use_dataloader=use_dataloader)
    expanded_interface_5 = "select expanded_5"  + selection[selection.find("_interface"):] + " expand 5"
    expaned_interface_7 = "select expanded_10"  + selection[selection.find("_interface"):] + " expand 7"


    pm = start_pymol("/home/fabian/Downloads/pymol/bin")
    pm("bg_color white")
    pm("set dash_width, 2")
    pm("set dash_gap, 0")
    pm("set dash_color, grey")
    pm(load_command)
    #pm(load_full)
    pm("spectrum chain, tv_blue marine teal")

    pm("png ~/pymol_images/full_comic.png, dpi=300")
    pm("show sticks")
    pm("png ~/pymol_images/full_sticks.png, dpi=300")

    residue_graph_edges = write_graph_connection_lines(pdb_id, hull_size=None, pdb_path=file_path,
                                                       use_dataloader=use_dataloader, node_distance=5,
                                                       node_type="residue")
    pm("hide cartoon")
    for edge in residue_graph_edges.split("\n"):
        pm(edge)
    pm("hide label")
    pm("hide sticks")
    pm("select all_ca_atoms, name ca")
    pm("show spheres, all_ca_atoms")
    pm("set sphere_scale, 0.4, (all)")
    pm("deselect")
    pm("png ~/pymol_images/full_graph.png, dpi=300")

    pm("set sphere_scale, 0.15, (all)")

    pm("select all_ca_atoms, name ca")

    pm(f"select hull_ca_atoms, {pdb_id} and name ca")
    pm("hide label")

    pm(res_hull)
    pm(selection)



    for edge in residue_interface_graph_connections.split("\n"):
        pm(edge)
    pm("hide label")


    """
    
    for edge in residue_graph_edges.split("\n"):
        pm(edge)
    pm("hide label")

    
    for edge in residue_graph_edges_2.split("\n"):
        pm(edge)
    pm("hide label")

    for edge in residue_graph_edges_3.split("\n"):
        pm(edge)
    pm("hide label")

    graph_edge_commands = atom_graph_edges.split("\n")
    for edge in graph_edge_commands:
        pm(edge)
    pm("hide label")

    for edge in atom_graph_edges_2.split("\n"):
        pm(edge)
    pm("hide label")

    for edge in atom_graph_edges_3.split("\n"):
        pm(edge)
    pm("hide label")

    for edge in interface_graph_connections.split("\n"):
        pm(edge)
    pm("hide label")

    for edge in deep_refine_edges.split("\n"):
        pm(edge)
    """


    return


if __name__ == "__main__":
    pdb_id = "5w6g".lower() # "3sdy"
    chain2protein = {"c": 1, "a": 0, "b": 0}
    #mutation_code = "dc167a"
    rotations = [("y", 40), ("x", 20), ("z", -15), ("x", -20)]
    zoom = -15

    file_path = '/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/cleaned_pdb/abag_affinity'
    #generate_graph_story(pdb_id, file_path, "/home/fabian/Desktop/Uni/Masterthesis/MA-images/graph_story", chain2protein, rotations , zoom)
    generate_graph_reasons(pdb_id, file_path, "/home/fabian/Desktop/Uni/Masterthesis/MA-images/graph_reasons", chain2protein, rotations , zoom)
    #antibody_antigen_binding(pdb_id, file_path, "/home/fabian/pymol_images/abag_binding", chain2protein, rotations , zoom)
    #show_mutation(pdb_id, file_path, "/home/fabian/pymol_images/relative_binding", chain2protein, mutation_code, rotations , zoom)
    #pdb_analysis()
    exit(0)
    #main()

    #pdb_id = "1bj1"

    sel, graph_dict = write_selection_for_interface(pdb_id, cutoff=5, use_dataloader=True)
    conn = write_connection_lines(pdb_id, cutoff=5, use_dataloader=True)
    atom_conn = write_atom_connection_lines(pdb_id, cutoff=5, use_dataloader=True)
    interace_hull = write_selection_for_atom_interface_hull(pdb_id, interface_cutoff = 5, hull_size = 7, use_dataloader = True)
    res_hull = write_selection_for_residue_interface_hull(pdb_id, interface_cutoff = 5, hull_size = 7, use_dataloader = True)
    a = 0
