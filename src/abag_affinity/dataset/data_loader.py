import os
from abc import ABC
import pandas as pd
from pathlib import Path
import numpy as np
from numpy.lib.npyio import NpzFile
import torch
from typing import Tuple, Dict, List
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import shutil

from abag_affinity.utils.config import read_yaml, get_data_paths, get_resources_paths
from abag_affinity.dataset.graph_generator import load_graph


def get_pdb_ids(config_path: str, dataset_name: str) -> List:
    config = read_yaml(config_path)
    data_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                               config["DATA"][dataset_name]["dataset_path"])
    return [str(file_path.split(".")[0]) for file_path in os.listdir(data_folder)]


class AffinityDataset(Dataset, ABC):
    """Superclass for all protein-protein binding affinity datasets"""

    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue", max_nodes: int = None,
                 relative_data: bool = False, save_graphs: bool = True, load_from_disc: bool = True, force_recomputation: bool = False):
        super(AffinityDataset, self).__init__()
        self.data_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                                        config["DATA"][dataset_name]["dataset_path"])

        self.pdb_ids = pdb_ids
        self.max_nodes = max_nodes
        self.dataset_name = dataset_name
        self.config = config
        self.node_type = node_type
        self.save_graphs = save_graphs
        self.load_from_disc = load_from_disc
        self.temp_dir = f"./temp_graphs/{node_type}/{dataset_name}"
        if os.path.exists(self.temp_dir) and force_recomputation:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if save_graphs:
            Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        self.force_recomputation = force_recomputation

        self.data_df = self._load_df()

        self.relative_data = relative_data
        if relative_data:
            self.get = self.get_relative
        else:
            self.get = self.get_absolute


    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    def len(self) -> int:
        return len(self.pdb_ids)

    def _load_df(self):
        if self.dataset_name == "Dataset_v1":
            summary_path, pdb_path = get_data_paths(self.config, "Dataset_v1")
            summary_df = pd.read_csv(summary_path)
            summary_df.set_index("pdb", inplace=True, drop=False)
        elif self.dataset_name == "PDBBind":
            summary_path, pdb_path = get_data_paths(self.config, "PDBBind")
            summary_df = pd.read_csv(summary_path)
            summary_df.set_index("pdb", inplace=True, drop=False)
        elif self.dataset_name == "SKEMPI.v2":
            summary_path, pdb_path = get_data_paths(self.config, "SKEMPI.v2")
            summary_df = pd.read_csv(summary_path)
            #summary_df = summary_df[summary_df["Hold_out_type"] == "AB/AG"]
            summary_df["pdb_mutation"] = summary_df.apply(lambda row: row["#Pdb"].split("_")[0] + "_" + row["Mutation(s)_cleaned"], axis=1)
            summary_df.set_index("pdb_mutation", inplace=True, drop=False)
            summary_df = summary_df[~summary_df.index.duplicated(keep='first')]

        else:
            raise ValueError("Invalid Dataset Type given (Dataset_v1, PDBBind, SKEMPI.v2")

        return summary_df

    def _get_edges(self, data_file: Dict):
        raise NotImplemented

    def _get_node_features(self, data_file: Dict):
        raise NotImplemented

    def _load_file(self, pdb_id: str) -> NpzFile:
        file_path = os.path.join(self.data_folder, pdb_id + ".npz")
        np_file: NpzFile = np.load(file_path, allow_pickle=True)
        return np_file

    def get_graph_dict(self, pdb_id: str, mutated_complex: str = None, distance_cutoff: int = 5) -> Dict:

        if mutated_complex is not None:
            pdb_code, mutation_code = pdb_id.split("_")
            if mutated_complex == "wildtype":
                file_path = os.path.join(self.temp_dir, pdb_code + mutated_complex + ".npz")
            else:
                file_path = os.path.join(self.temp_dir, "_".join([pdb_code, mutation_code, mutated_complex + ".npz"]))
        else:
            file_path = os.path.join(self.temp_dir, pdb_id + ".npz")

        if not self.force_recomputation and self.load_from_disc and os.path.exists(file_path):
            graph_dict = dict(np.load(file_path, allow_pickle=True))
        else:
            row = self.data_df.loc[pdb_id]
            graph_dict = load_graph(row, self.dataset_name, self.config, node_type=self.node_type,
                                    distance_cutoff=distance_cutoff, mutated_complex=mutated_complex)

        if self.save_graphs and not os.path.exists(file_path):
            graph_dict.pop("atom_names", None)

            np.savez_compressed(file_path, **graph_dict)

        return graph_dict

    def get_absolute(self, idx: int) -> Data:
        pdb_id = self.pdb_ids[idx]
        graph_dict = self.get_graph_dict(pdb_id)
        residue_features = self._get_node_features(graph_dict)
        edge_index, edge_features = self._get_edges(graph_dict)
        affinity = np.array(graph_dict["affinity"])

        data = Data(
            x=torch.Tensor(residue_features),
            edge_index=torch.Tensor(edge_index).long(),
            edge_attr=torch.Tensor(edge_features),
            y=torch.from_numpy(affinity).float()
        )

        return data

    def get_relative(self, idx: int) -> Tuple[Data, Data]:
        pdb_mutation_code = self.pdb_ids[idx]
        wildtype_graph_dict = self.get_graph_dict(pdb_mutation_code, "wildtype")
        wt_residue_features = self._get_node_features(wildtype_graph_dict)
        wt_edge_index, wt_edge_features = self._get_edges(wildtype_graph_dict)
        wt_affinity = np.array(wildtype_graph_dict["affinity"])

        wildtype = Data(
            x=torch.Tensor(wt_residue_features),
            edge_index=torch.Tensor(wt_edge_index).long(),
            edge_attr=torch.Tensor(wt_edge_features),
            y=torch.from_numpy(wt_affinity).float()
        )

        mutation_graph_dict = self.get_graph_dict(pdb_mutation_code, self.config["DATA"][self.dataset_name]["mutated_pdb_path"])
        mut_residue_features = self._get_node_features(mutation_graph_dict)
        mut_edge_index, mut_edge_features = self._get_edges(mutation_graph_dict)
        mut_affinity = np.array(mutation_graph_dict["affinity"])

        mutation = Data(
            x=torch.Tensor(mut_residue_features),
            edge_index=torch.Tensor(mut_edge_index).long(),
            edge_attr=torch.Tensor(mut_edge_features),
            y=torch.from_numpy(mut_affinity).float()
        )

        return wildtype, mutation


class SimpleGraphs(AffinityDataset, ABC):
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue",
                 relative_data: bool = False):
        super(SimpleGraphs, self).__init__(config, dataset_name, pdb_ids, node_type=node_type, relative_data=relative_data)

    def _get_edges(self, data_file: Dict) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix = data_file["adjacency_tensor"]
        edges = np.where(adjacency_matrix[0, :, :] > 0.001)
        edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))
        return np.array(edges), edge_attributes

    def _get_node_features(self, data_file: Dict) -> np.ndarray:
        return data_file["node_features"]


class InterfaceGraphs(AffinityDataset, ABC):
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, interface_hull_size: int = 5, node_type: str = "residue",
                 relative_data: bool = False, save_graphs: bool = True, load_from_disc: bool = True, force_recomputation: bool = False):
        super(InterfaceGraphs, self).__init__(config, dataset_name, pdb_ids, node_type=node_type,
                                              relative_data=relative_data, save_graphs=save_graphs,
                                              load_from_disc=load_from_disc, force_recomputation=force_recomputation)

        self.interface_hull_size = interface_hull_size

    def _get_edges(self, data_file: Dict) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix = data_file["adjacency_tensor"]
        adjacency_matrix = adjacency_matrix[:, data_file["interface_nodes"], :][:,:,data_file["interface_nodes"]]
        edges = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)
        edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))
        return np.array(edges), edge_attributes

    def _get_node_features(self, data_file: Dict) -> np.ndarray:
        adjacency_matrix = data_file["adjacency_tensor"]
        interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
        interface_nodes = np.unique(interface_nodes)

        interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < self.interface_hull_size)
        interface_hull_nodes = np.unique(interface_hull[1])

        data_file["interface_nodes"] = interface_hull_nodes

        return data_file["node_features"][interface_hull_nodes]


class FixedSizeGraphs(AffinityDataset, ABC):
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, max_nodes: int, node_type: str = "residue",
                 relative_data: bool = False):
        super(FixedSizeGraphs, self).__init__(config, dataset_name, pdb_ids, max_nodes=max_nodes, node_type=node_type,
                                              relative_data=relative_data)

    def _get_edges(self, data_file: Dict) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix = data_file["adjacency_tensor"]
        closest_nodes = data_file["closest_residues"][:self.max_nodes]

        closest_nodes_adj = adjacency_matrix[0, closest_nodes, :][:, closest_nodes]

        edges = np.where(closest_nodes_adj[ :, :] > 0.001)
        edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))

        return np.array(edges), edge_attributes

    def _get_node_features(self, data_file: Dict) -> np.ndarray:
        closest_nodes = data_file["closest_residues"][:self.max_nodes]
        node_features = data_file["node_features"][closest_nodes]

        if len(closest_nodes) < self.max_nodes:
            missing_nodes = self.max_nodes - len(closest_nodes)
            shape = (missing_nodes, ) + node_features.shape[1:]
            node_features = np.vstack([node_features, np.zeros(shape)])

        return node_features


class DDGBackboneInputs(AffinityDataset, ABC):
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, max_nodes: int, relative_data: bool = False):
        super(DDGBackboneInputs, self).__init__(config, dataset_name, pdb_ids, max_nodes, relative_data)

    def _get_edges(self, data_file: NpzFile) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix = data_file["adjacency_tensor"]
        closest_nodes = data_file["closest_residues"][:self.max_nodes]

        closest_nodes_adj = adjacency_matrix[0, closest_nodes, :][:, closest_nodes]

        edges = np.where(closest_nodes_adj[ :, :] > 0.001)
        edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))

        return np.array(edges), edge_attributes


    def _get_node_features(self, data_file: NpzFile) -> np.ndarray:
        closest_nodes = data_file["closest_residues"][:self.max_nodes]
        infos = data_file["residue_infos"][closest_nodes]
        coords = data_file["residue_atom_coordinates"][closest_nodes, :]
        node_features = [] # (N, 14 * 3 + 1 + 1 + 1 + 14): (Num_Nodes, Atom Coords * (x,y,z) + aa_index + seq_index, + chain_seq_index + atom_coordinate_mask)
        for residue_info, residue_coords in zip(infos, coords):
            node_coordinates = residue_coords.flatten()
            coordinate_mask = ~np.all(np.isinf(residue_coords), axis=-1)
            indices = np.array([residue_info["residue_pdb_index"],
                                           residue_info["chain_idx"], residue_info["on_chain_residue_idx"]])
            all_features = np.concatenate([node_coordinates, indices, coordinate_mask])
            node_features.append(all_features)

        return np.array(node_features)


if __name__ == "__main__":
    import random
    random.seed(1)
    config_file = "../config.yaml"
    config = read_yaml(config_file)

    dataset_name = "SKEMPI.v2"

    mutation_path = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                                 config["DATA"][dataset_name]["mutated_pdb_path"])
    summary_path, _ = get_data_paths(config, dataset_name)
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[~summary_df["-log(Kd)_mut"].isna()]
    available_affinities = set(
        summary_df.apply(lambda row: row["#Pdb"].split("_")[0] + "_" + row["Mutation(s)_cleaned"], axis=1).values)

    pdb_ids = summary_df["pdb"].unique().tolist()
    available_pdb_ids = []
    pdb_mutation_codes = []
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.upper()
        pdb_path = os.path.join(mutation_path, pdb_id)
        if os.path.exists(pdb_path):
            pdb_mutation_codes.extend(
                [pdb_id + "_" + mutation_code.split(".")[0] for mutation_code in os.listdir(pdb_path)])
            available_pdb_ids.append(pdb_id)

    pdb_mutation_codes = available_affinities.intersection(set(pdb_mutation_codes))
    # random split for PDBBind
    random.shuffle(available_pdb_ids)
    split = int(len(available_pdb_ids) * 0.9)
    val_pdb_ids = available_pdb_ids[split:]
    train_pdb_ids = available_pdb_ids[:split]

    train_ids = [code for code in pdb_mutation_codes if code.split("_")[0] in train_pdb_ids]
    val_ids = [code for code in pdb_mutation_codes if code.split("_")[0] in val_pdb_ids]

    train_data = InterfaceGraphs(config, dataset_name, train_ids, relative_data=True, force_recomputation=True)
    a = train_data.__getitem__(0)
    b = 0