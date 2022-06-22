import os
from abc import ABC

import numpy as np
from numpy.lib.npyio import NpzFile
import torch
from typing import Tuple, Dict, List
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from abag_affinity.utils.config import read_yaml, get_data_paths


def get_pdb_ids(config_path: str, dataset_name: str) -> List:
    config = read_yaml(config_path)
    data_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                               config["DATA"][dataset_name]["dataset_path"])
    return [str(file_path.split(".")[0]) for file_path in os.listdir(data_folder)]


class AffinityDataset(Dataset, ABC):
    """Superclass for all protein-protein binding affinity datasets"""

    def __init__(self, config_path: str, dataset_name: str, pdb_ids: List, max_nodes: int = None):
        super(AffinityDataset, self).__init__()
        config = read_yaml(config_path)
        self.data_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                                        config["DATA"][dataset_name]["dataset_path"])
        self.pdb_ids = pdb_ids
        self.max_nodes = max_nodes

    def len(self) -> int:
        return len(self.pdb_ids)

    def _get_edges(self, data_file: NpzFile):
        data_file.close()
        raise NotImplemented

    def _get_node_features(self, data_file: NpzFile):
        data_file.close()
        raise NotImplemented

    def _load_file(self, pdb_id: str) -> NpzFile:
        file_path = os.path.join(self.data_folder, pdb_id + ".npz")
        np_file: NpzFile = np.load(file_path, allow_pickle=True)
        return np_file

    def get(self, idx: int) -> Data:
        pdb_id = self.pdb_ids[idx]
        np_file = self._load_file(pdb_id)
        edge_index, edge_features = self._get_edges(np_file)
        residue_features = self._get_node_features(np_file)
        affinity = np_file["affinity"]
        np_file.close()

        data = Data(
            x=torch.Tensor(residue_features),
            edge_index=torch.Tensor(edge_index).long(),
            edge_attr=torch.Tensor(edge_features),
            y=torch.Tensor(affinity)
        )

        return data


class SimpleGraphs(AffinityDataset, ABC):
    def __init__(self, config_path: str, dataset_name: str, pdb_ids: List):
        super(SimpleGraphs, self).__init__(config_path, dataset_name, pdb_ids)

    def _get_edges(self, data_file: NpzFile) -> Tuple[Tuple[np.ndarray], np.ndarray]:
        adjacency_matrix = data_file["adjacency_tensor"]
        edges = np.where(adjacency_matrix[0, :, :] > 0.001)
        edge_attributes = adjacency_matrix[:, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))
        return np.array(edges), edge_attributes

    def _get_node_features(self, data_file: NpzFile) -> np.ndarray:
        return data_file["residue_features"]


class FixedSizeGraphs(AffinityDataset, ABC):
    def __init__(self, config_path: str, dataset_name: str, pdb_ids: List, max_nodes: int):
        super(FixedSizeGraphs, self).__init__(config_path, dataset_name, pdb_ids, max_nodes=max_nodes)

    def _get_edges(self, data_file: NpzFile) -> Tuple[Tuple[np.ndarray], np.ndarray]:
        adjacency_matrix = data_file["adjacency_tensor"]
        closest_nodes = data_file["closest_residues"][:self.max_nodes]

        closest_nodes_adj = adjacency_matrix[0, closest_nodes, :][:, closest_nodes]

        edges = np.where(closest_nodes_adj[ :, :] > 0.001)
        edge_attributes = adjacency_matrix[:, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))

        return np.array(edges), edge_attributes

    def _get_node_features(self, data_file: NpzFile) -> np.ndarray:
        closest_nodes = data_file["closest_residues"][:self.max_nodes]
        return data_file["residue_features"][closest_nodes]


class DDGBackboneInputs(AffinityDataset, ABC):
    def __init__(self, config_path: str, dataset_name: str, pdb_ids: List, max_nodes: int):
        super(DDGBackboneInputs, self).__init__(config_path, dataset_name, pdb_ids, max_nodes)

    def _get_edges(self, data_file: NpzFile) -> Tuple[Tuple[np.ndarray], np.ndarray]:
        return np.array([]), np.array([])

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