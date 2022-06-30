import os
from abc import ABC

import numpy as np
from numpy.lib.npyio import NpzFile
import torch
from typing import Tuple, Dict, List
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset

from abag_affinity.utils.config import read_yaml, get_data_paths


def get_pdb_ids(config_path: str, dataset_name: str) -> List:
    config = read_yaml(config_path)
    data_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                               config["DATA"][dataset_name]["dataset_path"])
    return [str(file_path.split(".")[0]) for file_path in os.listdir(data_folder)]


class AffinityHeteroDataset(Dataset, ABC):
    """Superclass for all protein-protein binding affinity datasets"""

    def __init__(self, config_path: str, dataset_name: str, pdb_ids: List, max_nodes: int = None):
        super(AffinityHeteroDataset, self).__init__()
        config = read_yaml(config_path)
        self.data_folder = os.path.join(config["DATA"]["path"], config["DATA"][dataset_name]["folder_path"],
                                        config["DATA"][dataset_name]["dataset_path"])
        self.pdb_ids = pdb_ids
        self.max_nodes = max_nodes

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data["aa"], 'num_node_features'):
            return data["aa"].num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

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

    def get(self, idx: int) -> HeteroData:
        pdb_id = self.pdb_ids[idx]
        np_file = self._load_file(pdb_id)
        edge_indices, edge_features = self._get_edges(np_file)
        residue_features = self._get_node_features(np_file)
        affinity = np_file["affinity"]
        np_file.close()

        data = HeteroData()

        data["aa"].x = torch.Tensor(residue_features).double()
        data.y = torch.Tensor(affinity).double()


        for edge_type, edges in edge_indices.items():
            data[edge_type].edge_index = edges
            data[edge_type].edge_attr = edge_features[edge_type]

        return data


class HeteroGraphs(AffinityHeteroDataset, ABC):
    def __init__(self, config_path: str, dataset_name: str, pdb_ids: List):
        super(HeteroGraphs, self).__init__(config_path, dataset_name, pdb_ids)
        self.edge_types = ["distance", "peptide_bond", "same_protein"]

    def _get_edges(self, data_file: NpzFile) -> Tuple[Dict, Dict]:
        adjacency_matrix = data_file["adjacency_tensor"]
        all_edges = {}
        edge_attributes = {}
        distance_idx = self.edge_types.index("distance")
        for i in range(len(adjacency_matrix)):
            edge_type = ("aa", self.edge_types[i], "aa")
            if self.edge_types[i] == "distance":
                edges = np.where(adjacency_matrix[i, :, :] > 0.001)
            elif self.edge_types[i] == "same_protein":
                edges = np.where((adjacency_matrix[i, :, :] == 1) & (adjacency_matrix[distance_idx, :, :] > 0.001))
            else:
                edges = np.where(adjacency_matrix[i, :, :] == 1)
            all_edges[edge_type] = torch.tensor(np.array(edges)).long()

            distance = adjacency_matrix[distance_idx, edges[0], edges[1]]
            edge_attributes[edge_type] = torch.tensor(distance).double()

        interface_edges = np.where((adjacency_matrix[2, :, :] != 1) & (adjacency_matrix[distance_idx, :, :] > 0.001))
        all_edges[("aa", "interface", "aa")] = torch.tensor(interface_edges).long()
        distance = adjacency_matrix[distance_idx, interface_edges[0], interface_edges[1]]
        edge_attributes[("aa", "interface", "aa")] = torch.tensor(distance).double()

        return all_edges, edge_attributes

    def _get_node_features(self, data_file: NpzFile) -> np.ndarray:
        return data_file["residue_features"]


if __name__ == "__main__":
    import pandas as pd
    validation_set = 1
    config_file = "../../abag_affinity/config.yaml"
    dataset_name = "Dataset_v1"

    config = read_yaml(config_file)
    summary_file, pdb_path = get_data_paths(config, dataset_name)
    dataset_summary = pd.read_csv(summary_file)
    val_ids = list(dataset_summary[dataset_summary["validation"] == validation_set]["pdb"].values)

    val_data = HeteroGraphs(config_file, dataset_name, val_ids)

    data = val_data.__getitem__(0)
    n_fets = val_data.num_node_features
    a = 0

