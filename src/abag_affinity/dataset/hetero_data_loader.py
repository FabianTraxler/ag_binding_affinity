import os
from abc import ABC
import shutil
import numpy as np
import pandas as pd
from numpy.lib.npyio import NpzFile
import torch
from typing import Tuple, Dict, List
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset
from pathlib import Path

from abag_affinity.utils.config import read_yaml, get_data_paths, get_resources_paths
from abag_affinity.dataset.graph_generator import load_graph


class AffinityHeteroDataset(Dataset, ABC):
    """Superclass for all protein-protein binding affinity datasets"""

    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue", max_nodes: int = None,
                 relative_data: bool = False, save_graphs: bool = True, load_from_disc: bool = True, force_recomputation: bool = False):
        super(AffinityHeteroDataset, self).__init__()

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
        if hasattr(data["aa"], 'num_node_features'):
            return data["aa"].num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    def len(self) -> int:
        return len(self.pdb_ids)

    def _get_edges(self, data_file: Dict):
        raise NotImplemented

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
            # summary_df = summary_df[summary_df["Hold_out_type"] == "AB/AG"]
            summary_df["pdb_mutation"] = summary_df.apply(
                lambda row: row["pdb"].upper() + "_" + row["Mutation(s)_cleaned"], axis=1)
            summary_df.set_index("pdb_mutation", inplace=True, drop=False)
            summary_df = summary_df[~summary_df.index.duplicated(keep='first')]

        else:
            raise ValueError("Invalid Dataset Type given (Dataset_v1, PDBBind, SKEMPI.v2")

        return summary_df

    def _get_node_features(self, data_file: Dict):
        raise NotImplemented

    def get_graph_dict(self, pdb_id: str, mutated_complex: str = None, distance_cutoff: int = 5) -> Dict:

        if mutated_complex is not None:
            pdb_code, mutation_code = pdb_id.split("_")
            if mutated_complex == "wildtype":
                file_path = os.path.join(self.temp_dir, pdb_code + mutated_complex + ".npz")
            else:
                file_path = os.path.join(self.temp_dir, "_".join([pdb_code, mutation_code, mutated_complex + ".npz"]))
        else:
            file_path = os.path.join(self.temp_dir, pdb_id + ".npz")

        if self.load_from_disc and os.path.exists(file_path):
            graph_dict = dict(np.load(file_path, allow_pickle=True))
        else:
            row = self.data_df.loc[pdb_id]
            graph_dict = load_graph(row, self.dataset_name, self.config, node_type=self.node_type,
                                    distance_cutoff=distance_cutoff, mutated_complex=mutated_complex)

        if self.save_graphs and not os.path.exists(file_path):
            np.savez_compressed(file_path, **graph_dict)

        return graph_dict

    def get_absolute(self, idx: int) -> HeteroData:
        pdb_id = self.pdb_ids[idx]
        graph_dict = self.get_graph_dict(pdb_id)
        edge_indices, edge_features = self._get_edges(graph_dict)
        residue_features = self._get_node_features(graph_dict)
        affinity = np.array(graph_dict["affinity"])

        data = HeteroData()

        data["aa"].x = torch.Tensor(residue_features).double()
        data.y = torch.from_numpy(affinity).double()


        for edge_type, edges in edge_indices.items():
            data[edge_type].edge_index = edges
            data[edge_type].edge_attr = edge_features[edge_type]

        return data

    def get_relative(self, idx: int) -> Tuple[HeteroData, HeteroData]:
        pdb_mutation_code = self.pdb_ids[idx]
        wildtype_graph_dict = self.get_graph_dict(pdb_mutation_code, "wildtype")
        wt_edge_indices, wt_edge_features = self._get_edges(wildtype_graph_dict)
        wt_residue_features = self._get_node_features(wildtype_graph_dict)
        wt_affinity = np.array(wildtype_graph_dict["affinity"])

        wildtype = HeteroData()
        wildtype["aa"].x = torch.Tensor(wt_residue_features).double()
        wildtype.y = torch.from_numpy(wt_affinity).double()

        for edge_type, edges in wt_edge_indices.items():
            wildtype[edge_type].edge_index = edges
            wildtype[edge_type].edge_attr = wt_edge_features[edge_type]


        mutation_graph_dict = self.get_graph_dict(pdb_mutation_code, self.config["DATA"][self.dataset_name]["mutated_pdb_path"])
        mut_edge_indices, mut_edge_features = self._get_edges(mutation_graph_dict)
        mut_residue_features = self._get_node_features(mutation_graph_dict)
        mut_affinity = np.array(mutation_graph_dict["affinity"])

        mutation = HeteroData()
        mutation["aa"].x = torch.Tensor(mut_residue_features).double()
        mutation.y = torch.from_numpy(mut_affinity).double()

        for edge_type, edges in mut_edge_indices.items():
            mutation[edge_type].edge_index = edges
            mutation[edge_type].edge_attr = mut_edge_features[edge_type]

        return wildtype, mutation


class HeteroGraphs(AffinityHeteroDataset, ABC):
    def __init__(self,config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue", max_nodes: int = None,
                 relative_data: bool = False, save_graphs: bool = True, load_from_disc: bool = True,
                 force_recomputation: bool = False):
        super(HeteroGraphs, self).__init__(config, dataset_name, pdb_ids, node_type, max_nodes, relative_data,
                                           save_graphs, load_from_disc, force_recomputation)
        self.edge_types = ["distance", "peptide_bond", "same_protein"]

    def _get_edges(self, data_file: Dict) -> Tuple[Dict, Dict]:
        adjacency_matrix = data_file["adjacency_tensor"]
        all_edges = {}
        edge_attributes = {}
        distance_idx = self.edge_types.index("distance")
        for i in range(len(self.edge_types)):
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

    def _get_node_features(self, data_file: Dict) -> np.ndarray:
        return data_file["node_features"]


