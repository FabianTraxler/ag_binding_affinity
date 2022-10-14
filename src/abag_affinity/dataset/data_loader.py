"""Module providing a basic superclass for all datasets as well as specific datasets for PyTorch geometric data"""
import os
import random
from abc import ABC
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from typing import Tuple, Dict, List, Union
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from parallel import submit_jobs
import itertools
import logging

from abag_affinity.utils.config import get_data_paths
from .utils import scale_affinity, load_graph

logger = logging.getLogger(__name__)


class AffinityDataset(Dataset, ABC):
    """Superclass for all protein-protein binding affinity datasets
    Provides functionality to load dataset csv files and graph generation as well as the item_getter methods
    """

    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue",
                 relative_data: bool = False, save_graphs: bool = False, num_threads: int = 1,
                 force_recomputation: bool = False, scale_values: bool = True, preprocess_data: bool = False):
        super(AffinityDataset, self).__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.node_type = node_type
        self.save_graphs = save_graphs

        self.max_nodes = None  # can be overwritten by subclasses
        self.interface_hull_size = None  # can be overwritten by subclasses
        self.interface_distance_cutoff = 5

        # create path for processed graphs
        self.results_dir = os.path.join(config["PROJECT_ROOT"], config["RESULTS"]["path"])
        self.temp_dir = os.path.join(config["RESULTS"]["processed_graph_path"], node_type, dataset_name)
        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        # create path for clean pdbs
        self.pdb_clean_dir = os.path.join(config["cleaned_pdbs"], dataset_name)
        logger.debug(f"Saving cleaned pdbs in {self.pdb_clean_dir}")
        Path(self.pdb_clean_dir).mkdir(exist_ok=True, parents=True)

        self.force_recomputation = force_recomputation
        self.scale_values = scale_values
        self.preprocess_data = preprocess_data

        # load dataframe with metainfo
        self.data_df = self._load_df(pdb_ids)

        # set dataset to load relative or absolute data points
        self.relative_data = relative_data
        if relative_data:
            self.get = self.get_relative
            self.data_points = self.get_all_pairs()

            self.preprocess_data = True # necessary to preprocess graphs in order to avoid race conditions in workers
        else:
            self.get = self.get_absolute
            self.data_points = pdb_ids

        self.num_threads = num_threads


    def get_all_pairs(self) -> List:
        """ Get all mutation pairs available for a complex that do not have the exact same affinity

        Returns:
            List: All Pairs of mutation data points of the same complex
        """
        grouped_data = self.data_df.groupby("pdb")  # group by pdb_id of complex
        all_data_points = []
        for pdb, data_points in grouped_data.groups.items():
            data_points = data_points.tolist()
            all_combinations = itertools.combinations(data_points, 2)
            # remove all combination that have exactly the same affinity (probably lowest or highest value)
            all_combinations = filter(
                lambda e: self.data_df.loc[e[0], "-log(Kd)"] != self.data_df.loc[e[1], "-log(Kd)"], all_combinations)
            all_data_points.extend(all_combinations)

        #TODO: Remove sampler and use all values
        all_data_points = random.sample(all_data_points, min(5000, len(all_data_points)))

        return all_data_points

    def len(self) -> int:
        """Returns the length of the dataset"""
        return len(self.data_points)

    def preprocess(self):
        """ Preprocess graphs for faster dataloader and avoiding file conflicts during parallel dataloading

        Use given threads to preprocess data and store them on disc

        Returns:
            None
        """

        pdb_info = []

        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        for df_idx, row in self.data_df.iterrows():

            file_path = os.path.join(self.temp_dir, str(df_idx) + ".npz")

            if not os.path.exists(file_path) or self.force_recomputation:
                pdb_info.append((row, file_path))

        logger.debug(f"Preprocessing {len(pdb_info)} graphs with {self.num_threads} threads")

        submit_jobs(self.preload_graphs, pdb_info, self.num_threads)

    def preload_graphs(self, row: pd.Series, out_path: str):
        """ Function to get graph dict and store to disc

        Used by preprocess functionality

        Args:
            row: Dataframe row corresponding to data point
            out_path: path to store the resulting dict

        Returns:
            None
        """

        graph_dict = load_graph(row, self.dataset_name, self.config, self.pdb_clean_dir,
                                node_type=self.node_type,
                                distance_cutoff=self.interface_distance_cutoff,
                                interface_hull_size=self.interface_hull_size,
                                force_recomputation=self.force_recomputation)

        np.savez_compressed(out_path, **graph_dict)

    def _load_df(self, pdb_ids: List):
        """ Load all the dataset information (csv) into a pandas dataframe

        Only consider the pdbs given as argument

        Args:
            pdb_ids: List of pdb ids in this dataset

        Returns:
            pd.DataFrame: Object with all information available for the pdb_ids
        """
        summary_path, _ = get_data_paths(self.config, self.dataset_name)
        summary_df = pd.read_csv(summary_path, index_col=0)

        summary_df = summary_df.loc[pdb_ids]

        return summary_df.fillna("")

    def _get_edges(self, data_file: Dict):
        # needs to be implemented by subclass
        raise NotImplemented

    def _get_node_features(self, data_file: Dict):
        # needs to be implemented by subclass
        raise NotImplemented

    def get_graph_dict(self, df_idx: str) -> Dict:
        """ Get the graph dict for a data point

        Either load from disc or compute anew

        Args:
            df_idx: Index of metadata dataframe

        Returns:
            Dict: graph information for index
        """

        file_path = os.path.join(self.temp_dir, df_idx + ".npz")
        graph_dict = {}

        if os.path.exists(file_path) and (not self.force_recomputation or self.preprocess_data):
            try:
                graph_dict = dict(np.load(file_path, allow_pickle=True))
                compute_graph = False
            except:
                os.remove(file_path)
                compute_graph = True
        else:
            compute_graph = True

        if compute_graph:  # graph not loaded from disc
            row = self.data_df.loc[df_idx]
            graph_dict = load_graph(row, self.dataset_name, self.config, self.pdb_clean_dir,
                                    node_type=self.node_type,
                                    distance_cutoff=self.interface_distance_cutoff,
                                    interface_hull_size=self.interface_hull_size,
                                    force_recomputation=self.force_recomputation)

            if self.save_graphs and not os.path.exists(file_path):
                graph_dict.pop("atom_names", None)  # remove unnecessary information that takes lot of storage
                np.savez_compressed(file_path, **graph_dict)

        return graph_dict

    def load_data_point(self, df_idx: str) -> Data:
        """ Load a data point either from disc or compute it anew

        Standard method for PyTorch geometric graphs

        Can be overwritten by subclasses with a need for differnt data type

        Args:
            df_idx: Index of the dataframe with metadata

        Returns:
            Dict: PyG Data object
        """
        graph_dict = self.get_graph_dict(df_idx)
        residue_features = self._get_node_features(graph_dict)
        edge_index, edge_features = self._get_edges(graph_dict)
        affinity = graph_dict["affinity"]
        if self.scale_values:
            affinity = scale_affinity(affinity)
        affinity = np.array(affinity)

        data = Data(
            x=torch.Tensor(residue_features),
            edge_index=torch.Tensor(edge_index).long(),
            edge_attr=torch.Tensor(edge_features),
            y=torch.from_numpy(affinity).float()
        )

        return data

    def get_absolute(self, idx: int) -> Union[Data, Dict]:
        """ Get the datapoint for a dataset index

        Args:
            idx: Index in the dataset

        Returns:
            Data: PyG data object containing node, edge and label information
        """
        pdb_id = self.data_points[idx]
        data = self.load_data_point(pdb_id)
        return data

    def get_relative(self, idx: int) -> List[Union[Data, Dict]]:
        """ Get the datapoints for a dataset index

        List of two related datapoints for siamese networks

        Args:
            idx: Index in the dataset

        Returns:
            List: List of PyG data object containing node, edge and label information
        """
        data_instances = []

        for pdb_mut in self.data_points[idx]:
            data = self.load_data_point(pdb_mut)
            data_instances.append(data)

        return data_instances

    @property
    def num_node_features(self) -> int:
        """Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, List) else data
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_edge_features(self) -> int:
        """Returns the number of features per edge in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, List) else data
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")


class BoundComplexGraphs(AffinityDataset, ABC):
    """ Dataset supplying a graph for the bound complexes
    Either only the interface hull, n-closest nodes or all residues in the PDB"""

    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, interface_hull_size: int = None,
                 max_nodes: int = None, interface_distance_cutoff: float = 5.0, num_threads: int = 1,
                 node_type: str = "residue", relative_data: bool = False, save_graphs: bool = False,
                 force_recomputation: bool = False, scale_values: bool = False,
                 preprocess_data: bool = False):
        super(BoundComplexGraphs, self).__init__(config, dataset_name, pdb_ids, node_type=node_type,
                                                 relative_data=relative_data, save_graphs=save_graphs,
                                                 force_recomputation=force_recomputation, num_threads=num_threads,
                                                 scale_values=scale_values, preprocess_data=preprocess_data)
        self.interface_hull_size = interface_hull_size
        self.interface_distance_cutoff = interface_distance_cutoff
        self.max_nodes = max_nodes

        # path for storing preloaded graphs
        self.temp_dir = os.path.join(self.temp_dir, f"complex-nodes_{max_nodes}-hull_{interface_distance_cutoff}.{interface_hull_size}")

        if self.save_graphs:
            logger.debug(f"Saving processed graphs in {self.temp_dir}")
            Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        if self.preprocess_data:
            logger.debug(f"Preprocessing graphs for {self.dataset_name}")
            Path(self.temp_dir).mkdir(exist_ok=True, parents=True)
            self.preprocess()

    def _get_edges(self, data_file: Dict) -> Tuple[np.ndarray, np.ndarray]:
        if self.max_nodes:
            adjacency_matrix = data_file["adjacency_tensor"]
            closest_nodes = data_file["closest_residues"][:self.max_nodes]
            closest_nodes_adj = adjacency_matrix[0, closest_nodes, :][:, closest_nodes]
            edges = np.where(closest_nodes_adj[:, :] > 0.001)
            edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
            edge_attributes = np.transpose(edge_attributes, (1, 0))
        elif self.interface_hull_size is not None:
            adjacency_matrix = data_file["adjacency_tensor"]
            adjacency_matrix = adjacency_matrix[:, data_file["interface_nodes"], :][:, :, data_file["interface_nodes"]]
            edges = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)
            edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
            edge_attributes = np.transpose(edge_attributes, (1, 0))
        else:
            adjacency_matrix = data_file["adjacency_tensor"]
            edges = np.where(adjacency_matrix[0, :, :] > 0.001)
            edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
            edge_attributes = np.transpose(edge_attributes, (1, 0))

        return np.array(edges), edge_attributes

    def _get_node_features(self, data_file: Dict) -> np.ndarray:
        if self.max_nodes is not None:  # use n-closest nodes
            closest_nodes = data_file["closest_residues"][:self.max_nodes]
            node_features = data_file["node_features"][closest_nodes]

            if len(closest_nodes) < self.max_nodes:
                missing_nodes = self.max_nodes - len(closest_nodes)
                shape = (missing_nodes,) + node_features.shape[1:]
                node_features = np.vstack([node_features, np.zeros(shape)])

            return node_features
        elif self.interface_hull_size is not None:  # use only nodes in interface hull
            adjacency_matrix = data_file["adjacency_tensor"]
            interface_nodes = np.where(adjacency_matrix[0, :, :] - adjacency_matrix[2, :, :] > 0.001)[0]
            interface_nodes = np.unique(interface_nodes)

            interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < self.interface_hull_size)
            interface_hull_nodes = np.unique(interface_hull[1])

            data_file["interface_nodes"] = interface_hull_nodes

            return data_file["node_features"][interface_hull_nodes]
        else:  # use all nodes
            return data_file["node_features"]
