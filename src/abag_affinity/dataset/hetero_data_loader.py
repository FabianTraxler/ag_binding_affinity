"""Module providing datasets for supplying PyG HeterData objects"""
from abc import ABC
import numpy as np
import torch
from typing import Tuple, Dict, List
from torch_geometric.data import HeteroData
import os
import logging
from pathlib import Path

from .data_loader import AffinityDataset
from .utils import scale_affinity, get_hetero_edges

logger = logging.getLogger(__name__)


class AffinityHeteroDataset(AffinityDataset, ABC):
    """Superclass for all protein-protein binding affinity datasets using hetero PyG graphs

    Differs from superclass by loading the graphs into a HeteroData object
    """

    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue",
                 scale_values: bool = False, relative_data: bool = False, save_graphs: bool = False,
                 force_recomputation: bool = False, preprocess_data: bool = False, num_threads: int = 1):
        super(AffinityHeteroDataset, self).__init__(config, dataset_name, pdb_ids,
                                                    node_type=node_type,
                                                    num_threads=num_threads,
                                                    scale_values=scale_values,
                                                    relative_data=relative_data,
                                                    save_graphs=save_graphs,
                                                    force_recomputation=force_recomputation,
                                                    preprocess_data=preprocess_data)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, List) else data
        if hasattr(data["node"], 'num_node_features'):
            return data["node"].num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        return 1 # only one feature (distance) per edge



    def load_data_point(self, df_idx: str) -> HeteroData:
        """ Load a data point either from disc or compute it anew

        Args:
            df_idx: Index of the dataframe with metadata

        Returns:
            Dict: PyG Data object
        """
        graph_dict = self.get_graph_dict(df_idx)
        edge_indices, edge_features = self._get_edges(graph_dict)
        node_features = self._get_node_features(graph_dict)
        affinity = graph_dict["affinity"]
        if self.scale_values:
            affinity = scale_affinity(affinity)
        affinity = np.array(affinity)

        data = HeteroData()

        data["node"].x = torch.Tensor(node_features).double()
        data.y = torch.from_numpy(affinity).double()

        for edge_type, edges in edge_indices.items():
            data[edge_type].edge_index = edges
            data[edge_type].edge_attr = edge_features[edge_type]

        return data


class HeteroGraphs(AffinityHeteroDataset, ABC):
    """HeteroGraphs for residue based graphs

    Different edges if below distance-cutoff, peptide bond and same protein
    """
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, node_type: str = "residue",
                 max_nodes: int = None, scale_values: bool = True, relative_data: bool = False,
                 save_graphs: bool = False, force_recomputation: bool = False, interface_distance_cutoff: float = 5.0,
                 preprocess_data: bool = False,  num_threads: int = 1):
        super(HeteroGraphs, self).__init__(config, dataset_name, pdb_ids, node_type, relative_data=relative_data,
                                           scale_values=scale_values, save_graphs=save_graphs,
                                           force_recomputation=force_recomputation,
                                           preprocess_data=preprocess_data,
                                           num_threads=num_threads)
        self.max_nodes = max_nodes

        if node_type == "residue":
            self.edge_types = ["proximity", "peptide_bond", "same_protein"]
        elif node_type == "atom":
            self.edge_types = ["proximity", "same_residue", "same_protein"]

        self.interface_distance_cutoff = interface_distance_cutoff

        # path for storing preloaded graphs
        self.temp_dir = os.path.join(self.temp_dir, f"{node_type}_hetero_graphs")

        if self.save_graphs:
            Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        if self.preprocess_data:
            logger.debug(f"Preprocessing graphs for {self.dataset_name}")
            self.preprocess()


    def _get_edges(self, data_file: Dict) -> Tuple[Dict, Dict]:
        """ Function to extract edge information based on graph dict

        Extract residue edges :
        - based on distance: < distance cutoff (interface_distance_cutoff)
        - based on residues: has peptide bond to other residue (neighbor)
        - based on residues: is in same protein

        Extract atom edges :
        - based on distance: < distance cutoff (interface_distance_cutoff)
        - based on atoms: is in same residue
        - based on residues: has peptide bond to residue of other atom (neighbor)
        - based on residues: is in same protein


        Args:
            data_file: Dict containing all relevant information about the complex

        Returns:
            Tuple: Edge Dict with indices, edge dict with attributes
        """
        all_edges, edge_attributes = get_hetero_edges(data_file, self.edge_types)

        return all_edges, edge_attributes

    def _get_node_features(self, data_file: Dict) -> np.ndarray:
        """ Extract residue features from data dict

        Args:
            data_file: Dict containing all relevant information about the complex

        Returns:
            np.ndarray: Residue encodings
        """
        return data_file["node_features"]
