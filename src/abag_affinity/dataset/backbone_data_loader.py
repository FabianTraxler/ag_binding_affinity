"""Module providing utilities to load data for specific backbone (pretrained) models"""
import atom3.database as db
import subprocess
from biopandas.pdb import PandasPdb
import pickle
from parallel import submit_jobs
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
import scipy.spatial as sp
from ast import literal_eval
import os
from abc import ABC
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from typing import Tuple, Dict, List, Union
import shutil
import logging
import dgl

from .utils import scale_affinity
from .data_loader import AffinityDataset

from abag_affinity.utils.pdb_processing import clean_and_tidy_pdb

# DeepRefine Imports
from project.utils.deeprefine_utils import process_pdb_into_graph


logger = logging.getLogger(__name__)  # setup module logger


class DDGBackboneInputs(AffinityDataset, ABC):
    """Dataloader providing inputs used by Binding DDG Predictor

    Code: https://github.com/HeliXonProtein/binding-ddg-predictor)
    Paper: https://www.pnas.org/doi/10.1073/pnas.2122954119
    """
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, max_nodes: int = None, scale_values: bool = False,
                 relative_data: bool = False, save_graphs: bool = False, force_recomputation: bool = False):
        super(DDGBackboneInputs, self).__init__(config=config, dataset_name=dataset_name, pdb_ids=pdb_ids,
                                                       node_type="atom", scale_values=scale_values,
                                                       relative_data=relative_data, save_graphs=save_graphs,
                                                       force_recomputation=force_recomputation)

        self.max_nodes = max_nodes

    def _get_edges(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """ Converts the graph dictionary to edges and edge embedding

        Args:
            data: Dict containing information about the complex graph

        Returns:
            Tuple: array with edge (node indices), array with edge encodings
        """
        adjacency_matrix = data["adjacency_tensor"]
        if self.max_nodes is not None:
            closest_nodes = data["closest_residues"][:self.max_nodes]
        else:
            closest_nodes = data["closest_residues"]
        closest_nodes_adj = adjacency_matrix[0, closest_nodes, :][:, closest_nodes]

        # use only edges in interface
        edges = np.where(closest_nodes_adj[ :, :] > 0.001)
        edge_attributes = adjacency_matrix[:3, edges[0], edges[1]]
        edge_attributes = np.transpose(edge_attributes, (1, 0))

        return np.array(edges), edge_attributes

    def _get_node_features(self, data: Dict) -> np.ndarray:
        """ Convert graph information to node encodings for every residue

        Shape: (N, 14 * 3 + 1 + 1 + 1 + 14)
        Info: (Num_Nodes, Atom Coords * (x,y,z) + aa_index + seq_index, + chain_seq_index + atom_coordinate_mask)

        Args:
            data: Dict containing information about the complex graph

        Returns:
            np.array: Node encodings, shape: (N, 14 * 3 + 1 + 1 + 1 + 14)
        """
        if self.max_nodes is not None:
            closest_nodes = data["closest_residues"][:self.max_nodes]
        else:
            closest_nodes = data["closest_residues"]

        infos = data["residue_infos"][closest_nodes]
        coords = data["residue_atom_coordinates"][closest_nodes, :]
        node_features = []
        for residue_info, residue_coords in zip(infos, coords):
            node_coordinates = residue_coords.flatten()
            coordinate_mask = ~np.all(np.isinf(residue_coords), axis=-1)
            indices = np.array([residue_info["residue_pdb_index"],
                                           residue_info["chain_idx"], residue_info["on_chain_residue_idx"]])
            all_features = np.concatenate([node_coordinates, indices, coordinate_mask])
            node_features.append(all_features)

        return np.array(node_features)


class DeepRefineBackboneInputs(AffinityDataset, ABC):
    """Dataloader providing inputs used by Equivariant Graph Refinement model (DeepRefine)

    Code: https://github.com/BioinfoMachineLearning/DeepRefine
    Paper: https://arxiv.org/abs/2205.10390
    """
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List, max_nodes: int = None, scale_values: bool = False,
                 preprocess_data: bool = False, relative_data: bool = False, num_threads: int = 1,
                 interface_hull_size: int = None, save_graphs: bool = False, force_recomputation: bool = False):
        super(DeepRefineBackboneInputs, self).__init__(config=config, dataset_name=dataset_name, pdb_ids=pdb_ids,
                                                       node_type="atom", scale_values=scale_values,
                                                       relative_data=relative_data, save_graphs=save_graphs,
                                                       force_recomputation=force_recomputation)
        # threads to use for preloading
        self.num_threads = num_threads
        self.pdb_parser = PDBParser(PERMISSIVE=3)

        self.max_nodes = max_nodes

        # interface definition
        self.interface_size = 5
        self.interface_hull_size = interface_hull_size

        # path for storing preloaded graphs
        self.temp_dir = os.path.join(self.temp_dir, "deeprefine")
        if os.path.exists(self.temp_dir) and self.force_recomputation:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.save_graphs:
            Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        if preprocess_data:
            logger.debug(f"Preprocessing graphs for {self.dataset_name}")
            self.preprocess()

    def preprocess(self):
        """ Preprocess graphs for faster dataloader and avoiding file conflicts during parallel dataloading

        Use given threads to preprocess data and store them on disc

        Returns:
            None
        """
        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)
        pdb_info = []

        for idx, row in self.data_df.iterrows():
            pdb_path, _ = self.get_path_and_affinity(row)

            graph_filepath = os.path.join(self.temp_dir, idx  + ".pickle")

            if not os.path.exists(graph_filepath):
                pdb_info.append((pdb_path, literal_eval(row["chain_infos"]), graph_filepath))

        logger.debug(f"Preprocessing {len(pdb_info)} graphs with {self.num_threads} threads")

        submit_jobs(self.preload_graphs, pdb_info, self.num_threads)

    def preload_graphs(self, pdb_filepath: str, chain_infos: Dict, out_path: str):
        """ Function to get graph dict and store to disc

        Used by preprocess functionality

        Args:
            pdb_filepath: Path of pdb file
            chain_infos: protein information for chains (antibody, antigen)
            out_path: path to store the resulting dict

        Returns:
            None
        """
        graph_dict = self.get_prot_dict(pdb_filepath, chain_infos)

        with open(out_path, 'wb') as f:
            pickle.dump(graph_dict, f)

    def get_path_and_affinity(self, row: pd.Series) -> Tuple[str, float]:
        """ Extract the filepath of the pdb file and the affinity of this specific complex from a pandas series

        Affinity is scaled if required

        Args:
            row: pd Series with information about the complex

        Returns:
            Tuple: path to the pdb file, affinity
        """
        data_location = row["data_location"]
        if "mutation_code" in row and row["mutation_code"] != "":
            pdb_file_path = os.path.join(self.config[data_location]["path"],
                                    self.config[data_location][self.dataset_name]["folder_path"],
                                    self.config[data_location][self.dataset_name]["mutated_pdb_path"])
        else:
            pdb_file_path = os.path.join(self.config[data_location]["path"],
                                    self.config[data_location][self.dataset_name]["folder_path"],
                                    self.config[data_location][self.dataset_name]["pdb_path"])
        pdb_file_path = os.path.join(pdb_file_path, row["filename"])
        if self.scale_values:
            affinity = scale_affinity(row["-log(Kd)"])
        else:
            affinity = row["-log(Kd)"]
        return pdb_file_path, affinity

    def tidy_up_pdb_file(self, pdb_filepath: str) -> str:
        """ Clean and remove artefacts in pdb files that lead to errors in DeepRefine

        1. Tidy PDB with pdb-tools
        2. remove multiple models in pdb
        3. remove HETATMs and alternate locations of atoms
        4. Only keep residues that have C, Ca and N atoms (required by DeepRefine)

        Args:
            pdb_filepath: Path of the original PDB file

        Returns:
            str: path of the cleaned pdb_file
        """
        """Run 'pdb-tools' to tidy-up and and create a new cleaned file"""
        # Make a copy of the original PDB filepath to circumvent race conditions with 'pdb-tools'
        head, tail = os.path.split(pdb_filepath)
        cleaned_path = os.path.join(self.temp_dir, "cleaned", tail)
        Path(cleaned_path).parent.mkdir(exist_ok=True, parents=True)
        if os.path.exists(cleaned_path) and not self.force_recomputation:
            return cleaned_path
        else:
            clean_and_tidy_pdb("_", pdb_filepath, cleaned_path)
            return cleaned_path

    def reduce2interface_hull(self, pdb_filepath: str, chain_infos: Dict) -> str:
        """ Reduce PDB file to only contain residues in interface-hull

        Interface hull defines as class variable

        1. Get distances between atoms
        2. Get interface atoms
        3. get all atoms in hull around interface
        4. expand to all resiudes that have at least 1 atom in interface hull

        Args:
            pdb_filepath: Path of the original pdb file
            chain_infos: Dict with information which chain belongs to which protein (necessary for interface detection)

        Returns:
            str: path to interface pdb file
        """
        head, tail = os.path.split(pdb_filepath)
        interface_path = os.path.join(head, "interface_hull_only", tail)
        Path(interface_path).parent.mkdir(exist_ok=True, parents=True)
        if os.path.exists(interface_path) and not self.force_recomputation:
            return interface_path

        pdb = PandasPdb().read_pdb(pdb_filepath)
        atom_df = pdb.df['ATOM']

        atom_df["chain_id"] = atom_df["chain_id"].str.upper()

        prot_1_chains = []
        prot_2_chains = []
        for chain, prot in chain_infos.items():
            if prot == 0:
                prot_1_chains.append(chain.upper())
            elif prot == 1:
                prot_2_chains.append(chain.upper())
            else:
                print("Error while loading interface hull - more than two proteins")

        # calcualte distances
        coords = atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy()
        distances = sp.distance_matrix(coords, coords)

        prot_1_idx = atom_df[atom_df["chain_id"].isin(prot_1_chains)].index.to_numpy()
        prot_2_idx = atom_df[atom_df["chain_id"].isin(prot_2_chains)].index.to_numpy()

        # get interface
        abag_distance = distances[prot_1_idx, :][:, prot_2_idx]
        interface_connections = np.where(abag_distance < self.interface_size)
        prot_1_interface = prot_1_idx[np.unique(interface_connections[0])]
        prot_2_interface = prot_2_idx[np.unique(interface_connections[1])]

        # get interface hull
        interface_atoms = np.concatenate([prot_1_interface, prot_2_interface])
        interface_hull = np.where(distances[interface_atoms, :] < self.interface_hull_size)[1]
        interface_hull = np.unique(interface_hull)

        # use complete residues if one of the atoms is in hull
        interface_residues = atom_df.iloc[interface_hull][["chain_id", "residue_number"]].drop_duplicates()
        interface_df = atom_df.merge(interface_residues)

        pdb.df['ATOM'] = interface_df
        pdb.to_pdb(path=interface_path,
                    records=["ATOM"],
                    gz=False,
                    append_newline=True)

        return interface_path

    def get_prot_dict(self, input_filepath: str, chain_infos: Dict) -> Dict:
        """ Convert PDB file to a graph with node and edge encodings

        Utilize DeepRefine functionality to get graphs

        Args:
            input_filepath: Path to PDB File
            chain_infos: Dict with protein information for each chain

        Returns:
            Dict: Information about graph, protein and filepath of pdb
        """
        # Process the unprocessed protein
        cleaned_path = self.tidy_up_pdb_file(input_filepath)
        if self.interface_hull_size is not None:
            cleaned_path = self.reduce2interface_hull(cleaned_path, chain_infos)

        # get graph info with DeepRefine utility
        graph = process_pdb_into_graph(cleaned_path, "all_atom", 20, 8.0)

        # Organize the input graph and its associated metadata as a dictionary
        prot_dict = {
            'protein': db.get_pdb_name(cleaned_path),
            'graph': graph,
            'filepath': cleaned_path
        }

        # Combine all distinct node features together into a single node feature tensor
        prot_dict['graph'].ndata['f'] = torch.cat((
            prot_dict['graph'].ndata['atom_type'],
            prot_dict['graph'].ndata['surf_prox']
        ), dim=1)

        # Combine all distinct edge features into a single edge feature tensor
        prot_dict['graph'].edata['f'] = torch.cat((
            prot_dict['graph'].edata['pos_enc'],
            prot_dict['graph'].edata['in_same_chain'],
            prot_dict['graph'].edata['rel_geom_feats'],
            prot_dict['graph'].edata['bond_type']
        ), dim=1)

        return prot_dict

    def load_data_point(self, df_idx: str) -> Dict:
        """ Load a data point either from disc or compute it anew

        Args:
            df_idx: Index of the dataframe with metadata

        Returns:
            Dict: Graph, Filepath and affinity information
        """
        row = self.data_df.loc[df_idx]
        chain_infos = literal_eval(row["chain_infos"])
        input_filepath, affinity = self.get_path_and_affinity(row)

        # Define graph metadata for the entire forward-pass pipeline
        graph_filepath = os.path.join(self.temp_dir, df_idx  + ".pickle")
        if not self.force_recomputation and os.path.exists(graph_filepath):
            with open(graph_filepath, 'rb') as f:
                graph_dict: Dict = pickle.load(f)
        else:
            graph_dict = self.get_prot_dict(input_filepath, chain_infos)
            if self.save_graphs:
                with open(graph_filepath, 'wb') as f:
                    pickle.dump(graph_dict, f)

        graph_dict["affinity"] = affinity

        return graph_dict

    @property
    def num_node_features(self) -> int:
        """Returns the number of features per node in the dataset."""
        data = self[0]
        graph = data["graph"] if isinstance(data, dict) else data[0]["graph"]
        if hasattr(graph, 'ndata'):
            return graph.ndata["f"].shape[1]
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @staticmethod
    def collate(input_dicts: List[Dict]) -> Union[List, Dict]:
        """ Collate multiple datapoints to a batch

        1. Batch the graphs (using DGL Batch)
        2. Batch the filepaths
        3. Batch the affinity values (using numpy)

        Args:
            input_dicts: List of input data points

        Returns:
            Dict: Containing the batches data points
        """

        if not isinstance(input_dicts[0], dict): # relative data
            batch = []
            for i in range(2):
                batched_dict = {"graph": dgl.batch([input_dict[i]["graph"] for input_dict in input_dicts]),
                                "filepath": [input_dict[i]["filepath"] for input_dict in input_dicts],
                                "affinity": np.array([input_dict[i]["affinity"] for input_dict in input_dicts])}
                batch.append(batched_dict)
            return batch
        else:
            batched_dict = {"graph": dgl.batch([input_dict["graph"] for input_dict in input_dicts]),
                            "filepath": [input_dict["filepath"] for input_dict in input_dicts],
                            "affinity": np.array([input_dict["affinity"] for input_dict in input_dicts])}
            return batched_dict

