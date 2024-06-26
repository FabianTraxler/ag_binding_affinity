"""Module providing utilities to load data for specific backbone (pretrained) models"""
import atom3.database as db
from biopandas.pdb import PandasPdb
import pickle
from parallel import submit_jobs
from Bio.PDB.PDBParser import PDBParser
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
from torch_geometric.data import HeteroData, Batch


from .utils import scale_affinity, load_graph, get_hetero_edges
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
                 relative_data: bool = False, save_graphs: bool = False, force_recomputation: bool = False, node_type: str = "residue",
                 preprocess_data: bool = False, num_threads: int = 1,):
        super(DDGBackboneInputs, self).__init__(config=config, dataset_name=dataset_name, pdb_ids=pdb_ids,
                                                node_type=node_type, scale_values=scale_values,
                                                relative_data=relative_data, save_graphs=save_graphs,
                                                force_recomputation=force_recomputation,
                                                preprocess_data=preprocess_data, num_threads=num_threads)

        self.max_nodes = max_nodes

        self.temp_dir = os.path.join(self.temp_dir, f"ddg-nodes_{max_nodes}")

        if self.save_graphs:
            logger.debug(f"Saving processed graphs in {self.temp_dir}")
            Path(self.temp_dir).mkdir(exist_ok=True, parents=True)


        if self.preprocess_data:
            logger.debug(f"Preprocessing graphs for {self.dataset_name}")
            self.preprocess()

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
            closest_nodes = data["closest_residues"][:self.max_nodes].astype(int)
        else:
            closest_nodes = data["closest_residues"].astype(int)

        node_features = []
        for idx in closest_nodes:
            residue_info = data["residue_infos"][idx]
            residue_coords = data["residue_atom_coordinates"][idx, :]
            node_coordinates = residue_coords.flatten()
            coordinate_mask = ~np.all(np.isinf(residue_coords), axis=-1)
            indices = np.array([residue_info["residue_pdb_index"],
                                           residue_info["chain_idx"], residue_info["on_chain_residue_idx"]])
            all_features = np.concatenate([node_coordinates, indices, coordinate_mask])
            node_features.append(all_features)

        node_features = np.array(node_features)

        if len(closest_nodes) < self.max_nodes:
            diff = self.max_nodes - len(closest_nodes)
            node_features = np.vstack(node_features, np.zeros((diff, len(node_features[0]))))

        return node_features


class DeepRefineBackboneInputs(AffinityDataset, ABC):
    """Dataloader providing inputs used by Equivariant Graph Refinement model (DeepRefine)

    Code: https://github.com/BioinfoMachineLearning/DeepRefine
    Paper: https://arxiv.org/abs/2205.10390
    """
    def __init__(self, config: Dict, dataset_name: str, pdb_ids: List,
                 max_nodes: int = None,
                 max_interface_edges: int = None,
                 scale_values: bool = False,
                 preprocess_data: bool = False,
                 relative_data: bool = False,
                 use_heterographs: bool = False,
                 num_threads: int = 1,
                 interface_hull_size: int = None,
                 save_graphs: bool = False,
                 force_recomputation: bool = False):
        super(DeepRefineBackboneInputs, self).__init__(config=config, dataset_name=dataset_name, pdb_ids=pdb_ids,
                                                       node_type="atom", scale_values=scale_values,
                                                       relative_data=relative_data, save_graphs=save_graphs,
                                                       force_recomputation=force_recomputation,
                                                       preprocess_data=preprocess_data, num_threads=num_threads)
        # threads to use for preloading
        self.pdb_parser = PDBParser(PERMISSIVE=3)

        self.max_nodes = max_nodes
        self.max_interface_edges = max_interface_edges
        # interface definition
        self.interface_size = 5
        self.interface_hull_size = interface_hull_size

        # add HeteroGraph Info to Protein Dict
        self.use_heterographs = use_heterographs

        # path for storing preloaded graphs
        self.temp_dir = os.path.join(self.temp_dir, "deeprefine")

        if self.interface_hull_size is not None:
            folder_name = f"hull_{self.interface_hull_size}_graph"
        else:
            folder_name = "complete_graph"

        self.graph_dir = os.path.join(self.temp_dir, folder_name)

        if self.save_graphs:
            Path(self.graph_dir).mkdir(exist_ok=True, parents=True)

        if self.preprocess_data:
            logger.debug(f"Preprocessing graphs for {self.dataset_name}")
            self.preprocess()

    def preprocess(self):
        """ Preprocess graphs for faster dataloader and avoiding file conflicts during parallel dataloading

        Use given threads to preprocess data and store them on disc

        Returns:
            None
        """


        pdb_info = []

        for idx, row in self.data_df.iterrows():
            pdb_path, _ = self.get_path_and_affinity(row)

            graph_filepath = os.path.join(self.graph_dir, idx  + ".pickle")

            if not os.path.exists(graph_filepath) or self.force_recomputation:
                pdb_info.append((idx, pdb_path, row, graph_filepath))

        logger.debug(f"Preprocessing {len(pdb_info)} graphs with {self.num_threads} threads")

        submit_jobs(self.preload_graphs, pdb_info, self.num_threads)

    def preload_graphs(self, filename: str, pdb_filepath: str, row: pd.Series, out_path: str):
        """ Function to get graph dict and store to disc

        Used by preprocess functionality

        Args:
            pdb_filepath: Path of pdb file
            chain_infos: protein information for chains (antibody, antigen)
            out_path: path to store the resulting dict

        Returns:
            None
        """

        chain_infos = literal_eval(row["chain_infos"])
        graph_dict = self.get_graph(filename, pdb_filepath, chain_infos)

        if self.use_heterographs:
            graph_dict["hetero_graph"] = self.get_hetero_graph(row, graph_dict["graph"].ndata["f"])

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

    def tidy_up_pdb_file(self, file_name: str, pdb_filepath: str) -> str:
        """ Clean and remove artefacts in pdb files that lead to errors in DeepRefine

        1. Tidy PDB with pdb-tools
        2. remove multiple models in pdb
        3. remove HETATMs and alternate locations of atoms
        4. Only keep residues that have C, Ca and N atoms (required by DeepRefine)

        Args:
            file_name: Name of the file
            pdb_filepath: Path of the original PDB file

        Returns:
            str: path of the cleaned pdb_file
        """
        """Run 'pdb-tools' to tidy-up and and create a new cleaned file"""
        # Make a copy of the original PDB filepath to circumvent race conditions with 'pdb-tools'
        cleaned_path = os.path.join(self.pdb_clean_dir, file_name + ".pdb")
        Path(cleaned_path).parent.mkdir(exist_ok=True, parents=True)
        if os.path.exists(cleaned_path):
            return cleaned_path
        else:
            clean_and_tidy_pdb("_", pdb_filepath, cleaned_path)
            return cleaned_path

    def reduce2interface_hull(self, file_name: str, pdb_filepath: str, chain_infos: Dict) -> str:
        """ Reduce PDB file to only contain residues in interface-hull

        Interface hull defines as class variable

        1. Get distances between atoms
        2. Get interface atoms
        3. get all atoms in hull around interface
        4. expand to all resiudes that have at least 1 atom in interface hull

        Args:
            file_name: Name of the file
            pdb_filepath: Path of the original pdb file
            chain_infos: Dict with information which chain belongs to which protein (necessary for interface detection)

        Returns:
            str: path to interface pdb file
        """
        head, tail = os.path.split(pdb_filepath)
        interface_path = os.path.join(head, f"interface_hull_{self.interface_hull_size}", file_name + ".pdb")
        if os.path.exists(interface_path):
            return interface_path

        Path(interface_path).parent.mkdir(exist_ok=True, parents=True)

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

        prot_1_idx = atom_df[atom_df["chain_id"].isin(prot_1_chains)].index.to_numpy().astype(int)
        prot_2_idx = atom_df[atom_df["chain_id"].isin(prot_2_chains)].index.to_numpy().astype(int)

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

        assert len(interface_df) > 0, f"No atoms after cleaning in file: {pdb_filepath}"

        pdb.df['ATOM'] = interface_df
        pdb.to_pdb(path=interface_path,
                    records=["ATOM"],
                    gz=False,
                    append_newline=True)

        return interface_path

    def get_graph(self, file_name: str, input_filepath: str, chain_infos: Dict) -> Dict:
        """ Convert PDB file to a graph with node and edge encodings

        Utilize DeepRefine functionality to get graphs

        Args:
            input_filepath: Path to PDB File
            chain_infos: Dict with protein information for each chain

        Returns:
            Dict: Information about graph, protein and filepath of pdb
        """
        # Process the unprocessed protein
        cleaned_path = self.tidy_up_pdb_file(file_name, input_filepath)
        if self.interface_hull_size is not None:
            cleaned_path = self.reduce2interface_hull(file_name, cleaned_path, chain_infos)

        # get graph info with DeepRefine utility
        graph = process_pdb_into_graph(cleaned_path, "all_atom", 20, 8.0)

        # Organize the input graph and its associated metadata as a dictionary
        prot_dict = {
            'deeprefine_graph': graph,
            'filepath': cleaned_path  # TODO fix filepath
        }

        # Combine all distinct node features together into a single node feature tensor
        prot_dict['deeprefine_graph'].ndata['f'] = torch.cat((
            prot_dict['deeprefine_graph'].ndata['atom_type'],
            prot_dict['deeprefine_graph'].ndata['surf_prox']
        ), dim=1)

        # Combine all distinct edge features into a single edge feature tensor
        prot_dict['deeprefine_graph'].edata['f'] = torch.cat((
            prot_dict['deeprefine_graph'].edata['pos_enc'],
            prot_dict['deeprefine_graph'].edata['in_same_chain'],
            prot_dict['deeprefine_graph'].edata['rel_geom_feats'],
            prot_dict['deeprefine_graph'].edata['bond_type']
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

        process_graph = True

        # Define graph metadata for the entire forward-pass pipeline
        graph_filepath = os.path.join(self.graph_dir, df_idx  + ".pickle")
        if (not self.force_recomputation or self.preprocess_data) and os.path.exists(graph_filepath):
            try:
                with open(graph_filepath, 'rb') as f:
                    graph_dict: Dict = pickle.load(f)
                    process_graph = False
            except:
                # recompute graph if saved graph is not parsable
                process_graph = True

        if process_graph:
            graph_dict = self.get_graph(df_idx, input_filepath, chain_infos)
            if self.use_heterographs:
                graph_dict["hetero_graph"] = self.get_hetero_graph(row, graph_dict["graph"].ndata["f"])

            if self.save_graphs:
                with open(graph_filepath, 'wb') as f:
                    pickle.dump(graph_dict, f)

        graph_dict["affinity"] = affinity

        if self.use_heterographs and "hetero_graph" not in graph_dict:
            graph_dict["hetero_graph"] = self.get_hetero_graph(row, graph_dict["graph"].ndata["f"])
        elif not self.use_heterographs:
            graph_dict["hetero_graph"] = HeteroData()

        if self.use_heterographs:
            assert torch.max(graph_dict["hetero_graph"][('node', 'same_residue', 'node')].edge_index[0]) < len(
                graph_dict["hetero_graph"]['node'][
                    "x"]), f"1. Edge Index ({torch.max(graph_dict['hetero_graph'][('node', 'same_residue', 'node')].edge_index[0])}) not matching number of nodes({len(graph_dict['hetero_graph']['node']['x'])} for {df_idx} in dataset {self.dataset_name} using as relativ: {self.relative_data}"
            assert torch.max(graph_dict["hetero_graph"][('node', 'same_residue', 'node')].edge_index[1]) < len(
                graph_dict["hetero_graph"]['node'][
                    "x"]), f"2. Edge Index not matching number of nodes for {df_idx} in dataset {self.dataset_name} using as relativ: {self.relative_data}"

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

    @property
    def num_edge_features(self) -> int:
        """Returns the number of features per node in the dataset."""
        data = self[0]
        graph = data["graph"] if isinstance(data, dict) else data[0]["graph"]
        if hasattr(graph, 'edata'):
            return graph.edata["f"].shape[1]
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
                                "affinity": np.array([input_dict[i]["affinity"] for input_dict in input_dicts]),
                                "hetero_graph": Batch.from_data_list([input_dict[i]["hetero_graph"] for input_dict in input_dicts])}
                batch.append(batched_dict)
            return batch
        else:
            batched_dict = {"graph": dgl.batch([input_dict["graph"] for input_dict in input_dicts]),
                            "filepath": [input_dict["filepath"] for input_dict in input_dicts],
                            "affinity": np.array([input_dict["affinity"] for input_dict in input_dicts]),
                            "hetero_graph": Batch.from_data_list([input_dict["hetero_graph"] for input_dict in input_dicts])}
            return batched_dict

