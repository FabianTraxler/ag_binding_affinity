"""
Module providing a PyTorch dataset class for binding affinity data
"""
import os
import random
from abc import ABC
from torch_geometric.data import HeteroData, Batch, Dataset
import dgl
from dgl.subgraph import node_subgraph
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from parallel import submit_jobs
import scipy.spatial as sp
import logging
from ast import literal_eval
import pickle
# import pdb

from ..utils.config import get_data_paths
from .utils import scale_affinity, load_graph_dict, get_hetero_edges, get_pdb_path_and_id, load_deeprefine_graph

logger = logging.getLogger(__name__)


class AffinityDataset(Dataset):
    """Dataset class all protein-protein binding affinity datasets

    Provides functionality to:
     - load dataset csv files
     - generate graphs (atom or residue) from PDB Files
     - handle relative binding affinity data
    """

    def __init__(self, config: Dict,
                 is_relaxed: bool,
                 dataset_name: str,
                 pdb_ids: Optional[List] = None,
                 node_type: str = "residue",
                 max_nodes: Optional[int] = None,
                 interface_distance_cutoff: int = 5,
                 interface_hull_size: Optional[int] = None,
                 max_edge_distance: int = 5,
                 pretrained_model: str = "",
                 scale_values: bool = True, scale_min: int = 0, scale_max: int = 16,
                 relative_data: bool = False,
                 save_graphs: bool = False, force_recomputation: bool = False,
                 preprocess_data: bool = False, num_threads: int = 1,
                 load_embeddings: Union[bool, str] = False
                 ):
        """ Initialization of class variables,
        generation of necessary directories,
        start of graph preprocessing

        Args:
            config: Dict with data configuration
            is_relaxed: Boolean indicator if relaxed structures are used
            dataset_name: Name of the dataset
            pdb_ids: List of PDB IDs to use in this dataset
            node_type: Type of graph nodes
            max_nodes: Maximal number of nodes
            interface_distance_cutoff: Maximal interface distance
            interface_hull_size: Size of the interface hull extension
            max_edge_distance: Maximal edge distance between two nodes
            pretrained_model: Pretrained model used for prediction
            scale_values: Boolean indicator if affinity values should be scaled
            scale_min: Minimum value of scaling range
            scale_max: Maximum value of scaling range
            relative_data: Boolean indicator if this dataset is relative
            save_graphs: Boolean indicator if processed graphs are stored to disc
            force_recomputation: Boolean indicator if graphs are newly computed even if they are found on disc
            preprocess_data: Boolean indicator if data should be processed after class initialization
            num_threads: Number of threads to use for preprocessing
        """
        super(AffinityDataset, self).__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.node_type = node_type

        self.save_graphs = save_graphs # store graphs to disc
        self.pretrained_model = pretrained_model # if pretrained model is used

        self.max_nodes = max_nodes
        self.interface_hull_size = interface_hull_size
        self.interface_distance_cutoff = interface_distance_cutoff
        self.max_edge_distance = max_edge_distance
        self.scale_values = scale_values
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.load_embeddings = load_embeddings

        if "-" in dataset_name: # part of DMS dataset
            dataset_name, publication_code = dataset_name.split("-")
            self.affinity_type = self.config["DATASETS"][dataset_name]["affinity_types"][publication_code]

            if self.affinity_type == "E" and not relative_data:
                logging.warning("Enrichment values are used 'absolutely'")
            self.dataset_name = dataset_name
            self.publication = publication_code
            self.full_dataset_name = dataset_name + "-" + publication_code
        else:
            self.affinity_type = self.config["DATASETS"][dataset_name]["affinity_type"]
            self.publication = None
            self.full_dataset_name = dataset_name

        # define edge types based on node type
        if node_type == "residue":
            self.edge_types = ["distance", "peptide_bond", "same_protein"]
        elif node_type == "atom":
            self.edge_types = ["distance", "same_residue", "same_protein"]
        else:
            raise ValueError(f"Invalid node type provided - got {node_type} - supported 'residue', 'atom'")

        # create path for results and processed graphs
        self.results_dir = os.path.join(self.config["PROJECT_ROOT"], self.config["RESULTS"]["path"])
        self.graph_dir = os.path.join(self.config["processed_graph_path"], self.full_dataset_name, node_type, pretrained_model, f"embeddings_{load_embeddings}_relaxed_{is_relaxed}")
        self.processed_graph_files = os.path.join(self.graph_dir, "{}.npz")
        if self.save_graphs or preprocess_data:
            logger.debug(f"Saving processed graphs in {self.graph_dir}")
            Path(self.graph_dir).mkdir(exist_ok=True, parents=True)
        # create path for clean pdbs
        self.interface_dir = os.path.join(self.config["interface_pdbs"], self.full_dataset_name)
        logger.debug(f"Saving cleaned pdbs in {self.interface_dir}")
        Path(self.interface_dir).mkdir(exist_ok=True, parents=True)

        self.force_recomputation = force_recomputation
        self.preprocess_data = preprocess_data

        # load dataframe with metainfo
        self.data_df, pdb_ids = self.load_df(pdb_ids)
        self.pdb_ids = pdb_ids

        # set dataset to load relative or absolute data points
        self.relative_data = relative_data
        if relative_data:
            self.get = self.get_relative
            logger.info(f"Forcing graph preprocessing, in order to avoid race conditions in workers")
            self.preprocess_data = True
            self.update_valid_pairs()  # should not be necessary in theory, but __repr__ calls len(), which requires the pairs
        else:
            self.get = self.get_absolute

        self.num_threads = num_threads

        if self.preprocess_data:
            logger.debug(f"Preprocessing {node_type}-graphs for {self.full_dataset_name}")
            self.preprocess()

    def update_valid_pairs(self):
        """ Find for each data point a valid partner
        If no partner is found then ignore the datapoint

        Criteria are:
        - If -log(Kd) values available: Different PDB ID
        - If E values available: Difference of E must be higher than average NLL

        Args:
            pdb_ids: List of PDB Ids to find a partner for

        Returns:
            List: All Pairs of mutation data points of the same complex
        """
        assert self.relative_data, "This function is only available for relative data"
        all_data_points = []
        for pdb_id in self.pdb_ids:
            other_pdb_id = self.get_compatible_pair(pdb_id)
            if other_pdb_id is None:
                # do not use this pdb
                continue
            all_data_points.append((pdb_id, other_pdb_id))

        logger.debug(f"There are in total {len(all_data_points)} valid pairs")

        self.relative_pairs = all_data_points

    def get_compatible_pair(self, pdb_id: str) -> str:
        """ Find a compatible data point for a PDB ID

        Criteria are:
        - -log(Kd) values available: Different PDB ID
        - E values available: Difference of E must be higher than average NLL

        Args:
            pdb_id: PDB ID to find a partner for

        Returns:
            str: PDB ID of partner or None if there is no valid partner
        """
        pdb_file = self.data_df.loc[pdb_id, "pdb"]
        if self.affinity_type == "-log(Kd)":
            other_mutations = self.data_df[(self.data_df["pdb"] == pdb_file) & (self.data_df.index != pdb_id)]
            possible_partner = other_mutations.index.tolist()
        elif self.affinity_type == "E":
            # get data point that has distance > avg(NLL) from current data point
            pdb_nll = self.data_df.loc[self.data_df.index == pdb_id, "NLL"].values[0]
            pdb_e = np.array(self.data_df.loc[self.data_df.index == pdb_id, "E"].values[0]).reshape(-1, 1)

            e_values = self.data_df["E"].values.reshape(-1, 1)
            nll_values = self.data_df["NLL"].values
            e_dists = sp.distance.cdist(pdb_e, e_values)[0, :]
            nll_avg = (pdb_nll + nll_values) / 2

            valid_pairs = (e_dists - nll_avg) >= 0
            valid_partner = np.where(valid_pairs)[0]
            possible_partner = self.data_df.index[valid_partner].tolist()

        else:
            raise ValueError(
                f"Wrong affinity type given - expected one of (-log(Kd), E) but got {self.affinity_type}")

        if len(possible_partner) > 0:  # random choose one of the available mutations
            return random.sample(possible_partner, 1)[0]
        else:  # compare to oneself if there is no other option available
            return None

    def len(self) -> int:
        """Returns the length of the dataset"""
        if self.relative_data:
            return len(self.relative_pairs)
        else:
            return len(self.pdb_ids)

    def get(self, idx: int):
        """
        Needs to be added, otherwise, there is an abstract class->get not implemented error
        """
        raise NotImplementedError


    def preprocess(self):
        """ Preprocess graphs to reduce redundant processing during training +
        avoid file conflicts during parallel dataloading

        Use given threads to preprocess data and store them on disc

        Returns:
            None
        """
        Path(self.graph_dir).mkdir(exist_ok=True, parents=True)
        if self.interface_hull_size is not None and self.interface_hull_size != "" and self.interface_hull_size != "None":
            Path(os.path.join(self.graph_dir, f"interface_hull_{self.interface_hull_size}")).mkdir(exist_ok=True,
                                                                                                   parents=True)
        graph_dicts2process = []
        deeprefine_graphs2process = []
        for df_idx, row in self.data_df.iterrows():
            # Pre-Load Dictionary containing all relevant information to generate graphs
            file_path = self.processed_graph_files.format(df_idx)
            if not os.path.exists(file_path) or self.force_recomputation:
                graph_dicts2process.append((row, file_path))

            if "pdb" not in row:
                row["pdb"] = "-".join((row["publication"], row["ab_ag"]))
            pdb_path, _ = get_pdb_path_and_id(row, self.dataset_name, self.config)

            # get final file name of graph
            if self.interface_hull_size is None or self.interface_hull_size == "" or self.interface_hull_size == "None":
                graph_filepath = os.path.join(self.graph_dir, str(df_idx) + ".pickle")
            else:
                graph_filepath = os.path.join(self.graph_dir, f"interface_hull_{self.interface_hull_size}",
                                              str(df_idx) + ".pickle")
            # Pre-Load DeepRefine graphs

            if not os.path.exists(graph_filepath) or self.force_recomputation:
                deeprefine_graphs2process.append((df_idx, pdb_path, row))

        # start processing with given threads
        logger.debug(f"Preprocessing {len(graph_dicts2process)} graph dicts with {self.num_threads} threads")

        def preload_graph_dict(row: pd.Series, out_path: str):
            """ Function to get graph dict and store to disc. Used to parallelize preprocessing

            Args:
                row: Dataframe row corresponding to data point
                out_path: path to store the resulting dict

            Returns:
                None
            """

            graph_dict = load_graph_dict(row, self.dataset_name, self.config, self.interface_dir,
                                        node_type=self.node_type,
                                        interface_distance_cutoff=self.interface_distance_cutoff,
                                        interface_hull_size=self.interface_hull_size,
                                        max_edge_distance=self.max_edge_distance,
                                        affinity_type=self.affinity_type,
                                        load_embeddings=self.load_embeddings,
                                        save_path=out_path
                                        )
        submit_jobs(preload_graph_dict, graph_dicts2process, self.num_threads)

        if self.pretrained_model == "DeepRefine":
            logger.debug(
                f"Preprocessing {len(deeprefine_graphs2process)} DeepRefine graphs with {self.num_threads} threads")
            submit_jobs(self.preload_deeprefine_graph, deeprefine_graphs2process, self.num_threads)


    def preload_deeprefine_graph(self, idx: str, pdb_filepath: str, row: pd.Series):
        """ Function to get graph dict of Deeprefine graphs and store to disc

        Used by preprocess functionality

        Args:
            idx: Name of the DataPoint
            pdb_filepath: Path of pdb file
            row: DataFrame Row of the data point

        Returns:
            None
        """

        graph = load_deeprefine_graph(idx, pdb_filepath, self.interface_dir,
                                      self.interface_distance_cutoff, self.interface_hull_size)

        if self.interface_hull_size is None or self.interface_hull_size == "" or self.interface_hull_size == "None":
            graph_filepath = os.path.join(self.graph_dir, idx + ".pickle")
        else:
            graph_filepath = os.path.join(self.graph_dir, f"interface_hull_{self.interface_hull_size}",
                                          idx + ".pickle")

        with open(graph_filepath, 'wb') as f:
            pickle.dump(graph, f)

    def load_df(self, pdb_ids: List):
        """ Load all the dataset information (csv) into a pandas dataframe

        Discard all PDBs not given in argument

        Args:
            pdb_ids: List of PDB ids to use in this dataset

        Returns:
            pd.DataFrame: Object with all information available for the pdb_ids
        """
        summary_path, _ = get_data_paths(self.config, self.dataset_name)
        if self.publication is not None:
            summary_path = os.path.join(summary_path, self.publication + ".csv")
        summary_df = pd.read_csv(summary_path, index_col=0)
        if pdb_ids is None:
            pdb_ids = summary_df.index.tolist()
        else:
            summary_df = summary_df.loc[pdb_ids]

        summary_df = summary_df[~summary_df.index.duplicated(keep='first')]
        return summary_df.fillna(""), pdb_ids

    def get_edges(self, data_file: Dict) -> Tuple[Dict, Dict]:
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

        # use only nodes actually available in the graph
        data_file["adjacency_tensor"] = data_file["adjacency_tensor"][:, data_file["graph_nodes"], :][:, :,
                                        data_file["graph_nodes"]]

        all_edges, edge_attributes = get_hetero_edges(data_file, self.edge_types,
                                                      max_interface_distance=self.interface_distance_cutoff,
                                                      max_edge_distance=self.max_edge_distance)

        return all_edges, edge_attributes

    def get_node_features(self, data_file: Dict) -> np.ndarray:
        """ Extract residue features from data dict

        NOTE: This function shuffles the amino acid order if not all are taken

        Args:
            data_file: Dict containing all relevant information about the complex

        Returns:
            np.ndarray: Residue encodings
        """
        if self.max_nodes is not None:  # use only the n-closest nodes
            graph_nodes = data_file["closest_residues"][:self.max_nodes].astype(int)
            logging.warning("Warning: Node order is impacted")
        elif self.interface_hull_size is not None:  # use only nodes in interface hull
            adjacency_matrix = data_file["adjacency_tensor"]
            # below interface distance threshold and not in same protein -> interface edge
            interface_nodes = \
            np.where((adjacency_matrix[-1, :, :] < self.interface_distance_cutoff) & (adjacency_matrix[2, :, :] != 1))[
                0]
            interface_nodes = np.unique(interface_nodes)

            interface_hull = np.where(adjacency_matrix[3, interface_nodes, :] < self.interface_hull_size)
            graph_nodes = np.unique(interface_hull[1])
            logging.warning("Warning: Node order is impacted")
        else:  # use all nodes
            graph_nodes = sorted(data_file["closest_residues"].astype(int))

        data_file["graph_nodes"] = graph_nodes

        if self.pretrained_model == "Binding_DDG":  # different node features required for binding-ddg pretrained model
            node_features = []
            for idx in graph_nodes:
                residue_info = data_file["residue_infos"][idx]
                residue_coords = data_file["residue_atom_coordinates"][idx, :]
                node_coordinates = residue_coords.flatten()
                coordinate_mask = ~np.all(np.isinf(residue_coords), axis=-1)
                indices = np.array([residue_info["residue_pdb_index"],
                                    residue_info["chain_idx"], residue_info["on_chain_residue_idx"]])
                all_features = np.concatenate([node_coordinates, indices, coordinate_mask])
                node_features.append(all_features)

            node_features = np.array(node_features)
        else:
            node_features = data_file["node_features"][graph_nodes]

        if self.max_nodes is not None and len(graph_nodes) < self.max_nodes:  # add zero nodes for fixed size graphs
            diff = self.max_nodes - len(graph_nodes)
            node_features = np.vstack((node_features, np.zeros((diff, len(node_features[0])))))

        return node_features

    def get_graph_dict(self, df_idx: str) -> Dict:
        """ Get the graph dict for a data point

        Either load from disc or compute anew

        Args:
            df_idx: Index of dataset dataframe

        Returns:
            Dict: graph information for index
        """
        file_path = self.processed_graph_files.format(df_idx)
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
            if isinstance(row, pd.DataFrame):
                # multiple instances with same id (== same mutation code)
                # sample one randomly == data augmentation
                row = row.sample(1).squeeze()

            graph_dict = load_graph_dict(row, self.dataset_name, self.config, self.interface_dir,
                                         node_type=self.node_type,
                                         interface_distance_cutoff=self.interface_distance_cutoff,
                                         interface_hull_size=self.interface_hull_size,
                                         max_edge_distance=self.max_edge_distance,
                                         affinity_type=self.affinity_type,
                                         load_embeddings=self.load_embeddings,
                                         save_path=self.save_graphs and file_path
                                         )
        return graph_dict

    def load_graph(self, df_idx: str) -> Tuple[HeteroData, Dict]:
        """ Load a data point either from disc or compute it anew

        Args:
            df_idx: Dataset index of the datapoint

        Returns:
            HeteroData: PyG HeteroData object
            Dict: Metadata for graph
        """
        graph_dict = self.get_graph_dict(df_idx)
        node_features = self.get_node_features(graph_dict)
        edge_indices, edge_features = self.get_edges(graph_dict)

        affinity = graph_dict["affinity"]
        if self.scale_values and self.affinity_type == "-log(Kd)":
            affinity = scale_affinity(affinity, self.scale_min, self.scale_max)
        affinity = np.array(affinity)

        graph = HeteroData()

        graph["node"].x = torch.Tensor(node_features).float()
        graph["node"].chain_id = torch.tensor([ord(residue_info["chain_id"]) for residue_info in graph_dict["residue_infos"]])
        graph["node"].residue_id = torch.tensor([residue_info["residue_id"] for residue_info in graph_dict["residue_infos"]])  # this is the pdb residue_id

        try:
            graph["node"].positions = torch.stack([residue_info["matched_position"] for residue_info in graph_dict["residue_infos"]])
            graph["node"].orientations = torch.stack([residue_info["matched_orientation"] for residue_info in graph_dict["residue_infos"]])
            graph["node"].residue_index = torch.stack([residue_info["matched_residue_index"] for residue_info in graph_dict["residue_infos"]]) # this is the index of the residue in the LOADED protein
        except KeyError:
            pass  # data is only available when of-embeddings are used
        graph.y = torch.from_numpy(affinity).float()

        for edge_type, edges in edge_indices.items():
            graph[edge_type].edge_index = edges
            graph[edge_type].edge_attr = edge_features[edge_type].float()

        return graph, graph_dict

    def load_deeprefine_graph(self, df_idx: str) -> dgl.DGLGraph:
        """Load a data point for Deeprefine model either from disc or compute it anew

        Utilize DeepRefine functionality to get graphs

        Args:
            df_idx: Dataset index of the datapoint

        Returns:
            dgl.DGLGraph: DGL graph object
        """
        if self.interface_hull_size is None or self.interface_hull_size == "" or self.interface_hull_size == "None":
            graph_filepath = os.path.join(self.graph_dir, df_idx + ".pickle")
        else:
            if not os.path.exists(os.path.join(self.graph_dir, f"interface_hull_{self.interface_hull_size}")):
                Path(os.path.join(self.graph_dir, f"interface_hull_{self.interface_hull_size}")).mkdir(exist_ok=True,
                                                                                                       parents=True)
            graph_filepath = os.path.join(self.graph_dir, f"interface_hull_{self.interface_hull_size}",
                                          df_idx + ".pickle")

        compute_graph = True
        graph = None
        if (not self.force_recomputation or self.preprocess_data) and os.path.exists(graph_filepath):
            try:
                with open(graph_filepath, 'rb') as f:
                    graph = pickle.load(f)
                    compute_graph = False
            except:
                # recompute graph if saved graph is not parsable
                compute_graph = True

        if compute_graph:  # graph not loaded from disc
            row = self.data_df.loc[df_idx]
            pdb_filepath, _ = get_pdb_path_and_id(row, self.dataset_name, self.config)

            graph = load_deeprefine_graph(df_idx, pdb_filepath, self.interface_dir,
                                          self.interface_distance_cutoff, self.interface_hull_size)

            if self.save_graphs and not os.path.exists(graph_filepath):
                with open(graph_filepath, 'wb') as f:
                    pickle.dump(graph, f)

        return graph

    def load_data_point(self, df_idx: str) -> Dict:
        """ Functionality to load the correct graph + all necessary meta information

        Loads either a PyG graph or DGL graph based on the model used later on

        Args:
            df_idx: Dataset index of the datapoint

        Returns:
            Dict: Dict containing graph(s) and path of the PDB
        """

        if self.interface_hull_size is None or self.interface_hull_size == "":
            filepath = os.path.join(self.interface_dir, df_idx + ".pdb")  # TODO this is wrong. We don't use self.interface_dir at the moment
        else:
            filepath = os.path.join(self.interface_dir, f"interface_hull_{self.interface_hull_size}", df_idx + ".pdb")

        graph, graph_dict = self.load_graph(df_idx)
        data_point = {
            "filepath": filepath,
            "graph": graph,
        }

        if self.pretrained_model == "DeepRefine":
            deeprefine_graph = self.load_deeprefine_graph(df_idx)
            nodes = graph_dict["graph_nodes"]
            data_point["deeprefine_graph"] = node_subgraph(deeprefine_graph, nodes)

        return data_point

    def get_absolute(self, idx: int) -> Dict:
        """ Get the datapoint with one graph for a dataset index

        Args:
            idx: Index in the dataset

        Returns:
            Dict: Graph object and meta information to this data point
        """
        pdb_id = self.pdb_ids[idx]
        graph_data = self.load_data_point(pdb_id)

        data = {
            "relative": False,
            "affinity_type": self.affinity_type,
            "input": graph_data,
            "dataset_name": self.full_dataset_name
        }
        # pdb.set_trace()
        return data

    def get_relative(self, idx: int) -> Dict:
        """ Get the datapoints for a dataset index

        List of two related datapoints for siamese networks

        Args:
            idx: Index in the dataset

        Returns:
            Dict: Dict containing two graphs and meta information for this data point
        """
        data = {
            "relative": True,
            "affinity_type": self.affinity_type,
            "input": [],
            "dataset_name": self.full_dataset_name
        }

        pdb_id, other_pdb_id = self.relative_pairs[idx]
        data["input"].append(self.load_data_point(pdb_id))

        data["input"].append(self.load_data_point(other_pdb_id))

        return data

    @property
    def num_node_features(self) -> int:
        """Returns the number of features per node in the dataset."""
        data = self[0]["input"]
        data = data[0]["graph"] if isinstance(data, List) else data["graph"]
        if hasattr(data["node"], 'num_node_features'):
            return data["node"].num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_edge_features(self) -> int:
        """Returns the number of features per edge in the dataset."""
        data = self[0]["input"]
        data = data[0]["graph"] if isinstance(data, List) else data["graph"]
        if hasattr(data["node", "edge", "node"], 'num_edge_features'):
            return data["node", "edge", "node"].num_edge_features

    @staticmethod
    def collate(input_dicts: List[Dict]) -> Union[List, Dict]:
        """ Collate multiple datapoints to a batch
        1. Batch the graphs (using DGL Batch or PyG Batch)
        2. Batch the metadata information
        Args:
            input_dicts: List of input data points
        Returns:
            Dict: Containing the batches data points
        """

        relative_data = input_dicts[0]["relative"]
        assert all([relative_data == input_dict["relative"] for input_dict in input_dicts])

        affinity_type = input_dicts[0]["affinity_type"]
        assert all([affinity_type == input_dict["affinity_type"] for input_dict in input_dicts])

        dataset_name = input_dicts[0]["dataset_name"]
        assert all([dataset_name == input_dict["dataset_name"] for input_dict in input_dicts])

        data_batch = {
            "relative": relative_data,
            "affinity_type": affinity_type,
            "dataset_name": dataset_name
        }
        if relative_data:  # relative data
            input_graphs = []
            for i in range(2):
                batched_dict = {"graph": Batch.from_data_list([input_dict["input"][i]["graph"] for input_dict in input_dicts]),
                                "filepath": [input_dict["input"][i]["filepath"] for input_dict in input_dicts]}

                if "deeprefine_graph" in input_dicts[0]["input"][i]:
                    batched_dict["deeprefine_graph"] = dgl.batch(
                        [input_dict["input"][i]["deeprefine_graph"] for input_dict in input_dicts])

                input_graphs.append(batched_dict)

            data_batch["input"] = input_graphs

        else:
            data_batch["input"] = {"graph": Batch.from_data_list([input_dict["input"]["graph"] for input_dict in input_dicts]),
                            "filepath": [input_dict["input"]["filepath"] for input_dict in input_dicts]}

            if "deeprefine_graph" in input_dicts[0]["input"]:
                data_batch["input"]["deeprefine_graph"] = dgl.batch(
                    [input_dict["input"]["deeprefine_graph"] for input_dict in input_dicts])

        return data_batch
