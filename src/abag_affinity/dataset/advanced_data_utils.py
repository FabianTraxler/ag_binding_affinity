import glob
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import spatial as sp
from scipy.stats import gmean
from torch.utils.data import DataLoader as DL_torch, Subset, ConcatDataset
from torch_geometric.data import DataLoader

from modules.ag_binding_affinity.src.abag_affinity.dataset import AffinityDataset
from modules.ag_binding_affinity.src.abag_affinity.dataset.utils import logger
from modules.ag_binding_affinity.src.abag_affinity.utils.config import get_data_paths


def complexes_from_dms_datasets(dataset_names: List, args) -> List:
    """
    Expand DMS dataset names to include the complexes
    """
    def complexes_from_dms_dataset(dataset_name, metadata):
        """
        Return complex names corresponding to dataset_name

        Args:
            dataset_name: Name of the dataset

        """

        if dataset_name.startswith("DMS-"):
            dataset_name = dataset_name.split("-")[1]
            if dataset_name.startswith("mason21"):
                complexes = metadata["mason21_optim_therap_antib_by_predic"]["complexes"]
            else:
                complexes = metadata[dataset_name]["complexes"]

            return [":".join([dataset_name, complex["antibody"]["name"], complex["antigen"]["name"]])
                    for complex in complexes]
        else:
            return [dataset_name]

    with open(os.path.join(args.config["DATASETS"]["path"], args.config["DATASETS"]["DMS"]["metadata"]), "r") as f:
        metadata = yaml.safe_load(f)

    unique_sets = np.unique([ds_name.split("#")[0] for ds_name in dataset_names]).tolist()
    with_complex_datasets = np.concatenate([complexes_from_dms_dataset(dataset_name, metadata) for dataset_name in unique_sets])
    return with_complex_datasets.tolist()


def get_dataloader(args: Namespace, train_dataset: AffinityDataset, val_datasets: List[AffinityDataset]) -> Tuple[
    DataLoader, DataLoader]:
    """ Get dataloader for train and validation dataset

    Use the DGL Dataloader for the DeepRefine Inputs

    Args:
        args: CLI arguments
        train_dataset: Torch dataset for train examples
        val_dataset: Torch dataset for validation examples

    Returns:
        Tuple: Train dataloader, validation dataloader
    """

    if args.batch_size == "max":
        batch_size = 0
    else:
        batch_size = args.batch_size

    if train_dataset.relative_data:
        train_dataset.update_valid_pairs()

    train_dataloader = DL_torch(train_dataset, num_workers=args.num_workers, batch_size=batch_size,
                                shuffle=args.shuffle, collate_fn=AffinityDataset.collate)

    val_dataloaders = [DL_torch(val_dataset, num_workers=args.num_workers, batch_size=batch_size,
                              collate_fn=AffinityDataset.collate) for val_dataset in val_datasets]

    return train_dataloader, val_dataloaders


def train_val_split(config: Dict, dataset_name: str, validation_set: Optional[int] = None, dms_validation_size: Optional[float] = 0.2) -> Tuple[List, List]:
    """ Split data in a train and a validation subset

    For the abag_affinity datasets, we use the predefined split given in the csv, otherwise use random split

    Args:
        config: Dict with configuration info
        dataset_name: Name of the dataset
        validation_set: Integer identifier of the validation split (1,2,3). Only required for abag_affinity datasets
        dms_validation_size: Size of the validation set (proportion (0.0-1.0)). Only applied to DMS datasets

    Returns:
        Tuple: List with indices for train and validation set
    """
    dms_train_size = 1 - dms_validation_size

    if "-" in dataset_name:
        # DMS data
        dataset_name, publication_code = dataset_name.split("-")
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_path = os.path.join(summary_path, publication_code + ".csv")
        summary_df = pd.read_csv(summary_path, index_col=0)

        if config["DATASETS"][dataset_name]["affinity_types"][publication_code] == "E":  # should be unnecessary
            summary_df = summary_df[(~summary_df["E"].isna())]
        else:
            summary_df = summary_df[~summary_df["-log(Kd)"].isna()]
        # filter datapoints with missing PDB files
        data_path = Path(config["DATASETS"]["path"]) / config["DATASETS"][dataset_name]["folder_path"] / config["DATASETS"][dataset_name]["mutated_pdb_path"]
        summary_df = summary_df[summary_df.filename.apply(lambda fn: (data_path / fn).exists())]

        affinity_type = config["DATASETS"][dataset_name]["affinity_types"][publication_code]
        # TODO we should refactor/DRY finding pairs here with the function update_pairs/get_compatible_pairs in the AffinityDataset class
        if affinity_type == "E":
            summary_df = summary_df[~summary_df.index.duplicated(keep='first')]

            # Normalize between 0 and 1 on a per-complex basis. This way value ranges of E and NLL fit, when computing possible pairs
            e_values = summary_df.groupby(summary_df.index.map(lambda i: i.split("-")[0]))["E"].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

            e_values = e_values.values.reshape(-1,1).astype(np.float32)

            nll_values = summary_df["NLL"].values

            # Scale the NLLs to (0-1). The max NLL value in DMS_curated.csv is 4, so 0-1-scaling should be fine
            if nll_values.shape[0] == 0:
                nll_values = np.full_like(nll_values, 0.5)
            elif np.max(nll_values) > np.min(nll_values):  # test that all values are not the same
                nll_values = (nll_values - np.min(nll_values)) / (np.max(nll_values) - np.min(nll_values))
                assert (nll_values < 0.7).sum() > (nll_values > 0.7).sum(), "Many NLL values are 'large'"
            else:
                nll_values = np.full_like(nll_values, 0.5)

            # TODO refactor this block (just always split, as it includes the else-case)
            if len(e_values) > 50000:

                n_splits = int(len(e_values) / 50000) + 1

                has_valid_partner = set()
                e_splits = np.array_split(e_values, n_splits)

                nll_splits = np.array_split(nll_values, n_splits)
                total_elements = 0
                for i in range(len(e_splits)):
                    for j in range(i, len(e_splits)):
                        split_e_dists = sp.distance.cdist(e_splits[i], e_splits[j])
                        split_nll_avg = (nll_splits[i][:, None] + nll_splits[j]) / 2

                        # We need to consider the std of the labels to always find relevant pairs
                        min_dist = split_nll_avg * 2 * e_splits.std()
                        valid_pairs = (split_e_dists - min_dist) >= 0
                        has_valid_partner_id = np.where(np.sum(valid_pairs, axis=1) > 0)[0] + total_elements
                        has_valid_partner.update(has_valid_partner_id)
                    total_elements += len(e_splits[i])
                has_valid_partner = np.fromiter(has_valid_partner, int, len(has_valid_partner))
                valid_partners = None
            else:
                e_dists = sp.distance.cdist(e_values, e_values)
                nll_avg = (nll_values[:, None] + nll_values) / 2
                min_dist = nll_avg * 2 * e_values.std()
                valid_pairs = (e_dists - min_dist) >= 0
                has_valid_partner = np.where(np.sum(valid_pairs, axis=1) > 0)[0]
                valid_partners = {
                    summary_df.index[idx]: set(summary_df.index[np.where(valid_pairs[idx])[0]].values.tolist()) for idx
                    in has_valid_partner}

            data_points_with_valid_partner = set(summary_df.index[has_valid_partner])
            total_valid_data_points = len(data_points_with_valid_partner)
            logger.debug(f"There are in total {len(data_points_with_valid_partner)} data points with valid partner")

            train_ids = set()
            val_ids = set()
            while len(data_points_with_valid_partner) > 0:
                pdb_idx = data_points_with_valid_partner.pop()
                if valid_partners is not None:
                    possible_ids = set(valid_partners[pdb_idx])
                    if len(possible_ids - train_ids) > 0:
                        # choose one of the unused ids
                        other_idx = random.choice(list(possible_ids - train_ids))
                    else:
                        # chosse one randomly with the risk that this idx is already in the train data
                        other_idx = random.choice(list(possible_ids))
                    data_points_with_valid_partner.discard(other_idx)
                else:
                    other_idx = pdb_idx
                if len(train_ids) >= total_valid_data_points * dms_train_size:  # add to val ids
                    val_ids.add(pdb_idx)
                    val_ids.add(other_idx)
                else:
                    train_ids.add(pdb_idx)
                    train_ids.add(other_idx)

        elif affinity_type == "-log(Kd)":
            pdb_idx = summary_df.index.values.tolist()
            random.shuffle(pdb_idx)
            train_ids = pdb_idx[:int(dms_train_size*len(pdb_idx))]
            val_ids = pdb_idx[int(dms_train_size*len(pdb_idx)):]
        else:
            raise ValueError(
                f"Wrong affinity type given - expected one of (-log(Kd), E) but got {affinity_type}")

        train_ids = list(train_ids)
        val_ids = list(val_ids)

        #train_ids = ["phillips21_bindin:cr9114:h1wiscon05-SD166N;ID188S;AD194T;ND195A;KD210I".lower()] + train_ids[                                                                                               :5]
        #val_ids = val_ids[:2]

    else:
        summary_path, _ = get_data_paths(config, dataset_name)
        summary_df = pd.read_csv(summary_path, index_col=0)

        summary_df["validation"] = summary_df["validation"].fillna("")

        val_pdbs = summary_df.loc[summary_df["validation"] == validation_set, "pdb"]
        train_pdbs = summary_df.loc[(summary_df["validation"] != validation_set) & (~summary_df["test"]), "pdb"]

        if "mutated_pdb_path" in config["DATASETS"][dataset_name]:  # only use files that were generated
            data_path = os.path.join(config["DATASETS"]["path"],
                                     config["DATASETS"][dataset_name]["folder_path"],
                                     config["DATASETS"][dataset_name]["mutated_pdb_path"])
            # TODO use `filename` in summary_df (see above! DRY?)
            all_files = glob.glob(data_path + "/*/*") + glob.glob(data_path + "/*")
            available_files = set(
                file_path.split("/")[-2].lower() + "-" + file_path.split("/")[-1].split(".")[0].lower() for file_path in
                all_files if file_path.split(".")[-1] == "pdb")
            summary_df = summary_df[summary_df.index.isin(available_files)]

        summary_df = summary_df[summary_df["-log(Kd)"].notnull()]

        val_ids = summary_df[summary_df["pdb"].isin(val_pdbs)].index.tolist()
        train_ids = summary_df[summary_df["pdb"].isin(train_pdbs)].index.tolist()

    return train_ids, val_ids


def load_datasets(config: Dict, dataset: str, validation_set: int,
                  args: Namespace, validation_size: Optional[float] = 0.1,
                  only_neglogkd_samples=False) -> Tuple[
    AffinityDataset, List[AffinityDataset]]:
    """ Get train and validation datasets for a specific dataset and data type

    1. Get train and validation splits
    2. Load the dataset in the specified data type class

    Args:
        config: training configuration as dict
        dataset: Name of the dataset:Usage of data (absolute, relative) - eg. SKEMPI.v2:relative
        validation_set: Integer identifier of the validation split (1,2,3)
        args: CLI arguments
        validation_size: Size of the validation set (proportion (0.0-1.0))
        only_neglogkd_samples: If True, only use only samples that have -log(Kd) labels

    Returns:
        Tuple: Train and validation dataset
    """
    dataset_name, loss_types = dataset.split("#")

    # We missuse the data_type to allow setting different losses
    # The losses used for this dataset come after a # seperated by a comma, when multiple losses are used
    # Optionally, the losses can contain some weight using -
    # E.g. data_type = relative#l2-1,l1-0.1,relative_l1-2,relative_2-0.1,relative_ce-1

    if "relative" in loss_types and not only_neglogkd_samples:
        relative_data = True
    else:
        relative_data = False
    train_ids, val_ids = train_val_split(config, dataset_name, validation_set, validation_size)

    if args.test:
        train_ids = train_ids[:50]
        val_ids = val_ids[:20]

    logger.debug(f"Get dataLoader for {dataset_name}#{loss_types}")
    train_data = AffinityDataset(config, args.relaxed_pdbs, dataset_name, loss_types,
                                 train_ids,
                                 node_type=args.node_type,
                                 max_nodes=args.max_num_nodes,
                                 interface_distance_cutoff=args.interface_distance_cutoff,
                                 interface_hull_size=args.interface_hull_size,
                                 max_edge_distance=args.max_edge_distance,
                                 pretrained_model=args.pretrained_model,
                                 scale_values=args.scale_values,
                                 scale_min=args.scale_min,
                                 scale_max=args.scale_max,
                                 relative_data=relative_data,
                                 save_graphs=args.save_graphs,
                                 force_recomputation=args.force_recomputation,
                                 preprocess_data=args.preprocess_graph,
                                 preprocessed_to_scratch=args.preprocessed_to_scratch,
                                 num_threads=args.num_workers,
                                 load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path),
                                 only_neglogkd_samples=only_neglogkd_samples,
                                 )

    val_datas = [AffinityDataset(config, relaxed, dataset_name, loss_types,  # TODO @marco val should be done with the same loss as training, right?
                                 val_ids,
                                 node_type=args.node_type,
                                 max_nodes=args.max_num_nodes,
                                 interface_distance_cutoff=args.interface_distance_cutoff,
                                 interface_hull_size=args.interface_hull_size,
                                 max_edge_distance=args.max_edge_distance,
                                 pretrained_model=args.pretrained_model,
                                 scale_values=args.scale_values,
                                 scale_min=args.scale_min,
                                 scale_max=args.scale_max,
                                 relative_data=relative_data,
                                 save_graphs=args.save_graphs,
                                 force_recomputation=args.force_recomputation,
                                 preprocess_data=args.preprocess_graph,
                                 preprocessed_to_scratch=args.preprocessed_to_scratch,
                                 num_threads=args.num_workers,
                                 load_embeddings=None if not args.embeddings_type else (args.embeddings_type, args.embeddings_path),
                                 only_neglogkd_samples=only_neglogkd_samples,
                                 )
                 for relaxed in [bool(args.relaxed_pdbs)]
                 ]  # TODO disabling , not args.relaxed_pdbs for now. Enable once we generated all relaxed data

    return train_data, val_datas


def get_bucket_dataloader(args: Namespace, train_datasets: List[AffinityDataset], val_datasets: List[AffinityDataset]):
    """ Get dataloader for bucket learning using a specific batch sampler

    Args:
        args: CLI arguments
        train_datasets: List of datasets with training examples
        val_datasets: List of datasets with validation examples
        train_bucket_size: Size of the training buckets
        config: config object from weights&bias

    Returns:
        Tuple: train dataloader, list of validation dataloaders
    """
    train_buckets = []
    #Data Indices per Loss Function:
    data_indices = {}

    if args.bucket_size_mode == "min":
        train_bucket_size = [min([len(dataset) for dataset in train_datasets])] * len(train_datasets)
    elif args.bucket_size_mode == "geometric_mean":
        data_sizes = [len(dataset) for dataset in train_datasets]
        train_bucket_size =  [int(gmean(data_sizes))] * len(train_datasets)
    elif args.bucket_size_mode == "double_geometric_mean":
        data_sizes = [len(dataset) for dataset in train_datasets]
        geometric_mean = gmean(data_sizes)
        train_bucket_size = [ int(gmean([geometric_mean, size])) for size in data_sizes]
    else:
        raise ValueError(f"bucket_size_mode argument not supported: Got {args.bucket_size_mode} "
                         f"but expected one of [min, geometric_mean]")
    i = 0
    # TODO We might want to modify this code to work on a per-dataset or even per-complex level
    for idx, train_dataset in enumerate(train_datasets):
        if len(train_dataset) >= train_bucket_size[idx]:
            if train_dataset.full_dataset_name == args.target_dataset.split("#")[0]:  # args.target_dataset includes loss function
                # always take all data points from the target dataset
                indices = range(len(train_dataset))
            else:
                # sample without replacement
                indices = random.sample(range(len(train_dataset)), train_bucket_size[idx])
        else:
            # sample with replacement
            indices = random.choices(range(len(train_dataset)), k=train_bucket_size[idx])

        train_buckets.append(Subset(train_dataset, indices))
        # We split the Indices by loss criterion as 1 Batch cannot contain different loss functions!
        if train_dataset.loss_criterion in data_indices.keys():
            data_indices[train_dataset.loss_criterion].extend(list(range(i, i + len(indices))))
        else:
            data_indices[train_dataset.loss_criterion] = list(range(i, i + len(indices)))

        i += len(indices)

    # shorten relative data for validation because DMS datapoints tend to have a lot of validation data if we use 10% split
    # TODO: find better way
    for dataset in val_datasets:
        if dataset.relative_data:
            dataset.relative_pairs = dataset.relative_pairs[:100]

    train_dataset = ConcatDataset(train_buckets)
    batch_sampler = generate_bucket_batch_sampler(data_indices.values(), args.batch_size,
                                                  shuffle=args.shuffle)

    train_dataloader = DL_torch(train_dataset, num_workers=args.num_workers,
                                collate_fn=AffinityDataset.collate, batch_sampler=batch_sampler)

    # We expect a list of datasets for validation anyway, there is no reason to combine them!
    # Also, why there could be a reason to overreprent a dataset during training, it does not make much sense during evaluation?
    # TODO maybe we want to add something like geometric mean also here for equal?
    val_dataloaders = []
    already_added_datasets = []
    for val_dataset in val_datasets:
        if f"{val_dataset.full_dataset_name}#{val_dataset.loss_criterion}" not in already_added_datasets:
            val_dataloaders.append(DL_torch(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                    collate_fn=AffinityDataset.collate))
            already_added_datasets.append(f"{val_dataset.full_dataset_name}#{val_dataset.loss_criterion}")



    return train_dataloader, val_dataloaders


def generate_bucket_batch_sampler(data_indices_list: List, batch_size: int, shuffle: bool = False) -> List[List[int]]:
    """ Generate batches for different data types only combining examples of the same type

    Args:
        data_indices_list: List of data indices list
        batch_size: Size of the batches
        shuffle: Boolean indicator if examples and batches are to be shuffled

    Returns:
        List: Data indices in batches
    """

    all_batches = []

    for data_indices in data_indices_list:  # different batches for different data types
        if len(data_indices) == 0: continue  # ignore empty indices
        batches = []

        if shuffle:
            random.shuffle(data_indices)

        i = 0
        while i < len(data_indices):  # batch samples of same type
            batches.append(data_indices[i:i + batch_size])
            i += batch_size

        all_batches.extend(batches)

    if shuffle:
        random.shuffle(all_batches)

    return all_batches
