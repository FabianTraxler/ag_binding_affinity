import glob

import pandas as pd
import os
from typing import Dict, Tuple, Set, List
import numpy as np
from biopandas.pdb import PandasPdb
from tqdm import tqdm
import shutil

np.random.seed(111)
tqdm.pandas()


if "snakemake" not in globals(): # use fake snakemake object for debugging
    from pathlib import Path
    from os.path import join
    project_root = Path(__file__).parents[4]  # two directories above - absolute paths not working
    resource_folder = join(project_root, "resources", "abag_affinity_dataset")
    results_folder = join(project_root, "results", "abag_affinity_dataset")

    snakemake = type('', (), {})()
    snakemake.input = [join(resource_folder, "sabdab_summary.tsv")]
    snakemake.output = [join(results_folder, "abag_affinity_dataset.csv")]

    snakemake.params = {
        "pdb_folder": join(resource_folder, "pdbs"),
        "abdb_pdb_folder": join(project_root, "resources", "AbDb", "pdbs"),
        "redundancy_file": join(resource_folder, "abdb_redundancy_info.txt"),
        "benchmark_dataset_file": join(project_root, "results", "antibody_benchmark", "benchmark.csv"),
        "n_val_splits": 3
    }


def get_abdb_dataframe(pdb_folder: str) -> pd.DataFrame:
    abdb_df = pd.DataFrame()
    pdb_files = os.listdir(pdb_folder)
    abdb_df["filename"] = pdb_files

    abdb_df["pdb"] = abdb_df["filename"].apply(lambda pdb_file: pdb_file.split("_")[0].lower())

    return abdb_df.set_index("pdb")


def get_redundancy_mapping(redundancy_file_path: str) -> Tuple[Dict, Set]:
    with open(redundancy_file_path) as f:
        lines = f.readlines()
    redundant_ids = {}
    all_ids = set()  # all (redundant) PDB ids in AbDb
    for line in lines:
        pdb_ids = line.split(",")
        pdb_ids = [pdb_id.strip().lower().split("_")[0] for pdb_id in pdb_ids]
        pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id.strip() != ""]
        all_ids.update(pdb_ids)
        for i, pdb_id in enumerate(pdb_ids):
            redundant_ids[pdb_id] = pdb_ids

    return redundant_ids, all_ids


def remove_benchmark_test_data(dataset: pd.DataFrame, benchmark_dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(benchmark_dataset_path):
        print("Benchmark dataset not found - May lead to redundant data")
        return dataset

    benchmark_df = pd.read_csv(benchmark_dataset_path)
    benchmark_pdb_ids = benchmark_df["pdb"].values

    dataset = dataset[~dataset.index.isin(benchmark_pdb_ids)].copy()
    return dataset


def add_train_val_split(dataset: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    dataset["validation"] = 0
    dataset.reset_index(inplace=True, drop=True)
    total_num_train_data = len(dataset)
    indices = np.arange(total_num_train_data)

    set_size = int((total_num_train_data / n_splits))

    np.random.shuffle(indices)
    val_indices = np.split(indices, [set_size * i for i in range(1, n_splits)])

    for i, val_idx in enumerate(val_indices):
        i += 1
        dataset.loc[val_idx, "validation"] = i

    return dataset


def get_chain_ids(row):
    path = os.path.join(snakemake.params["abdb_pdb_folder"], row["filename"])

    cleaned_pdb = PandasPdb().read_pdb(path)
    input_atom_df = cleaned_pdb.df['ATOM']

    chain_ids = input_atom_df["chain_id"].unique().tolist()
    return chain_ids


def get_chain_info(chains: List) -> Dict:
    chain_infos = {}

    ab_chains = ["h", "l"]
    chains = [chain.lower() for chain in chains]

    for chain in ab_chains:
        if chain in chains:
            chain_infos[chain.lower()] = 0

    chains = set(chains) - set(ab_chains)
    for chain in chains:
        chain_infos[chain.lower()] = 1

    return chain_infos


def copy_files(dataset: pd.DataFrame, pdb_folder: str, new_folder: str):
    os.makedirs(new_folder)
    for idx, row in dataset.iterrows():
        filename = row["filename"]
        pdb_id = row["pdb"]
        shutil.copyfile(os.path.join(pdb_folder, filename), os.path.join(new_folder, pdb_id + ".pdb"))


# get datasets
sabdb_df = pd.read_csv(snakemake.input[0], sep="\t")
sabdb_df = sabdb_df.set_index("pdb")
abdb_df = get_abdb_dataframe(snakemake.params["abdb_pdb_folder"])

# join datasets
sabdab_pdb_ids = set(sabdb_df.index.unique())
abdb_pdb_ids = set(abdb_df.index.unique())
overlapping_ids = abdb_pdb_ids.intersection(sabdab_pdb_ids)
abag_affintiy_df = sabdb_df[sabdb_df.index.isin(overlapping_ids)].copy()
abag_affintiy_df = abag_affintiy_df.join(abdb_df)
abag_affintiy_df["pdb"] = abag_affintiy_df.index

# remove benchmark pdb ids
abag_affintiy_df = remove_benchmark_test_data(abag_affintiy_df, snakemake.params["benchmark_dataset_file"])

# add chain information
abag_affintiy_df["chains"] = abag_affintiy_df.progress_apply(lambda row: get_chain_ids(row), axis=1)
abag_affintiy_df["chain_infos"] = abag_affintiy_df["chains"].apply(get_chain_info)

# add -log(Kd)
abag_affintiy_df["-log(Kd)"] = abag_affintiy_df.apply(lambda row: -np.log10(row["affinity"]), axis=1)

# remove pdbs that lead to errors
problematic_pdbs = ["5e8e", "5tkj", "3eo1", "2oqj"]
abag_affintiy_df = abag_affintiy_df[~abag_affintiy_df["pdb"].isin(problematic_pdbs)]

abag_affintiy_df = abag_affintiy_df.drop_duplicates(subset='pdb', keep='first')

# add validation splits
abag_affintiy_df = add_train_val_split(abag_affintiy_df, snakemake.params["n_val_splits"])

abag_affintiy_df = abag_affintiy_df[["pdb", "filename", "chain_infos", "-log(Kd)", "delta_g", "validation"]]
# add index
abag_affintiy_df.index = abag_affintiy_df["pdb"]
abag_affintiy_df.index.name = ""


copy_files(abag_affintiy_df, snakemake.params["abdb_pdb_folder"], snakemake.params["pdb_folder"])

abag_affintiy_df["filename"] = abag_affintiy_df["pdb"].apply(lambda x: str(x) + ".pdb")

# save dataset
abag_affintiy_df.to_csv(snakemake.output[0])