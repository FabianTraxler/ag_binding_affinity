from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.PDB.PDBParser import PDBParser
import pandas as pd
import os
from typing import Dict, Tuple, Set, List
import numpy as np
from biopandas.pdb import PandasPdb
from tqdm import tqdm
import shutil
from pathlib import Path
from os.path import join

np.random.seed(1234)
tqdm.pandas()

# TODO this is broken now
if "snakemake" not in globals(): # use fake snakemake object for debugging

    project_root = Path(__file__).parents[4]  # two directories above - absolute paths not working
    resource_folder = join(project_root, "resources")
    results_folder = join(project_root, "results", "abag_affinity_dataset")

    snakemake = type('', (), {})()
    snakemake.input = [join(resource_folder, "SAbDab", "sabdab_summary.tsv")]
    snakemake.output = [join(results_folder, "abag_affinity_dataset.csv")]

    snakemake.params = {
        "pdb_folder": join(resource_folder, "AbDb", "pdbs"),
        "abdb_pdb_folder": join(project_root, "resources", "AbDb", "pdbs"),
        "redundancy_file": join(resource_folder, "AbDb", "abdb_redundancy_info.txt"),
        "benchmark_dataset_file": join(project_root, "results", "antibody_benchmark", "benchmark.csv"),
        "benchmark_pdb_path": join(project_root, "results", "antibody_benchmark", "pdbs_bound"),
        "n_val_splits": 4,
        "test_size": 50,
        "redundancy_cutoff": 0.8
    }


sequences = {}

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


def get_sequence(filepath: str):
    if filepath in sequences:
        return sequences[filepath]
    structure = PDBParser(QUIET=True).get_structure('tmp', filepath)
    chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
    sequences[filepath] = chains
    return chains


def is_redundant(filepath: str, val_pdbs: List, pdb_paths, redudancy_cutoff: float = 0.8):
    orig_chains = get_sequence(os.path.join(snakemake.input[1], filepath))
    for pdb_id, path in zip(val_pdbs, pdb_paths):
        check_chains = get_sequence(os.path.join(snakemake.params["benchmark_pdb_path"], path))
        for orig_chain, orig_seq in orig_chains.items():
            seq1 = Seq(orig_seq)
            for check_chain, check_seq in check_chains.items():
                seq2 = Seq(check_seq)
                alignments = pairwise2.align.globalxx(seq1, seq2)
                for alignment in alignments:
                    score = alignment.score / (alignment.end - alignment.start)
                    if score > redudancy_cutoff:
                        return True, orig_chain, pdb_id, check_chain, score

    return False, None, None, None, None


def add_train_val_test_split(dataset: pd.DataFrame, n_splits: int, test_size: int = 50) -> pd.DataFrame:
    dataset["validation"] = 0
    dataset["test"] = False
    dataset.reset_index(inplace=True, drop=True)
    total_num_train_data = len(dataset) - test_size
    indices = np.arange(len(dataset))

    set_size = int((total_num_train_data / n_splits))

    np.random.shuffle(indices)
    val_indices = np.split(indices, [set_size * i for i in range(1, n_splits)] + [len(dataset) - test_size])

    for i, val_idx in enumerate(val_indices):
        i += 1
        if i == len(val_indices):
            i = 0
            dataset.loc[val_idx, "test"] = True

        dataset.loc[val_idx, "validation"] = i


    return dataset


def get_chain_ids(row):
    path = os.path.join(snakemake.input[1], row["filename"])

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
    Path(new_folder).mkdir(exist_ok=True, parents=True)
    for idx, row in dataset.iterrows():
        filename = row["filename"]
        pdb_id = row["pdb"]
        shutil.copyfile(os.path.join(pdb_folder, filename), os.path.join(new_folder, pdb_id + ".pdb"))


# get datasets
sabdb_df = pd.read_csv(snakemake.input[0], sep="\t")
sabdb_df = sabdb_df.set_index("pdb")
abdb_df = get_abdb_dataframe(snakemake.input[1])

# join datasets
sabdab_pdb_ids = set(sabdb_df.index.unique())
abdb_pdb_ids = set(abdb_df.index.unique())
overlapping_ids = abdb_pdb_ids.intersection(sabdab_pdb_ids)
abag_affintiy_df = sabdb_df[sabdb_df.index.isin(overlapping_ids)].copy()
abag_affintiy_df = abag_affintiy_df.join(abdb_df)
abag_affintiy_df["pdb"] = abag_affintiy_df.index

# remove pdbs that lead to errors
problematic_pdbs = ["5e8e", "5tkj", "3eo1", "2oqj"]
abag_affintiy_df = abag_affintiy_df[~abag_affintiy_df["pdb"].isin(problematic_pdbs)]

abag_affintiy_df = abag_affintiy_df.drop_duplicates(subset='pdb', keep='first')

# remove benchmark pdb ids
abag_affintiy_df = remove_benchmark_test_data(abag_affintiy_df, snakemake.params["benchmark_dataset_file"])
benchmark_df = pd.read_csv(snakemake.params["benchmark_dataset_file"])
val_pdbs = benchmark_df["pdb"]
pdb_paths = benchmark_df["filename"]
redundant_pdbs = []
for pdb in tqdm(abag_affintiy_df["pdb"].tolist()):
    filename = abag_affintiy_df[abag_affintiy_df["pdb"] == pdb]['filename'].tolist()[0]
    redundant, own_chain, pdb_id, chain, score = is_redundant(filename, val_pdbs, pdb_paths,
                                                              redudancy_cutoff=snakemake.params["redundancy_cutoff"])
    if redundant:
        redundant_pdbs.append(pdb)
abag_affintiy_df = abag_affintiy_df[~abag_affintiy_df["pdb"].isin(redundant_pdbs)]

# add chain information
abag_affintiy_df["chains"] = abag_affintiy_df.progress_apply(lambda row: get_chain_ids(row), axis=1)
abag_affintiy_df["chain_infos"] = abag_affintiy_df["chains"].apply(get_chain_info)

# add -log(Kd)
abag_affintiy_df["-log(Kd)"] = abag_affintiy_df.apply(lambda row: -np.log10(row["affinity"]), axis=1)

# add validation splits
abag_affintiy_df = add_train_val_test_split(abag_affintiy_df, n_splits=snakemake.params["n_val_splits"],
                                            test_size=snakemake.params["test_size"])

abag_affintiy_df = abag_affintiy_df[["pdb", "filename", "chain_infos", "-log(Kd)", "delta_g", "validation", "test"]]
# add index
abag_affintiy_df.index = abag_affintiy_df["pdb"]
abag_affintiy_df.index.name = ""


copy_files(abag_affintiy_df, snakemake.input[1], snakemake.params["pdb_folder"])

abag_affintiy_df["filename"] = abag_affintiy_df["pdb"].apply(lambda x: str(x) + ".pdb")

# save dataset
abag_affintiy_df.to_csv(snakemake.output[0])
