from functools import lru_cache
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.PDB.PDBParser import PDBParser
import pandas as pd
import os
from typing import Dict, Tuple, Set, List, Union, Optional
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
        "benchmark_pdb_path": join(project_root, "results", "antibody_benchmark", "pdbs"),
        "n_val_splits": 4,
        "test_size": 50,
        "redundancy_cutoff": 0.8
    }


def pdb_chain_mapping(pdb_file: Union[str, Path]) -> pd.DataFrame:
    """
    Return the chain mapping as provided by AbDb
    """
    mapping = []

    with open(pdb_file) as f:
        for l in f:
            if l.startswith("REMARK 950 CHAIN "):
                mapping.append(l.replace("REMARK 950 CHAIN ", "").split())
            elif len(mapping) > 0:
                break
        else:
            raise ValueError("pdb_file did not contain chain mapping")
    df = pd.DataFrame(data=mapping, columns=("type", "abdb_label", "original_label"))
    if "1ZV5" in str(pdb_file):  # fix error in dataset
        df.loc[df["abdb_label"] == "L", "abdb_label"] = "l"
    return df

def get_abdb_dataframe(pdb_folder: str) -> pd.DataFrame:
    abdb_df = pd.DataFrame()
    pdb_files = os.listdir(pdb_folder)
    abdb_df["filename"] = pdb_files

    abdb_df["pdb"] = abdb_df["filename"].apply(lambda pdb_file: pdb_file.split(".")[0].lower())

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

    # check for length 4 and lowercase
    assert np.all([len(v) == 4 and v.islower() for v in benchmark_pdb_ids]), "Benchmark PDB ids are not in the correct format"

    dataset = dataset[~dataset.index.map(lambda v: v[:4]).isin(benchmark_pdb_ids)].copy()
    return dataset


@lru_cache(maxsize=1000)
def get_sequence(filepath: str):
    structure = PDBParser(QUIET=True).get_structure('tmp', filepath)
    chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
    return chains

def antigen_chains_too_long(filepath: str, max_length: Optional[int] = None):
    """
    filter out complexes where the antigen chain is too long
    Params:
        filepath: path to the pdb file
        max_length: maximum length of the antigen chains (sum).
    Returns:
        True if the antigen chain is too long
    """
    if max_length is None:
        return False
    antigen_seqs = [seq for key, seq in get_sequence(filepath).items() if key in "ABCDE"]
    return sum(len(seq) for seq in antigen_seqs) > max_length


def is_redundant(filepath: str, val_pdbs: List, pdb_paths, redundancy_cutoff: float):
    """
    filter out complexes where the three chains are below the redundancy cutoff
    """
    orig_chains = get_sequence(filepath)
    for pdb_id, path in zip(val_pdbs, pdb_paths):
        check_chains = get_sequence(os.path.join(snakemake.params["benchmark_pdb_path"], path))
        scores = []
        for chain in "HLA":  # issue: there might be a mismatch between antigen chain identifier such that redundancy remains undetected here :/
            if chain not in orig_chains or chain not in check_chains:
                if chain == "L":
                    # print(f"Chain {chain} not found in {filepath} or {path}")  # spam
                    continue
                else:
                    raise ValueError(f"Chain {chain} not found in {filepath} or {path}")

            seq1 = Seq(orig_chains[chain])
            seq2 = Seq(check_chains[chain])
            alignment_score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
            scores.append(alignment_score / min(len(seq1), len(seq2)))

        if all(score > redundancy_cutoff for score in scores):
        # if np.mean(scores) > redundancy_cutoff:  # TODO this should be used with `redundancy_cutoff=0.9` to stick to the other redundancy analysis
            return True

    return False


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
    path = os.path.join(snakemake.input["cleaned_abdb_folder"], row["filename"])

    cleaned_pdb = PandasPdb().read_pdb(path)
    input_atom_df = cleaned_pdb.df['ATOM']

    return ''.join(input_atom_df["chain_id"].unique())


def copy_files(dataset: pd.DataFrame, pdb_folder: str, new_folder: str):
    Path(new_folder).mkdir(exist_ok=True, parents=True)
    for idx, row in dataset.iterrows():
        filename = row["filename"]
        pdb_id = row["pdb"]
        shutil.copyfile(os.path.join(pdb_folder, filename), os.path.join(new_folder, pdb_id + ".pdb"))

# get datasets
sabdab_df = pd.read_csv(snakemake.input["sabdab_data"], sep="\t")
sabdab_df = sabdab_df.set_index("pdb")
abdb_df = get_abdb_dataframe(snakemake.input["cleaned_abdb_folder"])

# get information about original chain IDs (to merge with sabdab)
chain_mappings = []

for f in Path(snakemake.input["uncleaned_abdb_folders"]).glob("*/*.pdb"):
    df = pdb_chain_mapping(f)
    df["filename"] = f.name
    chain_mappings.append(df)

chain_mappings = pd.concat(chain_mappings)

# Convert to sabdab_df compatible format
def first_or_none(x):
    if len(x) == 0:
        return None
    return x[0]
chain_mappings = chain_mappings.groupby(["filename"]).apply(lambda group: [{"pdb": group.name[:4].lower(),
                                                                            "filename": group.name,
                                                                            "pdb_num": group.name.split('.')[0].lower(),
                                                                            "Hchain": first_or_none(group[group["type"] == "H"]["original_label"].values),
                                                                            "Lchain": first_or_none(group[group["type"] == "L"]["original_label"].values),
                                                                            "antigen_chain": antigen_chain}
                                                                           for antigen_chain in group[group["type"] == "A"]["original_label"].values])

chain_mappings = pd.DataFrame.from_records(chain_mappings.explode()).fillna(np.nan)

# join datasets
# TODO this is non-ideal because sabdab sometimes has only one entry for multiple redundant PDBs. AbDb just lists one of the redundant PDBs, so the whole thing fails
# would be better to start with SabDab and then add AbDb information (e.g. pull the PDBs from there as they are neatly processed)
abag_affinity_df = sabdab_df.merge(chain_mappings, on=["pdb", "Hchain", "Lchain", "antigen_chain"], how="inner")
abag_affinity_df["pdb"] = abag_affinity_df["pdb_num"]  # the new standard from here on!
abag_affinity_df.set_index("pdb", inplace=True, drop=False)

assert abag_affinity_df["pdb"].is_unique
# abag_affintiy_df = abag_affintiy_df.drop_duplicates(subset='pdb', keep='first')

# remove benchmark pdb ids
abag_affinity_df = remove_benchmark_test_data(abag_affinity_df, snakemake.params["benchmark_dataset_file"])
benchmark_df = pd.read_csv(snakemake.params["benchmark_dataset_file"])
val_pdbs = benchmark_df["pdb"]
pdb_paths = benchmark_df["filename"]

# Remove pdbs that lead to errors or are redundant

problematic_pdbs = ["1zmy_1",  # 1zmy is wrongly annotated by AbDb
                    "1kxv_1",  # too large.
                    "2nz9_1",  # too big.  In chain A, one could cut away everything that is smaller than residue 875. also filtered in SKEMPI
                    "2nyy_1",  # same as for 2nz9
                    "5i5k_1"   # Also huge. In chain A, only residues 820-930 would be required
                    ] + "1jrh_1 1bj1_1 2jel_1 2bdn_1 2b2x_1 4u6h_1 4jpk_1 3a6c_1 3a6b_1 3a67_1 1nma_1 1nby_1 1cz8_1 2b2x_1".split(" ")  # overlapping with SKEMPI
for pdb in tqdm(abag_affinity_df["pdb"].tolist()):
    if pdb in problematic_pdbs:
        continue
    filename = abag_affinity_df[abag_affinity_df["pdb"] == pdb]['filename'].tolist()[0]
    filepath = os.path.join(snakemake.input["cleaned_abdb_folder"], filename)
    redundant = is_redundant(filepath, val_pdbs, pdb_paths, redudancy_cutoff=snakemake.params["redundancy_cutoff"])
    too_long = antigen_chains_too_long(filepath, snakemake.params["max_antigen_length"])
    if redundant or too_long:
        problematic_pdbs.append(pdb)

print(f"Removed {len(problematic_pdbs)} PDBs from the dataset: {problematic_pdbs}")
abag_affinity_df = abag_affinity_df[~abag_affinity_df["pdb"].isin(problematic_pdbs)]

# add chain information
abag_affinity_df["chains"] = abag_affinity_df.progress_apply(lambda row: get_chain_ids(row), axis=1)


# add -log(Kd)
abag_affinity_df["-log(Kd)"] = abag_affinity_df.apply(lambda row: -np.log10(row["affinity"]), axis=1)

# add validation splits
abag_affinity_df = add_train_val_test_split(abag_affinity_df, n_splits=snakemake.params["n_val_splits"],
                                            test_size=snakemake.params["test_size"])

abag_affinity_df = abag_affinity_df[["pdb", "filename", "chains", "-log(Kd)", "delta_g", "validation", "test"]]
# add index
abag_affinity_df.index = abag_affinity_df["pdb"]
abag_affinity_df.index.name = ""

# copy files to new folder (doesn't make a lot of sense but whatever)
copy_files(abag_affinity_df, snakemake.input["cleaned_abdb_folder"], snakemake.params["pdb_folder"])
abag_affinity_df["filename"] = abag_affinity_df["pdb"].apply(lambda x: str(x) + ".pdb")

# save dataset
abag_affinity_df.to_csv(snakemake.output[0])
