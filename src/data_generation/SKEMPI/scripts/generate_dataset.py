import random

import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Union, Tuple, List
import os
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio import pairwise2
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from collections import  defaultdict
import warnings
warnings.filterwarnings("ignore")

if "snakemake" not in globals(): # use fake snakemake object for debugging
    project_root =  "../../../../" # three directories above
    folder_path = os.path.join(project_root, "resources", "SKEMPI_v2")

    publication = "phillips21_bindin"
    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(folder_path, "skempi_v2.csv")]
    snakemake.output = [os.path.join(project_root, "results", "SKEMPI_v2", "skempi_v2.csv")]

    snakemake.params = {}
    snakemake.params["pdb_folder"] = os.path.join(project_root, "results", "SKEMPI_v2", "mutated")
    snakemake.params["abag_affinity_df_path"] = "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/abag_affinity_dataset/abag_affinity_dataset.csv"
    snakemake.params["abag_affinity_pdb_path"] = "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/abag_affinity_dataset/pdbs"
    snakemake.params["redundancy_cutoff"] = 80





out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]
pdb_path = snakemake.params["pdb_folder"]

gas_constant = 8.31446261815324  # 0.0821 kcal


def convert_mutation_code(row):
    skempi_code = row["Mutation(s)_cleaned"]
    codes = skempi_code.split(",")

    new_codes = []
    for code in codes:
        wt_res = code[0]
        chain = code[1]
        index = code[2:-1]
        mut_res = code[-1]
        new_codes.append(f"{chain}{wt_res}{index}{mut_res}")

    return ";".join(new_codes)


def calc_delta_g(row, affinity_col):
    temperature = row["Temperature_cleaned"]
    affinity = row[affinity_col]
    delta_g = gas_constant * temperature * np.log(affinity)
    return delta_g / 4184  # convert to kcal


def clean_temp(value):
    value = value.replace("(assumed)", "")
    try:
        return int(value)
    except:
        return np.nan


def get_chain_info(row):
    _, chain1, chain2 = row["#Pdb"].split("_")

    info = {}
    for chain in chain1:
        info[chain.lower()] = 0
    for chain in chain2:
        info[chain.lower()] = 1
    return info


def get_index(row):
    if row["mutation_code"] == "":
        return row["pdb"]
    else:
        return row["pdb"] + "-" + row["Mutation(s)_cleaned"].lower()


def get_sequence(filepath: str):
    structure = PDBParser(QUIET=True).get_structure('tmp', filepath)
    chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}

    return chains


def is_redundant(filepath: str, val_pdbs: List, pdb_paths, redudancy_cutoff: float = 0.8):
    orig_chains = get_sequence(os.path.join(pdb_path, filepath))
    for pdb_id, path in zip(val_pdbs, pdb_paths):
        check_chains = get_sequence(os.path.join(snakemake.params["abag_affinity_pdb_path"], path))
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



skempi_df = pd.read_csv(file_path, sep=";")
skempi_df = skempi_df[skempi_df["iMutation_Location(s)"].isin(["COR", "RIM", "SUP"])].copy()

protein_skempi = skempi_df[skempi_df["Hold_out_type"].isin(["AB/AG", "Pr/PI", "AB/AG,Pr/PI"])].copy()
protein_skempi["pdb"] = protein_skempi["#Pdb"].apply(lambda x: x.split("_")[0].lower() )

protein_skempi["mutation_code"] = protein_skempi.apply(lambda row: convert_mutation_code(row), axis=1)

protein_skempi["Temperature_cleaned"] = protein_skempi["Temperature"].apply(lambda val: clean_temp(val))

protein_skempi["delta_g_wt"] = protein_skempi.apply(lambda row: calc_delta_g(row,"Affinity_wt_parsed"), axis=1)
protein_skempi["delta_g_mut"] = protein_skempi.apply(lambda row: calc_delta_g(row,"Affinity_mut_parsed"), axis=1)

protein_skempi["-log(Kd)_wt"] = protein_skempi["Affinity_wt_parsed"].apply(lambda kd: -np.log10(kd))
protein_skempi["-log(Kd)_mut"] = protein_skempi["Affinity_mut_parsed"].apply(lambda kd: -np.log10(kd))

protein_skempi["chain_infos"] = protein_skempi.apply(get_chain_info, axis=1)

protein_skempi["filename"] = protein_skempi.apply(lambda row: os.path.join(row["pdb"].upper(), row["Mutation(s)_cleaned"] + ".pdb") ,axis=1)

protein_skempi["index"] = protein_skempi.apply(get_index, axis=1)

clean_skempi = protein_skempi[["pdb", "filename", "-log(Kd)_mut", "chain_infos", "mutation_code", "index"]]
clean_skempi = clean_skempi.rename({"-log(Kd)_mut": "-log(Kd)"}, axis=1)
clean_skempi = clean_skempi[~clean_skempi["-log(Kd)"].isna()]


wildtypes = clean_skempi["pdb"].unique()
for pdb in wildtypes:
    wt_row = protein_skempi[protein_skempi["pdb"] == pdb].iloc[0]
    row = {
        "pdb": pdb,
        "mutation_code": "",
        "-log(Kd)": wt_row["-log(Kd)_wt"],
        "chain_infos": wt_row["chain_infos"],
        "filename": os.path.join(wt_row["pdb"].upper(), "original" + ".pdb"),
        "index": pdb + "-original"
    }
    clean_skempi = clean_skempi.append(row, ignore_index=True)

clean_skempi = clean_skempi.set_index("index")
clean_skempi.index.name = ""

clean_skempi = clean_skempi[~clean_skempi.index.duplicated(keep='first')]


# make train val splits according to abag_affinity split
clean_skempi["validation"] = ""
clean_skempi["test"] = False

abag_affinity_df_path = snakemake.params["abag_affinity_df_path"]
abag_affinity_df = pd.read_csv(abag_affinity_df_path)
num_splits = max(abag_affinity_df["validation"].astype(int).values)

for i in range(1, num_splits + 1):
    val_pdbs = abag_affinity_df[abag_affinity_df["validation"].isin([0,i])]["pdb"]
    pdb_paths = abag_affinity_df[abag_affinity_df["validation"].isin([0,i])]["filename"]
    valset_count = 0
    valset_ids = set()
    #clean_skempi = clean_skempi.sample(frac=1, random_state=123)
    pdbs = clean_skempi["pdb"].unique().tolist()
    random.shuffle(pdbs)
    for pdb in pdbs:
        filename = clean_skempi[clean_skempi["pdb"] == pdb]['filename'].tolist()[0]
        redundant, own_chain, pdb_id, chain, score = is_redundant(filename, val_pdbs, pdb_paths,
                                                                  redudancy_cutoff=snakemake.params["redundancy_cutoff"])

        if redundant or pdb in val_pdbs.tolist():  # add to valdiation set if there is a redudancy to abag_validation set
            valset_count += 1
            valset_ids.add(pdb)

    if valset_count / len(pdbs) < 0.1:
        diff = int(0.1 * len(clean_skempi["pdb"].unique()) - valset_count)
        possible_pbds = clean_skempi[(~clean_skempi["pdb"].isin(valset_ids))]["pdb"].unique()
        additional_pdbs = random.sample(possible_pbds, diff)
        valset_ids.update(additional_pdbs)
        valset_count = len(valset_ids)

    clean_skempi.loc[clean_skempi["pdb"].isin(valset_ids), "validation"] += str(i)

clean_skempi.to_csv(out_path)
