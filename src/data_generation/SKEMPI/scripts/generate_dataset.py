from collections import defaultdict, deque
import random
import pandas as pd
from pathlib import Path
import numpy as np
import string
from typing import List
import os
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.PDB.PDBParser import PDBParser
import warnings
warnings.filterwarnings("ignore")

# if "snakemake" not in globals(): # use fake snakemake object for debugging
#     project_root =  "../../../../" # three directories above
#     folder_path = os.path.join(project_root, "resources", "SKEMPI_v2")

#     publication = "phillips21_bindin"
#     snakemake = type('', (), {})()
#     snakemake.input = [os.path.join(folder_path, "skempi_v2.csv")]
#     snakemake.output = [os.path.join(project_root, "results", "SKEMPI_v2", "skempi_v2.csv")]

#     snakemake.params = {}
#     snakemake.params["pdb_folder"] = os.path.join(project_root, "results", "SKEMPI_v2", "mutated")
#     snakemake.params["abag_affinity_df_path"] = "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/abag_affinity_dataset/abag_affinity_dataset.csv"
#     snakemake.params["abag_affinity_pdb_path"] = "/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/abag_affinity_dataset/pdbs"
#     snakemake.params["redundancy_cutoff"] = 80

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
    """"
    Very similar to  get_chains in clean_metadata.py (antibody_benchmark)
    """
    pdb, chain1, chain2 = row["#Pdb"].split("_")

    info = {}

    # normally, chain1 is the antibody and chain2 is the antigen. However, this is not always the case (wtf?) :/
    if np.any([v in row["Protein 2"].lower() for v in ["igg", "fab", "fv", "antibody", "nanobody"]]) and not np.any([v in row["Protein 1"].lower() for v in ["igg", "fab", "fv", "antibody", "nanobody"]]):
        print(f"Swapping chains for PDB {pdb} with names {row['Protein 1']} and {row['Protein 2']}")
        chain1, chain2 = chain2, chain1

    if set(chain2) == {"L", "H"}:
        print(f"Swapping chains for PDB {pdb} because chain2 is LH")
        chain1, chain2 = chain2, chain1

    for chain, new in zip(chain1, "HL"):  # NOTE: H is usually first, but not always
        info[chain] = new
    if "L" in info and "H" in info:
        del info["L"]
        del info["H"]
    for chain, new in zip(chain2, string.ascii_uppercase):
        info[chain] = new
    if pdb in ["4GXU_ABCDEF_MN", "4NM8_ABCDEF_HL"]:  # CDEF are large and unnecessary. 4GXU is also (trimmed) in antibody_benchmark
        info["C"] = None
        info["D"] = None
        info["E"] = None
        info["F"] = None

    def order_substitutions(substitutions):
        """
        Order substiutions to avoid chain overlaps (and thereby loss of chain information)
        """
        # Create a dependency graph with nodes as keys and values
        graph = defaultdict(list)
        for src, dest in substitutions.items():
            graph[src].append(dest)

        # Perform a topological sorting on the graph
        sorted_nodes = []
        visited = set()
        stack = deque()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    visit(neighbor)
                stack.appendleft(node)

        for node in list(graph.keys()):
            visit(node)

        # Apply substitutions in the sorted order
        result = {}
        for node in reversed(stack):
            if node in substitutions:
                result[node] = substitutions[node]

        return result

    return order_substitutions(info)


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
        if snakemake.params.check_redundancy:
            redundant, own_chain, pdb_id, chain, score = is_redundant(filename, val_pdbs, pdb_paths,
                                                                    redudancy_cutoff=snakemake.params["redundancy_cutoff"])
        else:
            redundant = False

        if redundant or pdb in val_pdbs.tolist():  # add to valdiation set if there is a redudancy to abag_validation set
            valset_count += 1
            valset_ids.add(pdb)

    if valset_count / len(pdbs) < 0.1:
        diff = int(0.1 * len(clean_skempi["pdb"].unique()) - valset_count)
        possible_pdbs = clean_skempi[(~clean_skempi["pdb"].isin(valset_ids))]["pdb"].unique()
        additional_pdbs = random.sample(list(possible_pdbs), diff)
        valset_ids.update(additional_pdbs)
        valset_count = len(valset_ids)

    clean_skempi.loc[clean_skempi["pdb"].isin(valset_ids), "validation"] += str(i)

if snakemake.params.only_abs:
    selected_pdbs = skempi_df.loc[skempi_df["Hold_out_type"] == "AB/AG", "#Pdb"].drop_duplicates()
    selected_pdbs = selected_pdbs.apply(lambda v: v.split("_")[0].lower()).drop_duplicates()
# These PDBs have more than 11000 lines and their antigens have >= 350 residues, so we filter them for simplicity
blacklist_pdbs = "2NZ9 2NYY".lower().split(" ")  # these two have massive antigens. Also filtered in abag_affinity
# previous blacklist: 4KRP 3W2D 2VIR 2VIS 4KRO 1NCA 3NGB 3SE8 3SE9 5C6T 1N8Z 3N85 1YY9 4NM8 4GXU
clean_skempi = clean_skempi[clean_skempi["pdb"].isin(selected_pdbs.values)& ~(clean_skempi["pdb"].isin(blacklist_pdbs))]

clean_skempi.to_csv(out_path)
