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

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]
pdb_path = snakemake.input[1]

parser = PDBParser(PERMISSIVE=3)
tqdm.pandas()


def read_metadata(filepath: str) -> pd.DataFrame:
    # load info file and convert to dataframe
    with open(filepath) as f:
        lines = f.readlines()

    all_records = []
    for line in lines[6:]:
        line = [token for token in line.split(" ") if token != ""]
        if "IC50" in line[3]:
            affinity_type = line[3][:4]
            affinity = line[3][5:]
        else:
            affinity_type = line[3][:2]
            affinity = line[3][3:]
        affinity_value = affinity[:-2]
        affinity_unit = affinity[-2:]

        all_records.append({
            "pdb": line[0],
            "resolution": line[1],
            "release_year": line[2],
            "affinty": float(affinity_value),
            "affinity_unit": affinity_unit,
            "affinity_type": affinity_type,
            "ligand_name": " ".join(line[6:])
        })

    summary_df = pd.DataFrame.from_records(all_records)
    return summary_df


def read_file(structure_id: str, path: Union[str, Path]) -> Tuple[Structure, Dict]:
    """ Read a PDB file and return the structure and header

    Args:
        structure_id: PDB ID
        path: Path of the PDB file

    Returns:
        Tuple: Structure (Bio.PDB object), header (Dict)
    """
    structure = parser.get_structure(structure_id, str(path))
    header = parser.get_header()

    return structure, header


def read_pdb(filename: str):
    path = os.path.join(pdb_path, filename)
    pdb_id = filename.split(".")[0]
    structure, header = read_file(pdb_id, path)
    chains = structure.get_chains()
    chain_ids = [chain.id for chain in chains]

    chain_residue_count = {}

    for model in structure:
        for chain in model:
            chain_residues = 0
            for r in chain.get_residues():
                if r.id[0] == ' ' and r.resname not in ["UNK", "HOH", "H_GOL", "W"]:
                    chain_residues += 1
            chain_residue_count[chain.id.lower()] = chain_residues

    compound_info = header.get("compound")
    if compound_info is None:
        return "No Compound Info"

    chain_info = {}
    molecules = []
    mol_id2chain = defaultdict(list)
    for info in compound_info.values():
        if isinstance(info, str):
            print(info)
            return "Invalid Information format"
        if info.get("molecule") is not None:
            molecule = info.get("molecule")
            if molecule == 'uncharacterized protein':
                return "Uncharacterized protein in complex"
            if "light" in molecule:
                molecule = molecule[:molecule.find("light")]
            if "heavy" in molecule:
                molecule = molecule[:molecule.find("heavy")]
            if molecule in molecules:
                mol_id = molecules.index(molecule)
            else:
                mol_id = len(molecules)
                molecules.append(molecule)
            chain_info[info["chain"]] = mol_id
            mol_id2chain[mol_id].append(info["chain"])
        else:
            return "No Molecule Info"

    for chain in structure.get_chains():
        if chain.id.lower() not in chain_info:
            return "No Info for chain {}".format(chain.id)

    if len(molecules) > 2:
        return "Too many molecules"

    for mol_id, chains in mol_id2chain.items():
        for chain in chains:
            if chain not in chain_residue_count or chain_residue_count[chain] == 0:
                return "Not all chains have residues"
    # print(chain_residue_count)
    return chain_info


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


summary_df = read_metadata(file_path)

# convert to mol
convert_unit = {
    'mM': 1e03,
    'uM': 1e06,
    "nM": 1e09,
    'pM': 1e12,
    'fM': 1e15
}
summary_df["Kd"] = summary_df.apply(lambda row: row["affinty"] / convert_unit[row["affinity_unit"]], axis=1)
summary_df["-log(Kd)"] = summary_df.apply(lambda row: -np.log10(row["Kd"]), axis=1)

gas_constant =  8.31446261815324 # 0.0821
def calc_delta_g(row):
    delta_g = -1 * gas_constant * row["temperature_kelvin"] * np.log(1/row["Kd"])
    return delta_g / 4184 # convert to kcal

summary_df["temperature_kelvin"] = 298.15 # assume temperature of 25° Celcius
summary_df["delta_g"] = summary_df.apply(lambda row: calc_delta_g(row), axis=1)


summary_df["filename"] = summary_df["pdb"].apply(lambda pdb_id: pdb_id + ".pdb")

chain_infos = summary_df["filename"].progress_apply(lambda name: read_pdb(name))
summary_df["chain_infos"] = chain_infos

mask = summary_df["chain_infos"].apply(lambda x: isinstance(x, dict))
cleaned_summary = summary_df[mask]
mask = cleaned_summary["chain_infos"].apply(lambda x: list(x.values()).count(list(x.values())[0]) != len(x.values()))
cleaned_summary = cleaned_summary[mask]

cleaned_summary = cleaned_summary[["pdb",  "filename", "-log(Kd)", "delta_g", "chain_infos"]]
cleaned_summary = cleaned_summary.set_index("pdb", drop=False)
cleaned_summary.index.name = ""


# make train val splits according to abag_affinity split
cleaned_summary["validation"] = ""
cleaned_summary["test"] = False

abag_affinity_df_path = snakemake.params["abag_affinity_df_path"]
abag_affinity_df = pd.read_csv(abag_affinity_df_path)
num_splits = max(abag_affinity_df["validation"].astype(int).values)
for i in range(1, num_splits + 1):
    val_pdbs = abag_affinity_df[abag_affinity_df["validation"].isin([0,i])]["pdb"]
    pdb_paths = abag_affinity_df[abag_affinity_df["validation"].isin([0,i])]["filename"]
    valset_count = 0
    valset_ids = set()
    cleaned_summary = cleaned_summary.sample(frac=1, random_state=123) # shuffle
    for idx, row in cleaned_summary.iterrows():
        redundant, own_chain, pdb_id, chain, score = is_redundant(row['filename'], val_pdbs, pdb_paths,
                                                                  redudancy_cutoff=snakemake.params["redundancy_cutoff"])
        if redundant:  # add to valdiation set if there is a redudancy to abag_validation set
            valset_count += 1
            valset_ids.add(row['pdb'])

    if valset_count / len(cleaned_summary) < 0.1:
        diff = int(0.1 * len(cleaned_summary) - valset_count)
        additional_rows = cleaned_summary[(~cleaned_summary["pdb"].isin(valset_ids))].sample(diff)
        valset_ids.update(additional_rows["pdb"].values)
        valset_count = len(valset_ids)

    cleaned_summary.loc[cleaned_summary["pdb"].isin(valset_ids), "validation"] += str(i)

cleaned_summary.to_csv(out_path)