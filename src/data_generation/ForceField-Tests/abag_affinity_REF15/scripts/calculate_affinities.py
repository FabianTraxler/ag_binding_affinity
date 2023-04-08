# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose

if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os

    from abag_affinity.utils.config import get_data_paths, read_config
    config = read_config("../../../abag_affinity/config.yaml")
    _, pdb_path = get_data_paths(config, "AbDb_REF15")
    abdb_folder_path = os.path.join(config["DATA"]["path"], config["DATA"]["AbDb_REF15"]["folder_path"])

    all_types = ["bound_wildtype", "unbound_wildtype", "unbound_relaxed", "bound_relaxed", "relaxed_unbound", "relaxed_unbound_relaxed"]

    type_files = []
    for file_type in all_types:
        files = os.listdir(os.path.join(abdb_folder_path, file_type))
        type_files.append(set(files))
    files_available = set.intersection(*type_files)

    all_pdb_ids = list(files_available)

    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(abdb_folder_path, pdb_type, pdb_id) for pdb_id in all_pdb_ids for pdb_type in all_types ]
    snakemake.output = ["../abdb_summary.csv"]


pyrosetta.init(extra_options="-mute all")

scorefxn = pyrosetta.get_fa_scorefxn()
packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)

comparisons = [
    ("bound_wildtype", "unbound_wildtype"),
    ("bound_wildtype", "unbound_relaxed"),
    ("bound_relaxed", "relaxed_unbound"),
    ("bound_relaxed", "relaxed_unbound_relaxed")
]

all_folders = set([ folder for folders in comparisons for folder in folders])


def get_energy_score(pdb_path: str):

    with open(pdb_path) as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[-i][:20] == "rosetta_energy_score":
            break
    else:
        return None
    score = lines[-i].split(" ")[-1].strip()
    try:
        return float(score)
    except:
        return None


def get_alt_energy_score(pdb_path: str):

    with open(pdb_path) as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[-i][:26] == "#BEGIN_POSE_ENERGIES_TABLE":
            break
    else:
        return None

    i -= 3
    score = 0
    for part_score in lines[-i].split(" ")[1:-1]:
        score += float(part_score.strip())
    try:
        return float(score)
    except:
        return None


def get_folder2input_files(list_of_input_files):
    folder2input_files = defaultdict(list)
    for input_file in list_of_input_files:
        path_components = input_file.split("/")
        for folder in all_folders:
            if path_components[-2] == folder:
                folder2input_files[folder].append(input_file)
                break
    return folder2input_files


def get_binding_affinity(bound_filepath:str, unbound_filepath: str):
    bound_score = get_energy_score(bound_filepath)
    unbound_score = get_energy_score(unbound_filepath)

    if bound_score is not None and unbound_score is not None:
        return bound_score - unbound_score
    else:
        return None


out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

folder2input_files = get_folder2input_files(snakemake.input)

pdb_ids = [ file.split(".")[0].split("/")[-1] for file in folder2input_files["bound_wildtype"] ]

summary_df = pd.DataFrame()

for idx, pdb_id in enumerate(pdb_ids):
    pdb_info = {
        "pdb_id": pdb_id.split("_")[0].lower(),
        "file_name": pdb_id + ".pdb"
    }
    for comparison in comparisons:
        bound_file = folder2input_files[comparison[0]][idx]
        unbound_file = folder2input_files[comparison[1]][idx]
        if comparison[0] not in pdb_info:
            bound_score = get_energy_score(bound_file)
            pdb_info[comparison[0]] = bound_score
        if comparison[1] not in pdb_info:
            unbound_score = get_energy_score(unbound_file)
            pdb_info[comparison[1]] = unbound_score

        if pdb_info[comparison[0]] is not None and pdb_info[comparison[1]] is not None:
            binding_affinity = pdb_info[comparison[0]] - pdb_info[comparison[1]]
        else:
            binding_affinity = None

        pdb_info[comparison[0] + " vs " + comparison[1]] = binding_affinity
    summary_df = summary_df.append(pdb_info, ignore_index=True)

summary_df.to_csv(out_path, index=False)
