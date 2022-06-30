# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb, get_score_line_string, get_all_comments
import pandas as pd


if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os
    from abag_affinity.utils.config import read_yaml, get_data_paths
    config = read_yaml("../../../abag_affinity/config.yaml")
    _, pdb_path = get_data_paths(config, "AbDb")
    abdb_folder_path = os.path.join(config["DATA"]["path"], config["DATA"]["AbDb"]["folder_path"])

    sample_pdb_id = "1A2Y_1.pdb"
    snakemake = type('', (), {})()
    snakemake.input = [[os.path.join(abdb_folder_path, pdb_type, sample_pdb_id)] for pdb_type in ["bound_wildtype", "unbound_wildtype", "unbound_relaxed", "bound_relaxed", "relaxed_unbound", "relaxed_unbound_relaxed" ]]
    snakemake.output = ["../summary_df.csv"]




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

def get_folder2input_files(list_of_input_file_lists):
    folder2input_files = {}
    for folder in all_folders:
        for input_file_list in list_of_input_file_lists:
            if len(input_file_list) > 0:
                if "/" + folder + "/" in input_file_list[0]:
                    folder2input_files[folder] = input_file_list
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

pdb_ids = [ file.split(".")[0].split("/")[-1] for file in snakemake.input[0] ]

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
            binding_affinity =  pdb_info[comparison[0]] - pdb_info[comparison[1]]
        else:
            binding_affinity = None

        pdb_info[comparison[0] + " vs " + comparison[1]] = binding_affinity
    summary_df = summary_df.append(pdb_info, ignore_index=True)

summary_df.to_csv(out_path)