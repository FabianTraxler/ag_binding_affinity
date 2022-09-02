# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb, get_score_line_string, get_all_comments
import pandas as pd
from collections import defaultdict
import os


if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os
    from abag_affinity.utils.config import read_yaml, get_data_paths
    config = read_yaml("../../../abag_affinity/config.yaml")
    data_path = config["DATA"]["path"]
    skempi_folder_path = os.path.join(data_path, config["DATA"]["SKEMPI.v2"]["folder_path"])

    sample_pdb_id = "3BN9"
    mutation_code = "RB51A"
    snakemake = type('', (), {})()
    snakemake.input = []
    snakemake.input.extend([os.path.join(skempi_folder_path, "wildtype", sample_pdb_id + ".pdb")])
    snakemake.input.extend([os.path.join(skempi_folder_path, "mutated_wildtype", sample_pdb_id, mutation_code + ".pdb")])
    snakemake.input.extend([os.path.join(skempi_folder_path, "mutated_relaxed", sample_pdb_id, mutation_code + ".pdb")])
    snakemake.input.extend([os.path.join(skempi_folder_path, "relaxed_wildtype", sample_pdb_id + ".pdb")])
    snakemake.input.extend([os.path.join(skempi_folder_path, "relaxed_mutated", sample_pdb_id, mutation_code + ".pdb")])
    snakemake.input.extend([os.path.join(skempi_folder_path, "relaxed_mutated_relaxed", sample_pdb_id, mutation_code + ".pdb")])

    snakemake.output = ["../skempi_generation_summary.csv"]


pyrosetta.init(extra_options="-mute all")


mutations_to_be_scored = [
    ("wildtype", "mutated_wildtype"),
    ("wildtype", "mutated_relaxed"),
    #("relaxed_wildtype", "relaxed_mutated"),
    #("relaxed_wildtype", "relaxed_mutated_relaxed")
]


all_folders = set([ folder for folders in mutations_to_be_scored for folder in folders])


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


def get_folder_pdb2input_files(list_of_input_files):
    folder_pdb2input_files = defaultdict(list)
    for input_file in list_of_input_files:
        file_path_components = input_file.split("/")

        for folder in all_folders:
            if folder == file_path_components[-2]:
                if folder not in folder_pdb2input_files:
                    folder_pdb2input_files[folder] = dict()
                pdb = input_file.split("/")[-1].split(".")[0]
                folder_pdb2input_files[folder][pdb] = input_file
                break
            elif folder == file_path_components[-3]:
                if folder not in folder_pdb2input_files:
                    folder_pdb2input_files[folder] = defaultdict(list)
                pdb = input_file.split("/")[-2]
                folder_pdb2input_files[folder][pdb].append(input_file)
                break

    return folder_pdb2input_files


def get_binding_affinity(bound_filepath:str, unbound_filepath: str):
    bound_score = get_energy_score(bound_filepath)
    unbound_score = get_energy_score(unbound_filepath)

    if bound_score is not None and unbound_score is not None:
        return bound_score - unbound_score
    else:
        return None


out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

mutation_codes = snakemake.params["mutation_codes"]
pdb_out_folder = snakemake.params["out_folder"]

summary_df = pd.DataFrame()

all_rows = {}


for (wildtype_type, mutation_type) in mutations_to_be_scored:
    wildtype_path = os.path.join(pdb_out_folder, wildtype_type + ".pdb")

    wildtype_score = get_energy_score(wildtype_path)

    for mutation_code in mutation_codes:
        if mutation_code not in all_rows:
            all_rows[mutation_code] = {"mutation_code": mutation_code}
        mutated_file = os.path.join(pdb_out_folder, mutation_type, mutation_code + ".pdb")
        mutation_score = get_energy_score(mutated_file)

        if wildtype_score is not None and mutation_score is not None:
            delta_delta_g = wildtype_score - mutation_score
        else:
            delta_delta_g = None

        all_rows[mutation_code][mutation_type] = delta_delta_g


for row in all_rows.values():
    summary_df = summary_df.append(row, ignore_index=True)

summary_df.to_csv(out_path, index=False)
