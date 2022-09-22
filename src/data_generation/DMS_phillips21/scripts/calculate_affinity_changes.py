# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyrosetta
from pyrosetta.rosetta.core.pose import (Pose, add_comment, dump_comment_pdb,
                                         get_all_comments,
                                         get_score_line_string)

if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os

    from abag_affinity.utils.config import get_data_paths, read_config
    config = read_config("../../../abag_affinity/config.yaml")
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


out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

mutation_codes = snakemake.params["mutation_codes"]
complexes = snakemake.params["complexes"]
pdb_out_folder = snakemake.params["out_folder"]

summary_df = pd.DataFrame()

all_rows = []


for (mutation, complex) in zip(mutation_codes, complexes):
    mut_file = pdb_out_folder + f"/mutated_wildtype/{complex}_{mutation}.pdb"
    relax_mut = pdb_out_folder + f"/mutated_relaxed/{complex}_{mutation}.pdb"

    mut_score = get_energy_score(mut_file)
    relax_mut_score = get_energy_score(relax_mut)
    row= {
        "complex": complex,
        "antibody_id": complex.split("_")[0],
        "pdb_file": complex.split("_")[1],
        "mutation_code": mutation,
        "mutation_score": mut_score,
        "relax_mut_score": relax_mut_score
    }
    all_rows.append(row)


for row in all_rows:
    summary_df = summary_df.append(row, ignore_index=True)

summary_df.to_csv(out_path, index=False)
