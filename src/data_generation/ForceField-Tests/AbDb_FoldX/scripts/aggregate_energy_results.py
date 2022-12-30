# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
import pandas as pd
from io import StringIO
import os


if "snakemake" not in globals(): # use fake snakemake object for debugging
    out_folder = "/msc/home/ftraxl96/projects/ag_binding_affinity/data/AbDb_REF15/FoldX_results"
    all_pdb_ids = ["6B0S_1"]

    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(out_folder, "Summary_{}_AC.fxout".format(pdb_id)) for pdb_id in all_pdb_ids ]
    snakemake.output = [os.path.join(out_folder, "../foldx_results.csv")]


def read_file(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "Output type: AnalyseComplex\n":
            break
    else:
        print("Error Loading File", filepath)
        return pd.DataFrame()

    csv = StringIO("\n".join(lines[i+1:]))

    return pd.read_csv(csv, "\t")

def get_row(pdb_id: str):
    pdb_info = {
        "pdb_id": pdb_id.split("_")[0].lower(),
        "file_name": pdb_id + ".pdb"
    }

    # read Interaction file
    file_path = os.path.join(folder, "Interaction_{}_AC.fxout".format(pdb_id))
    results = read_file(file_path)
    total_interface_residues = 0
    for idx, row in results.iterrows():
        if row["Group1"] in ["H", "L"] and row["Group2"] in ["H", "L"]:
            continue
        total_interface_residues += row["Interface Residues"]

    pdb_info["Interface Residues"] = total_interface_residues

    # read summary file
    file_path = os.path.join(folder, "Summary_{}_AC.fxout".format(pdb_id))
    results = read_file(file_path)
    total_interaction_energy = 0
    for idx, row in results.iterrows():
        if row["Group1"] in ["H", "L"] and row["Group2"] in ["H", "L"]:
            continue
        total_interaction_energy += row["Interaction Energy"]
    pdb_info["Interaction Energy"] = total_interaction_energy

    return pdb_info

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

all_pdb_ids = [ "_".join(file_path.split("/")[-1].split("_")[1:3]) for file_path in snakemake.input ]

folder = "/".join(snakemake.input[0].split("/")[:-1])

summary_df = pd.DataFrame()

for idx, pdb_id in enumerate(all_pdb_ids):
    row = get_row(pdb_id)
    summary_df = summary_df.append(row, ignore_index=True)

summary_df.to_csv(out_path, index=False)