import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict

if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os
    project_root = (Path(__file__).parents[4]).resolve()

    DATASET_NAME = "antibody_benchmark"

    resource_folder = os.path.join(project_root, "resources")
    results_folder = os.path.join(project_root, "results")

    dataset_resource_folder = os.path.join(resource_folder, DATASET_NAME)
    dataset_results_folder = os.path.join(results_folder, DATASET_NAME)

    in_file = os.path.join(dataset_resource_folder, "antibody_benchmark_cases.xlsx")
    out_file = os.path.join(dataset_results_folder, "benchmark.csv")

    snakemake = type('', (), {})()
    snakemake.input = [in_file]
    snakemake.output = [out_file]
    snakemake.params = {}
    snakemake.params["dataset_resource_folder"] = dataset_resource_folder

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]

summary_df = pd.read_excel(file_path)
print(file_path)
summary_df = summary_df.replace(" ", np.nan)
summary_df = summary_df[summary_df["Kd (nM)"].notnull()]

summary_df["pdb"] = summary_df["Complex PDB"].apply(lambda x: x.split("_")[0].lower())

summary_df["filename"] = summary_df["pdb"].apply(lambda x: x.upper() + ".pdb")

summary_df["-log(Kd)"] = summary_df["Kd (nM)"].apply(lambda x: -np.log10(x * 1e-9))

summary_df["delta_g"] = summary_df["ΔG (kcal/mol)"]


def get_chains(complex_name: str) -> Dict:
    ab_chains, ag_chains = complex_name.split("_")[-1].split(":")
    chain_info = {}
    for ab_chain in ab_chains.strip():
        chain_info[ab_chain.lower()] = 0
    for ag_chain in ag_chains.strip():
        chain_info[ag_chain.lower()] = 1

    # manually add chains since there are errors in the metadata file
    if complex_name.split("_")[0].lower() == "5kov":
        chain_info["l"] = 0

    return chain_info


summary_df["chain_infos"] = summary_df["Complex PDB"].apply(get_chains)
summary_df["validation"] = 0
summary_df["test"] = True

summary_df = summary_df[["pdb", "filename", "-log(Kd)", "Kd (nM)", "delta_g", "chain_infos", "validation", "test"]]

summary_df.index = summary_df["pdb"]
summary_df.index.name = ""

summary_df.to_csv(out_path)
