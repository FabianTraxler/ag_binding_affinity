# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import json
from pathlib import Path
import yaml
import os
import pandas as pd
from typing import Dict


if "snakemake" not in globals(): # use fake snakemake object for debugging
    project_root = (Path(__file__).parents[4]).resolve()
    out_folder = os.path.join(project_root, "results/DMS/")

    publication = "phillips21_bindin"
    complexes = [("phillips21_bindin","cr9114", "h1newcal99"),
                 ("phillips21_bindin","cr6261", "h1newcal99"),
                 ("phillips21_bindin","cr9114", "h5ohio05"),
                 ("phillips21_bindin","cr9114", "h1wiscon05"),
                 ("phillips21_bindin","cr6261", "h9hk99")] #("starr21_prosp_covid","lycov016", "cov2rbd")

    #publication = "mason21_optim_therap_antib_by_predic_dms_H"
    #complexes = [("mason21_optim_therap_antib_by_predic_dms_H","trastuzumab", "her2")] #("starr21_prosp_covid","lycov016", "cov2rbd")

    file = out_folder + "mutated/{}/{}_{}.logs"

    metadata_file = os.path.join(project_root, "data/metadata_dms_studies.yaml")


    snakemake = type('', (), {})()
    snakemake.input = [file.format(complex, antibody, antigen) for (complex, antibody, antigen) in complexes ]
    snakemake.output = [os.path.join(out_folder, f"{publication}.csv")]
    snakemake.params = {}
    dms_info_file = os.path.join(project_root, "data/DMS/dms_curated.csv")
    info_df = pd.read_csv(dms_info_file)

    snakemake.params["info_df"] = info_df
    snakemake.params["metadata_file"] = metadata_file


def get_complex_metadata(publication:str, antibody: str, antigen: str, metadata: Dict) -> Dict:
    if "mason21" in publication:
        publication = "mason21_optim_therap_antib_by_predic"
    publication_data = metadata[publication]
    for complex in publication_data["complexes"]:
        if complex["antigen"]["name"] == antigen and complex["antibody"]["name"] == antibody:
            return complex["pdb"]

    raise RuntimeError(f"Complex not found in Metadata: {publication}, {antibody}, {antigen}")


def get_chain_infos(metadata: Dict) -> Dict:
    chain_info = {}
    for protein, chains in metadata["chains"].items():
        prot_id = 1 if protein == "antibody" else 0
        for chain in chains:
            chain_info[chain.lower()] = prot_id

    return chain_info


def get_mutation(original_mutation_code: str, mutation_mapping: Dict, **kwargs):
    if original_mutation_code == "WT" or not isinstance(original_mutation_code, str):
        return original_mutation_code

    elif original_mutation_code in mutation_mapping:
        return mutation_mapping[original_mutation_code]

    else:
        print(kwargs["antibody"], kwargs["antigen"], kwargs["publication"])
        return "Not Found"

def get_extended_df(complex_log: str, full_df: pd.DataFrame):
    path_components = complex_log.split("/")
    antibody, antigen = path_components[-1].split(".")[0].split("_")
    publication = path_components[-2]

    mutation_mapping_path = complex_log.replace(".logs", ".json")
    with open(mutation_mapping_path) as f:
        mutation_mapping = json.load(f)

    complex_df = full_df[(full_df["publication"] == publication) & (full_df["antibody"] == antibody) &
                         (full_df["antigen"] == antigen)].copy()

    complex_df["mutation_code"] = complex_df["original_mutation"].apply(lambda code: get_mutation(code, mutation_mapping,
                                                                                                  antigen=antigen, antibody=antibody,
                                                                                                  publication=publication))

    complex_df = complex_df[complex_df["mutation_code"] != "Not Found"]

    complex_df["mutation_code"] = complex_df["mutation_code"].replace({"": "original"})

    complex_metadata = get_complex_metadata(publication, antibody, antigen, metadata)
    chain_infos = get_chain_infos(complex_metadata)
    complex_df["chain_infos"] = str(chain_infos)

    complex_df["mutation_code"] = complex_df["mutation_code"].fillna("original")

    complex_df["filename"] = complex_df.apply(lambda row:
                                              os.path.join(row["publication"],
                                                           row["antibody"] + "_" + row["antigen"],
                                                           row["mutation_code"] + ".pdb"),
                                              axis=1)

    complex_df["ab_ag"] = complex_df.apply(lambda row: row["antibody"] + "_" + row["antigen"], axis=1)
    complex_df["pdb"] = complex_df.apply(lambda row:row["publication"] + ":" + row["antibody"] + ":" + row["antigen"], axis=1)

    complex_df["index"] = complex_df.apply(lambda row:row["pdb"] + "-" + row["mutation_code"].lower(), axis=1)
    complex_df = complex_df.set_index("index")
    complex_df.index.name = ""

    complex_df = complex_df[["pdb", "publication", "mutation_code", "data_location", "filename", "-log(Kd)", "E", "NLL", "chain_infos", "original_mutation"]]

    return complex_df


out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

publication = ".".join(os.path.split(out_path)[1].split(".")[:-1])

with open(snakemake.params["metadata_file"]) as f:
    metadata = yaml.safe_load(f)

data_df = snakemake.params["info_df"]

# add values for all complexes
data_df["data_location"] = "DATA"
data_df["original_mutation"] = data_df["mutation_code"]
data_df["pdb"] = data_df.apply(lambda row: ":".join([row["publication"], row["antibody"], row["antigen"]]), axis=1)

all_dataset_slices = []

for complex_log in snakemake.input:
    if publication in complex_log:
        complex_df = get_extended_df(complex_log, data_df)
        all_dataset_slices.append(complex_df)

final_df = pd.concat(all_dataset_slices)

final_df = final_df[~final_df.index.duplicated(keep='first')]

final_df.to_csv(out_path)