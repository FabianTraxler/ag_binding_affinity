# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import yaml
import os
import pandas as pd
from typing import Dict
from common import substitute_chain


with open(snakemake.input["metadata_file"], "r") as f:
    METADATA = yaml.safe_load(f)

# TODO to I need this?
def get_complex_metadata(publication:str, antibody: str, antigen: str, metadata: Dict) -> Dict:
    if "mason21" in publication:
        publication = "mason21_optim_therap_antib_by_predic"
    publication_data = metadata[publication]
    for complex in publication_data["complexes"]:
        if complex["antigen"]["name"] == antigen and complex["antibody"]["name"] == antibody:
            return complex["pdb"]

    raise RuntimeError(f"Complex not found in Metadata: {publication}, {antibody}, {antigen}")

# TODO unused, replace?
def get_mutation(original_mutation_code: str, mutation_mapping: Dict, **kwargs):
    if original_mutation_code == "WT" or not isinstance(original_mutation_code, str):
        return original_mutation_code

    elif original_mutation_code in mutation_mapping:
        return mutation_mapping[original_mutation_code]

    else:
        print(kwargs["antibody"], kwargs["antigen"], kwargs["publication"])
        return "Not Found"


# Load DMS CSV
df = pd.read_csv(snakemake.input.dms_curated)
# remove this publication because they have redundant information to mason21_comb_optim_therap_antib_by_predic_combined_H3_3. TODO really? why not augmentation?  # TODO should anyways be filtered on Snakefile level?
df = df[~df["publication"].isin(["mason21_comb_optim_therap_antib_by_predic_combined_H3_2", "mason21_comb_optim_therap_antib_by_predic_combined_H3_1"])]
df = df[df["publication"] == snakemake.wildcards.publication]
df = df.reset_index(drop=True)

# Cleanup
df["mutation_code"] = df["mutation_code"].replace({"": "WT"}).fillna("WT")
df["ab_ag"] = df.apply(lambda row: row["antibody"] + "_" + row["antigen"], axis=1)
df["pdb"] = df.apply(lambda row: ":".join([row["publication"], row["antibody"], row["antigen"]]), axis=1) # TODO  what's this???
df["index"] = df.apply(lambda row:row["pdb"] + "-" + row["mutation_code"].lower(), axis=1)
df = df.set_index("index")
df.index.name = ""
df["data_location"] = "DATA"  # TODO delete?

df["original_mutation"] = df["mutation_code"]
df["mutation_code"] = df.apply(lambda row: substitute_chain(METADATA, row.mutation_code, snakemake.wildcards.publication, row.antibody, row.antigen), axis=1)

# Filename
df["filename"] = df.apply(lambda row:
                                            os.path.join(row["publication"],
                                                        row["antibody"] + "_" + row["antigen"],
                                                        row["mutation_code"] + ".pdb"),
                                            axis=1)

# Cleanup
df = df[["pdb", "publication", "mutation_code", "data_location", "filename", "-log(Kd)", "E", "NLL", "original_mutation"]]
df = df[~df.index.duplicated(keep='first')]
# Save
out_path = snakemake.output[0]
df.to_csv(out_path)
