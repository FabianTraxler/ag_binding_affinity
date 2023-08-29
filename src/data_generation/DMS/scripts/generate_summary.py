# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import yaml
import os
import pandas as pd
from typing import Dict
from common import substitute_chain, get_chain_info


with open(snakemake.input["metadata_file"], "r") as f:
    METADATA = yaml.safe_load(f)

# TODO unused, remove?
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

# Filter on publication
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

# Filter complexes that are not in the metadata NOTE: In the longer term, I should be able to include all complexes, thus removing this code block!

missing_groups = []
for group_index, group in df.groupby(["publication", "antibody", "antigen"]):
    try:
        chain_info = get_chain_info(METADATA, group_index[0], group_index[1], group_index[2])
    except Exception as e:
        missing_groups.append(group_index)
        continue
print(f"Removing {len(missing_groups)} groups because they are not in the metadata: {missing_groups}")
for group in missing_groups:
    df = df.drop(df[(df['publication'] == group[0]) & (df['antibody'] == group[1]) & (df['antigen'] == group[2])].index)

# Substitute chain in mutation code
if len(df) > 0:  # may happen that df is empty because all complexes are commented in the metadata. then in here everything would fail
    df["mutation_code"] = df.apply(lambda row: substitute_chain(METADATA, row.mutation_code, snakemake.wildcards.publication, row.antibody, row.antigen), axis=1)

    # Filename
    df["filename"] = df.apply(lambda row:
                            os.path.join(row["publication"],
                                        row["antibody"] + "_" + row["antigen"],
                                        row["mutation_code"].replace(";", "_") + ".pdb"),
                            axis=1)

    # Cleanup
    df = df[["pdb", "publication", "mutation_code", "data_location", "filename", "-log(Kd)", "E", "NLL", "original_mutation"]]
    df = df[~df.index.duplicated(keep='first')]

    # # Trim number of datapoints (performed in its own rule now)
    # if snakemake.params.num_max_datapoints:
    #     def _subsample_complex(complex_df):
    #         """
    #         Randomly subsample the complex data such that the distribution is preserved
    #         """
    #         if len(complex_df) <= snakemake.params.num_max_datapoints:
    #             return complex_df
    #         else:
    #             return complex_df.sample(snakemake.params.num_max_datapoints, replace=False)

    #     df = df.groupby(["publication", "antibody", "antigen"], as_index=False).apply(_subsample_complex)

# Save
out_path = snakemake.output[0]
df.to_csv(out_path)

