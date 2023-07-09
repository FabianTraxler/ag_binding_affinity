import yaml
from abag_affinity.utils.pdb_processing import clean_and_tidy_pdb

with open(snakemake.input.metadata, "r") as f:
    metadata = yaml.safe_load(f)

# extract the chains
antibody, antigen = snakemake.wildcards.complex.split("_") # TODO correct?

for complex in metadata[snakemake.wildcards.study]["complexes"]:
    if complex["antigen"]["name"] == antigen and complex["antibody"]["name"] == antibody:
        break
else:
    raise ValueError(f"Complex {snakemake.wildcards.complex} not found")

rename_chains = {**complex["pdb"]["chains"]["antibody"], **complex["pdb"]["chains"]["antigen"]}
rename_chains = {old_id: new_id for new_id, old_id in rename_chains.items()}  # revert, because this is how we saved it in the metadata file

# fix_insert is False to make sure that the mutations afterwards are applied correctly
clean_and_tidy_pdb(snakemake.wildcards.complex, snakemake.input.pdb, snakemake.output[0], fix_insert=False, rename_chains=rename_chains)
