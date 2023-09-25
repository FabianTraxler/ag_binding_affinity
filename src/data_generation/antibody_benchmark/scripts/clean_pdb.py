
import pandas as pd
from abag_affinity.utils.pdb_processing import clean_and_tidy_pdb
from tempfile import NamedTemporaryFile

import subprocess

df = pd.read_csv(snakemake.input.csv, index_col="pdb")
chain_infos = eval(df.loc[snakemake.wildcards.pdb_id.lower(), "chain_infos"])  # only for chain replacement

# Concat the files in snakemake.input.antibody snakemake.input.antigen into a temporary file
with NamedTemporaryFile() as tmp:
    tmp.write(open(snakemake.input.antibody).read().encode())
    tmp.write(open(snakemake.input.antigen).read().encode())
    tmp.flush()

    clean_and_tidy_pdb(snakemake.wildcards.pdb_id, tmp.name, snakemake.output[0], rename_chains=chain_infos,
                       max_h_len=snakemake.params.max_h_len, max_l_len=snakemake.params.max_l_len)
