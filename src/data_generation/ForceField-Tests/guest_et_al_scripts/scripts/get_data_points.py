import os

import pandas as pd
from ast import literal_eval

df = pd.read_csv(snakemake.input[0], index_col=0)

files = df["filename"].tolist()
pdb_path = snakemake.params["pdb_path"]

def get_chains(chain_infos):
    chain_infos = literal_eval(chain_infos)
    ab_chains = []
    ag_chains = []
    for chain, protein in chain_infos.items():
        if protein == 0:
            ab_chains.append(chain)
        else:
            ag_chains.append(chain)

    return "".join(ab_chains), "".join(ag_chains)

all_abag_chains = df["chain_infos"].apply(get_chains)

file_lines = []

for file, abag_chains in zip(files, all_abag_chains):
    if os.path.exists(os.path.join(pdb_path, file)):
        filename = file.split(".")[0]
        file_lines.append(",".join([filename, *abag_chains]))

with open(snakemake.output[0], "w") as f:
    f.write("\n".join(file_lines))