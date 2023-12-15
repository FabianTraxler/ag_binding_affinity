from pathlib import Path
from Bio.PDB import PDBParser
import string

from abag_affinity.utils.pdb_processing import clean_and_tidy_pdb

input_fn = Path(snakemake.input.pdb)
input_fn_stem = input_fn.stem

# Use BioPython to read the chains within the PDB file

parser = PDBParser()
structure = parser.get_structure("test", input_fn)

# Now read out the chain IDs
chain_ids = [chain.id for chain in structure.get_chains()]

# Read out chain IDs from filename
if "_" not in input_fn_stem:  # original
    # take matching mutation
    other_mutation_file = next(input_fn.parent.glob(f"{input_fn_stem}_*.pdb"))
    pdb_id, ab_chains, ag_chains, _ = other_mutation_file.name.split("_")
    mutation = "original"
else:
    pdb_id, ab_chains, ag_chains, mutation = input_fn_stem.split("_")
drop_chains = set(chain_ids) - (set(ab_chains) | set(ag_chains))
ag_chains = list(ag_chains)

# Rename chains to A, B, C, ...
sorted(ag_chains)
rename_chains = dict(zip(ag_chains, string.ascii_uppercase))
rename_chains[ab_chains[0]] = "H"
if len(ab_chains) > 1:
    rename_chains[ab_chains[1]] = "L"

for k in drop_chains:
    rename_chains[k] = None

for k in list(rename_chains.keys()):
    if rename_chains[k] == k:
        del rename_chains[k]  # no need to identity-rename chains that are already named correctly

# Clean and tidy
clean_and_tidy_pdb(input_fn_stem,
                   input_fn,
                   snakemake.output.pdb,
                   fix_insert=True,
                   rename_chains=rename_chains,
                   max_h_len=snakemake.params.max_h_len,
                   max_l_len=snakemake.params.max_l_len)
