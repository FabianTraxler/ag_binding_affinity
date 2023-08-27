# Preface

Initial complex structure PDB drafts were created manually and assisted by AF2. See also

- [[id:d3f15b68-62fe-4cf4-bb78-46f09c928cef][Using multimer templates for AlphaFold multimer prediction]]
- [[id:f1e36ab1-11c4-420d-9d0c-83f1e37f9311][Obtaining antibody binding data (E-values and structures)]]
- muwhpc:~/af_predictions

In the case of phillips21, this led to muwhpc:/msc/home/mschae83/af_predictions/ab_ag_complexes/phillips21_predictions and then ~/ag_binding_affinity/data/prepared_pdbs/phillips21.
All others were generated automatically using rosetta-based mutations onto the given input structures, based on the mutations in the metadata_dms_studies.yaml (TODO check where Fabian implemented this code)

From these, the final complexes were derived and stored in ag_binding_affinity/results/prepared_pdbs (sometimes/always in scfv form)

This is on hold at the moment. See issue https://github.com/moritzschaefer/guided-protein-diffusion/issues/294 for next steps.

# What now?

To make our lives easier and obtain standardized structures, we do the following.

- Derive sequences and original (PDB) template structures from metadata_dms_studies.yaml
- Use AF2 (v2.3) to generate the complexes. We apply patches (see below) to make sure the template uses cross-chain information for the proper positioning of antibody<->antigen

# Output files

Outputs of this pipeline previously went into muwhpc:~/af_predictions/ab_ag_complexes/multi_chain_templates, where this Snakefile has been linked and executed.

HOWEVER, files now go into ~/ag_binding_affinity/results/DMS/

# Install

This pipeline relies on the installation of the `colabfold_multimer_patch`  environment. To install it run `install_colabfold_multimer_patched.sh`

- The script is derived from https://github.com/YoshitakaMo/localcolabfold
- It was modified to work with a pre-installed conda
- I added commands to make (see below)

## Modifications to alphafold/colabfold (now included in the install script so nothing needs to be done)
The following tweaks were applied to colabfold to allow this (see also [[id:d3f15b68-62fe-4cf4-bb78-46f09c928cef][Using multimer templates for AlphaFold multimer prediciton]] in Moritz' notes):

- Set multichain_mask = jnp.ones_like(multichain_mask) (see ./install_colabfold_multimer_patched.sh)
- Generate only 1 template (max_hit=1 instead of 20) colabfold/batch.py

# Previous analyses

- [[id:d3f15b68-62fe-4cf4-bb78-46f09c928cef][Using multimer templates for AlphaFold multimer prediction]]

# Cool test PDBs
Mutated input_pdbs were downloaded from PDB and cleaned manually

- 6WWC and 6WX2 contain single point mutations (good for validation) for [cite:@madan21_mutat_hiv]
- WT: 4FP8, Mutant(VPGSGW): 5UMN for [cite:@wu17_in]  <- most important
- TODO: 6NHP, 6NHQ, 6NHR for [cite:@wu20_differ_ha_h3_h1]
pdb_sort vfp1602_fp8v1:6WWC_S48K.pdb | pdb_tidy | pdb_selchain -c,l,h | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > vfp1602_fp8v1:6WWC_S48K_clean.pdb
