Initial structures and mutations were determined using a notebook by Moritz Schaefer (will eventually become a Snakemake pipeline here).

For now, this module only hosts a small pipeline that predicts mutated AF2 complexes. The following tweaks were applied to colabfold to allow this (see also [[id:d3f15b68-62fe-4cf4-bb78-46f09c928cef][Using multimer templates for AlphaFold multimer prediciton]] in Moritz' notes):


- Set template_mask to 1 everywhere
  https://github.com/deepmind/alphafold/blob/main/alphafold/model/modules_multimer.py#L657
- Generate only 1 template (max_hit=1 instead of 20) colabfold/batch.py

Mutated input_pdbs were downloaded from PDB and cleaned manually


- 6WWC and 6WX2 contain single point mutations (good for validation) for [cite:@madan21_mutat_hiv]
- WT: 4FP8, Mutant(VPGSGW): 5UMN for [cite:@wu17_in]  <- most important
- TODO: 6NHP, 6NHQ, 6NHR for [cite:@wu20_differ_ha_h3_h1]
pdb_sort vfp1602_fp8v1:6WWC_S48K.pdb | pdb_tidy | pdb_selchain -c,l,h | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > vfp1602_fp8v1:6WWC_S48K_clean.pdb
