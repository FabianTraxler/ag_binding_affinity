Initial structures and mutations were determined using a notebook by Moritz Schaefer (will eventually become a Snakemake pipeline here).

For now, this module only hosts a small pipeline that predicts mutated AF2 complexes. The following tweaks were applied t colabfold to allow this (see also [[id:d3f15b68-62fe-4cf4-bb78-46f09c928cef][Using multimer templates for AlphaFold multimer prediciton]] in Moritz' notes):


- Set template_mask to 1 everywhere
  https://github.com/deepmind/alphafold/blob/main/alphafold/model/modules_multimer.py#L657
- Generate only 1 template (max_hit=1 instead of 20)
