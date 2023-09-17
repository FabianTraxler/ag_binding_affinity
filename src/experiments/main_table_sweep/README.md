Main training sweep
====================================

See also https://github.com/moritzschaefer/guided-protein-diffusion/issues/308 for more details.

The goal of this experiment is to run a sweep that trains and evaluates all models we need to fill in the main table of our paper, which includes the following affinity prediction models:
- GNN
- IPA
- GNN + DMS augmentation
- IPA + DMS augmentation

The base arguments are defined in 'base_args.txt', and "sweep_config.yaml" contains all the sweep-specific settings.
