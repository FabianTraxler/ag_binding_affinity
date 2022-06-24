#!/bin/bash

# We expect the repository to be in the home directory of the user
# some configuration
export ABAG_PATH=~/ag_binding_affinity

cd ~/ag_binding_affinity/data_generation/AbDb

# download data
scripts/download_abdb.sh -p ~/ag_binding_affinity/data/AbDb

# transform the data
snakemake --use-conda --cores 32