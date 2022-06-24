#!/bin/bash

# We expect the repository to be in the home directory of the user
# some configuration
export ABAG_PATH=~/ag_binding_affinity

cd ~/ag_binding_affinity/data_generation/SKEMPI

# download data
scripts/download_skempi.sh -p ~/ag_binding_affinity/data/SKEMPI_v2/PDBs

# transform the data
snakemake --cores 32