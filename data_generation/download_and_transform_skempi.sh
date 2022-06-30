#!/bin/bash

# We expect the repository to be in the home directory of the user
# some configuration

cd SKEMPI

# download data
# mkdir -p data/SKEMPI_v2
scripts/download_skempi.sh -p ./

# transform the data
snakemake --use-conda --cores 1
