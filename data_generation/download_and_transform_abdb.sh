#!/bin/bash

# We expect the repository to be in the home directory of the user
# some configuration

# Using a proper snakemake-structure (e.g. one single main Snakefile in the project's root directory will avoid directory issues. Note you can also include additional snakemake rules using the include-directive in your Snakefile (use .smk as file extension. Syntax corresponds exactly to the main Snakefile))
cd AbDb

# download data
scripts/download_abdb.sh -p data/AbDb

# transform the data
snakemake --use-conda --cores 1
