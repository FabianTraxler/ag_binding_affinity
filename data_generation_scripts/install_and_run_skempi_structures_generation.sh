#!/bin/bash
cd ~/ag_binding_affinity

apt update

# apt install -y everything-you-might-need
conda env update -n base --file rosetta_enviroment.yml

# some configuration
mkdir logs
export ABAG_PATH=~/ag_binding_affinity

# install package with utilities
python setup.py install

# run scripts
python data_generation_scripts/generate_skempi_structures.py
