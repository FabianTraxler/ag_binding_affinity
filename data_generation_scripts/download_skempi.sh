#!/bin/bash
cd ~/ag_binding_affinity

mkdir data
mkdir ./data/SKEMPI_v2
mkdir ./data/SKEMPI_v2/PDBs
mkdir ./data/SKEMPI_v2/mutated_pdb

cd ./data/SKEMPI_v2

wget https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

wget https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz

tar -xvzf SKEMPI2_PDBs.tgz -C ./PDBs