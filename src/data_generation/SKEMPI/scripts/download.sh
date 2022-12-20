#!/bin/bash

mkdir -p $1
cd $1 || exit

wget -O $2 https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

wget https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz
tar -xvzf SKEMPI2_PDBs.tgz

mkdir -p $3
mv PDBs/[a-zA-Z0-9]*.pdb $3
rm -rf PDBs