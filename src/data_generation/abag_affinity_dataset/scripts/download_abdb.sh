#!/bin/bash

mkdir -p $1
cd $1 || exit;

# download AbDB structures in martin numbering scheme
wget -O abdb_pdbs.tar.bz2 http://www.abybank.org/abdb/Data/LH_Protein_Martin.tar.bz2

mkdir -p $2
tar -xf abdb_pdbs.tar.bz2

mv LH_Protein_Martin/* $2
rm -rf LH_Protein_Martin

# download information on redundant AbDB structures
wget -O $3 http://www.abybank.org/abdb/Data/Redundant_files/Redundant_LH_Protein_Martin.txt