#!/bin/bash

pdb_path=""

while getopts p: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        p) pdb_path=${OPTARG};;
    esac
done

if [ -z "$pdb_path" ]
then
  echo "No path given"
  exit 1;
fi


if [ -d $pdb_path ]
then
  echo "Data already downloaded"
else
  echo "Downloading Files"
  mkdir -p $pdb_path
  cd $pdb_path || exit

  wget https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

  wget https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz

  tar -xvzf SKEMPI2_PDBs.tgz -C $pdb_path
fi