#!/bin/bash
path=""

while getopts p: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        p) path=${OPTARG};;
    esac
done

if [ -z "$path" ]
then
  echo "No path given"
  exit 1;
fi


if [ -d $path ]
then
  echo "Data already downloaded"
else
  echo "Downloading Files"
  mkdir -p $path
  cd $path || exit

  wget http://www.abybank.org/abdb/Data/NR_LH_Protein_Martin.tar.bz2

  tar -xvzf NR_LH_Protein_Martin.tar.bz2
fi