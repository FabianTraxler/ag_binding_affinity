#!/bin/bash

mkdir -p $3
cd $3 || exit;

wget -O pdbs.tar.bz2 https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_PP.tar.gz

tar -xf pdbs.tar.bz2

echo "$1"
mv PP/index/INDEX_general_PP.2020 $1

mkdir -p $2

cd PP || exit;
find * -name "*.ent.pdb" | while read -r file; do mv $file "$2/${file%.ent.pdb}.pdb"; done

cd ..
rm -rf PP