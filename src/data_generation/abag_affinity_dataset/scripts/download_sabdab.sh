#!/bin/bash

mkdir -p $1
cd $1 || exit;

# download SAbDab summary
wget -O sabdab_summary.tsv $2
