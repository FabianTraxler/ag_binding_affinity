#!/bin/bash

mkdir -p $1;

cd $1 || exit;
git clone https://github.com/piercelab/antibody_benchmark.git;
