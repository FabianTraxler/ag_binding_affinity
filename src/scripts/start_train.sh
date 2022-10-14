#!/bin/bash

# link login node paths to compute node paths
mkdir -p /msc/home/ftraxl96
ln -s /root/miniconda3 /msc/home/ftraxl96
ln -s /root/projects /msc/home/ftraxl96

# activate login node conda enviroment
source /msc/home/ftraxl96/miniconda3/etc/profile.d/conda.sh
conda activate abag_cluster

# start training
python -m abag_affinity.main "$@"