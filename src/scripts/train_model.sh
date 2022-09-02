#!/bin/bash

#mkdir -p /msc/home/ftraxl96
#ln -s /root/miniconda3 /msc/home/ftraxl96
#ln -s /root/projects /msc/home/ftraxl96

#export ABAG_PATH=/msc/home/ftraxl96/projects/ag_binding_affinity

#source $HOME/miniconda3/bin/activate
conda activate abag_cluster

#cd $HOME/projects/ag_binding_affinity/
#python setup.py install

cd $HOME/projects/ag_binding_affinity/src/abag_affinity/
python main.py -t bucket_train -m GraphAttention -d InterfaceGraphs -w 8 -b 5 --verbose True