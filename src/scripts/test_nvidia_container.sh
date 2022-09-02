#!/bin/sh

# srun --mem=100G -q a100 --gres=gpu:a100:1 --container-image=nvcr.io#nvidia/pytorch:21.10-py3 "/root/deepGlue/src/test_nvidia_container.sh"

# srun --mem=100G -q a100 --gres=gpu:a100:1 --container-image=nvcr.io#muwsc/cemsii/deepglue:latest "/root/deepGlue/src/test_nvidia_container.sh"

# Goal: Python 3.8.12
# Goal: pytorch-1.10.0

python --version

echo "To /root"
cd /root
ls
echo "--"

# conda init bash
# source .bashrc
# conda env create -f deepGlue/deepGlue_GPU_VSC3_no_builds.yml
# conda activate deepGlue_GPU

cd deepGlue/src
#python test_GPU.py

cd ../docker
#python test.py

echo "Done."

