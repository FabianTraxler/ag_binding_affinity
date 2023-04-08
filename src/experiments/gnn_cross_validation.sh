#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ~/miniconda3/etc/profile.d/conda.sh # change to $CONDA_PATH/etc/profile.d/conda.sh
#conda env create -f $SCRIPT_DIR/../../envs/abag_env.yaml || conda env update --name abag_affinity --file $SCRIPT_DIR/../../envs/abag_env.yaml
conda activate abag_affinity

pip install -e $SCRIPT_DIR/..
pip install -e $SCRIPT_DIR/../../../other_repos/DeepRefine # change to DEEPREFINE DIR and add git download

python -m abag_affinity.main --args_file $SCRIPT_DIR/../../results/best_model_config.json --cross_validation --no-tqdm_output "$@"
