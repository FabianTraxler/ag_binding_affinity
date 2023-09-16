#!/bin/bash
# Current directory path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# SCRIPT="${DIR}/../../scripts/cluster_affinity"
SCRIPT="$HOME/guided-protein-diffusion/scripts/cluster_affinity"
# SCRIPT="python -m ipdb -c c ../../abag_affinity/main.py"
# SCRIPT="python ../../abag_affinity/main.py"

$SCRIPT @${DIR}/base_args.txt

