#!/bin/bash
# Current directory path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# SCRIPT="python ../../abag_affinity/main.py"
SCRIPT="${DIR}/../../scripts/cluster_affinity"

$SCRIPT @${DIR}/base_args.txt

