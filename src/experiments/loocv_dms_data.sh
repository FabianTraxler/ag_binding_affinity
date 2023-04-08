#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ~/miniconda3/etc/profile.d/conda.sh # change to $CONDA_PATH/etc/profile.d/conda.sh
#conda env create -f $SCRIPT_DIR/../../envs/abag_env.yaml || conda env update --name abag_affinity --file $SCRIPT_DIR/../../envs/abag_env.yaml
#conda activate abag_env
conda activate abag_cluster

pip install -e $SCRIPT_DIR/..
pip install -e $SCRIPT_DIR/../../../other_repos/DeepRefine # change to DEEPREFINE DIR and add git download

dms_publications="DMS-b.20_funct_screen_strat_engin_chimer:relative
DMS-madan21_mutat_hiv:relative
DMS-mason21_comb_optim_therap_antib_by_predic_combined_H3_3:relative
DMS-mason21_comb_optim_therap_antib_by_predic_combined_L3_3:relative
DMS-mason21_optim_therap_antib_by_predic_dms_H:relative
DMS-mason21_optim_therap_antib_by_predic_dms_L:relative
DMS-starr21_prosp_covid:relative
DMS-taft22_deep_mutat_learn_predic_ace2:relative
DMS-wu17_in:relative
DMS-wu20_differ_ha_h3_h1:relative"

for publication in $dms_publications
do
  echo "Validation on "$publication
  train_datasets=${dms_publications//$publication/}
  echo "Training with" $train_datasets
  python -m abag_affinity.main --train_strategy bucket_train --target_dataset $publication --validation_size 100 -tld $train_datasets --args_file $SCRIPT_DIR/../../results/best_model_config.json  "$@"
done

#python -m abag_affinity.main.py --args_file $SCRIPT_DIR/../../results/best_model_config.json --cross_validation "$@"
