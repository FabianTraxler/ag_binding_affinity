"""
Run tests (benchmark/SKEMPI) based on trained models
"""

from guided_protein_diffusion.models.classifier import _load_affinity_model
from abag_affinity.train.utils import run_and_log_benchmarks
from argparse import ArgumentParser
import wandb
from pathlib import Path
import pandas as pd


parser = ArgumentParser()

def wandb_init(name):
    if name is not None:
        run = wandb.init(project="abag_binding_affinity")
        run.name = name
        return wandb  # this should become binary (True/False). It is the same logic right now as everywhere else with wandb_inst. The problem is that we pass wandb objects in the first place. This should be eliminated everywhere.
    else:
        return None

parser.add_argument("model_path", type=str)
parser.add_argument("--output_path", type=Path)
parser.add_argument("--wandb", help="name of wandb job", type=wandb_init)

args = parser.parse_args()

aff_pred_net = _load_affinity_model(args.model_path)

train_args = aff_pred_net.hparams.args

results = run_and_log_benchmarks(aff_pred_net, train_args, args.wandb_inst)

if args.output_path is not None:
    pd.Series(results).to_csv(args.output_path)

print(pd.Series(results))
