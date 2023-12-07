# %%
from typing import Dict, List, Tuple, Union, Optional, Callable

import sys
import wandb
import numpy as np
import math
import pandas as pd
import os
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
import logging
import torch
from scipy import stats
from torch.optim import Adam
from torch.utils.data import DataLoader as DL_torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from functools import partial
# import pdb

from abag_affinity.dataset import AffinityDataset
from abag_affinity.dataset.advanced_data_utils import complexes_from_dms_datasets, get_bucket_dataloader, load_datasets
from abag_affinity.model import AffinityGNN, TwinWrapper
from abag_affinity.train.wandb_config import configure
from abag_affinity.utils.config import get_data_paths, read_config
from abag_affinity.utils.visualize import plot_correlation
from abag_affinity.utils.argparse_utils import read_args_from_file, parse_args
from abag_affinity.train.utils import load_model
# %load_ext autoreload
# %autoreload 2
# %%
args_file = "base_args.txt"

sys.argv = sys.argv[:1]
args = parse_args(args_file=args_file)

config = read_config(args.config_file)

use_cuda = False
device = 'cpu'

train_data, val_datas = load_datasets(config, args.target_dataset, args.validation_set, args)
model = load_model(train_data.num_features, train_data.num_edge_features, args.target_dataset, args, device)

# wandb_benchmark_log = run_and_log_benchmarks(model, args, wandb_inst)
# %%
args = parse_args(artifical_args={"args_file": 'base_args.txt'})

config = read_config(args.config_file)

use_cuda = False
device = 'cpu'

train_data, val_datas = load_datasets(config, args.target_dataset, args.validation_set, args)
model = load_model(train_data.num_features, train_data.num_edge_features, args.target_dataset, args, device)

