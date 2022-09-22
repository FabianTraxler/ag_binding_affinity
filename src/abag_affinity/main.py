"""This module provides all training utilities for the antibody-antigen binding affinity prediction"""
import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from abag_affinity.train import (bucket_train, cross_validation, model_train,
                                 pretrain_model)
from abag_affinity.utils.config import read_config

# different training modalities
training = {
    "bucket_train": bucket_train,
    "pretrain_model": pretrain_model,
    "model_train": model_train,
    "cross_validation": cross_validation
}


def logging_setup(args: Namespace):
    """ Logging setup functionality

    Based on verbose CLI argument set log-level to verbose or info

    Args:
        args: CLI Arguments

    Returns:
        None
    """
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    main_logger = logging.getLogger("abag_affinity")
    main_logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    main_logger.addHandler(handler)
    main_logger.propagate = False


def parse_args() -> Namespace:
    """ CLI arguments parsing functionality

    Parse all CLI arguments and set not available to default values

    Returns:
        Namespace: Class with all arguments
    """

    parser = ArgumentParser(description='CLI for using the abag_affinity module')

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-t", "--train_strategy", type=str, help='The training strategy to use', required=True,
                        choices=["bucket_train", "pretrain_model", "model_train", "cross_validation"])
    required.add_argument("-m", "--model_type", type=str, help='The type of model to train',
                        choices=["GraphConv", "GraphAttention", "FixedSizeGraphConv", "DDGBackboneFC", "KpGNN",
                                 "DeepRefineBackbone"])
    required.add_argument("-d", "--data_type", type=str, help='The type of dataset (graphs) to use for training',
                        choices=["BoundComplexGraphs", "HeteroGraphs", "DeepRefineInputs",
                                                         "DDGBackboneInputs"])
    optional.add_argument("-b", "--batch_size", type=int, help="Batch size used for training", default=1)
    optional.add_argument("-e", "--max_epochs", type=int, help="Max number of training epochs", default=100)
    optional.add_argument("-lr", "--learning_rate", type=float, help="Initial learning rate", default=1e-4)
    optional.add_argument("-p", "--patience", type=int, help="Number of epochs with no improvement until end of training",
                        default=10)
    optional.add_argument("-n", "--node_type", type=str, help="Type of nodes in the graphs", default="residue",
                        choices=["residue", "atom"])
    optional.add_argument("--max_num_nodes", type=int, help="Maximal number of nodes for fixed sized graphs", default=None)
    optional.add_argument("--interface_hull_size", type=int, help="Maximal number of nodes for fixed sized graphs", default=None)
    optional.add_argument("-w", "--num_workers", type=int, help="Number of workers to use for data loading", default=0)
    optional.add_argument("-wdb", "--use_wandb", type=bool, help="Use Weight&Bias to log training process", default=False)
    optional.add_argument("--wandb_name", type=str, help="Name of the Weight&Bias logs", default="")
    optional.add_argument("-v", "--validation_set", type=int, help="Which validation set to use", default=1,
                        choices=[1,2,3])
    optional.add_argument("-c", "--config_file", type=str, help="Path to config file for datasets and training strategies",
                        default=(Path(__file__).parents[1] / "config.yaml").resolve())
    optional.add_argument("--verbose", type=bool, help="Print verbose logging statements",
                        default=False)
    optional.add_argument("--preprocess_graph", type=bool, help="Compute graphs beforehand to speedup training (especially for DeepRefine",
                        default=False)
    optional.add_argument("--save_graphs", type=bool, help="Saves computed graphs to speed up training in later epochs",
                        default=False)
    optional.add_argument("--force_recomputation", type=bool, help="Force recomputation of graphs - deletes folder containing processed graphs",
                        default=False)
    optional.add_argument("--scale_values", type=bool, help="Scale affinity values between 0 and 1",
                        default=False)

    args = parser.parse_args()
    args.config = read_config(args.config_file)

    if args.wandb_name == "":
        args.wandb_name = f'{args.model_type}' \
                            f' -d {args.data_type}' \
                            f' -t {args.train_strategy}' \
                            f' -n {args.node_type}'

    return args


def main() -> Dict:
    """ Main functionality of the abag_affinity module

    Provides train functionalities based on provided CLI arguments

    Returns:
        Dict: Results of training
    """
    args = parse_args()
    logging_setup(args)
    results = training[args.train_strategy](args)

    return results


if __name__ == "__main__":
    main()


