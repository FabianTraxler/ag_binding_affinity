"""This module provides all training utilities for the antibody-antigen binding affinity prediction"""
import logging
import sys
from argparse import ArgumentParser, Namespace, Action
from pathlib import Path
from typing import Dict
import wandb

from abag_affinity.utils.config import read_config
from abag_affinity.train import (bucket_train, cross_validation, model_train,
                                 pretrain_model)

# different training modalities
training = {
    "bucket_train": bucket_train,
    "pretrain_model": pretrain_model,
    "model_train": model_train,
    "cross_validation": cross_validation
}

model2data = {
    "GraphConv": "BoundComplexGraphs",
    "GraphAttention": "BoundComplexGraphs",
    "FixedSizeGraphConv": "BoundComplexGraphs",
    "DeepRefineBackbone": "DeepRefineInputs",
    "DDGBackboneFC": "DDGBackboneInputs",
    "KpGNN": "HeteroGraphs"
}

enforced_node_type = {
    "DDGBackboneFC": "residue",
    "DeepRefineBackbone": "atom"
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

    return main_logger


class BooleanOptionalAction(Action):
    """Taken from argparse Python >= 3.9 and added here to suppoert python < 3.9"""
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


def parse_args() -> Namespace:
    """ CLI arguments parsing functionality

    Parse all CLI arguments and set not available to default values

    Returns:
        Namespace: Class with all arguments
    """

    parser = ArgumentParser(description='CLI for using the abag_affinity module')

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument("-t", "--train_strategy", type=str, help='The training strategy to use',
                          choices=["bucket_train", "pretrain_model", "model_train", "cross_validation"],
                          default="model_train")
    optional.add_argument("-m", "--model_type", type=str, help='The type of model to train',
                          choices=["GraphConv", "GraphAttention", "FixedSizeGraphConv", "DDGBackboneFC", "KpGNN",
                                   "DeepRefineBackbone"], default="GraphConv")
    optional.add_argument("-d", "--data_type", type=str, help='The type of dataset (graphs) to use for training',
                          choices=["BoundComplexGraphs", "HeteroGraphs", "DeepRefineInputs",
                                   "DDGBackboneInputs"], default="BoundComplexGraphs")
    optional.add_argument("-b", "--batch_size", type=int, help="Batch size used for training", default=1)
    optional.add_argument("-e", "--max_epochs", type=int, help="Max number of training epochs", default=100)
    optional.add_argument("-lr", "--learning_rate", type=float, help="Initial learning rate", default=1e-4)
    optional.add_argument("-p", "--patience", type=int,
                          help="Number of epochs with no improvement until end of training",
                          default=10)
    optional.add_argument("-n", "--node_type", type=str, help="Type of nodes in the graphs", default="residue",
                          choices=["residue", "atom"])
    optional.add_argument("--max_num_nodes", type=int, help="Maximal number of nodes for fixed sized graphs",
                          default=None)
    optional.add_argument("--interface_hull_size", type=int, help="Size of the extension from interface to generate interface hull",
                          default=None)
    optional.add_argument("--scale_values", action=BooleanOptionalAction, help="Scale affinity values between 0 and 1",
                          default=False)
    optional.add_argument("-l", "--loss_function", type=str, help="Type of Loss Function", default="L1",
                          choices=["L1", "L2"]
                          )
    optional.add_argument("-w", "--num_workers", type=int, help="Number of workers to use for data loading", default=0)
    optional.add_argument("-wdb", "--use_wandb", action=BooleanOptionalAction, help="Use Weight&Bias to log training process",
                          default=False)
    optional.add_argument("--wandb_mode", type=str, help="Mode of Weights&Bias Process", choices=["online", "offline"],
                          default="online")
    optional.add_argument("--wandb_name", type=str, help="Name of the Weight&Bias logs", default="")
    optional.add_argument("--init_sweep", action=BooleanOptionalAction, help="Use Weight&Bias sweep to search hyperparameter space",
                          default=False)
    optional.add_argument("--sweep_config", type=str, help="Path to the configuration file of the sweep",
                          default=(Path(__file__).resolve().parents[1] / "config.yaml").resolve())
    optional.add_argument("--sweep_id", type=str, help="The sweep ID to use for all runs")
    optional.add_argument("--sweep_runs", type=int, help="Number of runs to perform in this sweep instance",
                          default=30)
    optional.add_argument("-v", "--validation_set", type=int, help="Which validation set to use", default=1,
                          choices=[1, 2, 3])
    optional.add_argument("-c", "--config_file", type=str,
                          help="Path to config file for datasets and training strategies",
                          default=(Path(__file__).resolve().parents[1] / "config.yaml").resolve())
    optional.add_argument("--verbose", action=BooleanOptionalAction, help="Print verbose logging statements",
                          default=False)
    optional.add_argument("--preprocess_graph", action=BooleanOptionalAction,
                          help="Compute graphs beforehand to speedup training (especially for DeepRefine",
                          default=False)
    optional.add_argument("--save_graphs", action=BooleanOptionalAction, help="Saves computed graphs to speed up training in later epochs",
                          default=False)
    optional.add_argument("--force_recomputation", action=BooleanOptionalAction,
                          help="Force recomputation of graphs - deletes folder containing processed graphs",
                          default=False)
    optional.add_argument("--no_shuffle", action=BooleanOptionalAction,
                          help="Shuffle train-dataloader",
                          default=False)

    args = parser.parse_args()
    args.config = read_config(args.config_file)

    if args.wandb_name == "":
        args.wandb_name = f'{args.model_type}' \
                          f' -d {args.data_type}' \
                          f' -t {args.train_strategy}' \
                          f' -n {args.node_type}'

    # check arguments
    if args.data_type != model2data[args.model_type]:
        args.data_type = model2data[args.model_type]
    if args.model_type in enforced_node_type and args.node_type != enforced_node_type[args.model_type]:
        args.__dict__["node_type"] = enforced_node_type[args.model_type]

    return args


def run_sweep(args: Namespace, logger):
    import traceback
    import shutil
    import os

    def sweep_train():
        wandb.init()
        config = wandb.config
        for param in config.keys():
            args.__dict__[param] = config[param]
            if param == "model_type":
                args.__dict__["data_type"] = model2data[config[param]]
                if config[param] in enforced_node_type:
                    args.__dict__["node_type"] = enforced_node_type[config[param]]
        logger.info(f"Performing {args.train_strategy} for {args.model_type} using {args.data_type}")
        try:
            training[args.train_strategy](args)
            # remove all processed graphs to keep disk clean
            if os.path.exists(args.config["cleaned_pdbs"]) and os.path.isdir(args.config["cleaned_pdbs"]):
                shutil.rmtree(args.config["processed_graph_path"])
            if os.path.exists(args.config["cleaned_pdbs"]) and os.path.isdir(args.config["cleaned_pdbs"]):
                shutil.rmtree(args.config["cleaned_pdbs"])

            wandb.finish(0)
        except Exception:
            logger.error(traceback.print_exc())
            wandb.finish(-1)

    logger.info(f"Starting {args.sweep_runs} runs in this instance")
    wandb.agent(args.sweep_id, function=sweep_train, count=args.sweep_runs, project="abag_binding_affinity")


def main() -> Dict:
    """ Main functionality of the abag_affinity module

    Provides train functionalities based on provided CLI arguments

    Returns:
        Dict: Results of training
    """
    args = parse_args()
    logger = logging_setup(args)

    if args.init_sweep:
        sweep_configuration = args.config["HYPERPARAMETER_SEARCH"]
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='abag_binding_affinity')
        args.sweep_id = sweep_id
        logger.info(f"W&B Sweep initialized with ID: {args.sweep_id}")

    if args.sweep_id is not None and args.sweep_runs > 0:
        run_sweep(args, logger)
    else:
        logger.info(f"Performing {args.train_strategy} for {args.model_type} using {args.data_type}")
        results = training[args.train_strategy](args)

        return results


if __name__ == "__main__":
    main()
