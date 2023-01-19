"""This module provides all training utilities for the antibody-antigen binding affinity prediction"""
import logging
import random
import sys
from argparse import Namespace
from typing import Dict
import wandb
import subprocess

from abag_affinity.utils.argparse_utils import parse_args, enforced_node_type
from abag_affinity.train import (bucket_train, cross_validation, model_train,
                                 pretrain_model)

# different training modalities
training = {
    "bucket_train": bucket_train,
    "pretrain_model": pretrain_model,
    "model_train": model_train,
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


def run_sweep(args: Namespace, logger):
    import traceback

    ATOM_NODES_MULTIPLIKATOR = 5

    def sweep_train():
        run = wandb.init(mode=args.wandb_mode)
        run_dir = run.dir[:-6]

        try:
            config = wandb.config
            for param in config.keys():
                if config[param] == "None":
                    param_value = None
                else:
                    param_value = config[param]

                if param == "max_num_nodes" and param_value is None and "aggregation_method" in config and config["aggregation_method"] == "fixed_size":
                    continue # ignore since it is manually overwritten below
                if param == "node_type" and "pretrained_model" in config and config["pretrained_model"] in enforced_node_type:
                    continue # ignore since it is manually overwritten below

                args.__dict__[param] = param_value
                if param == "pretrained_model":
                    if config[param] in enforced_node_type:
                        args.__dict__["node_type"] = enforced_node_type[config[param]]
                        config["node_type"] = enforced_node_type[config[param]]

                if param == 'aggregation_method':
                    if param_value == "fixed_size" and "max_num_nodes" in config and config["max_num_nodes"] == "None":
                        max_num_nodes_values = [ int(value) for value in args.config["HYPERPARAMETER_SEARCH"]["parameters"]["max_num_nodes"]["values"] if value != "None"]
                        args.max_num_nodes = random.choice(max_num_nodes_values)
                        config["max_num_nodes"] = args.max_num_nodes

            # adapt hyperparameter based on node type
            if args.node_type == "atom" and args.max_num_nodes is not None:
                args.max_num_nodes = int(args.max_num_nodes * ATOM_NODES_MULTIPLIKATOR)

            # adapt batch size bases on node type
            if args.node_type == "atom":
                args.batch_size = int(args.batch_size / ATOM_NODES_MULTIPLIKATOR) + 1

            # adapt learning rate bases on batch size
            args.learning_rate = args.learning_rate * args.batch_size

            args.tqdm_output = False # disable tqdm output to reduce log syncing

            logger.info(f"Performing {args.train_strategy}")
            training[args.train_strategy](args)
            wandb.finish(0)
        except Exception as e:
            # log errors before finishing job
            logger.error(e)
            logger.error(traceback.print_exc())
            wandb.finish(-1)

        if args.wandb_mode == "offline":
            command = f'wandb sync --id {run.id} {run_dir}'
            subprocess.run(command, shell=True)

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
        logger.info(f"Performing {args.train_strategy}")
        if args.cross_validation:
            results = cross_validation(args)
        else:
            results = training[args.train_strategy](args)

        return results


if __name__ == "__main__":
    main()
