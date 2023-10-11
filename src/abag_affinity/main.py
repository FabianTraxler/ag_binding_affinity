"""This module provides all training utilities for the antibody-antigen binding affinity prediction"""
from datetime import datetime
import numpy as np

import logging
from pathlib import Path
import random
import sys
import time
from argparse import Namespace
from typing import Dict

import torch
from torch.utils.data import DataLoader
import wandb
import subprocess
import yaml
import pytorch_lightning as pl

from abag_affinity.utils.argparse_utils import parse_args, enforced_node_type
from abag_affinity.train import (bucket_train, cross_validation, model_train,
                                 pretrain_model, train_transferlearnings_validate_target)

from abag_affinity.train.utils import run_and_log_benchmarks

# different training modalities
training = {
    "bucket_train": bucket_train,
    "pretrain_model": pretrain_model,
    "model_train": model_train,
    "train_transferlearnings_validate_target": train_transferlearnings_validate_target,
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

def seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

def run_sweep(args: Namespace, logger):
    import traceback

    ATOM_NODES_MULTIPLIKATOR = 5
    ORIGINAL_LEARNINGRATE = args.learning_rate
    def sweep_train():
        sweep_args = Namespace(**vars(args))
        run = wandb.init(mode=sweep_args.wandb_mode, settings=wandb.Settings(start_method="fork"))
        run_dir = run.dir[:-6]

        try:
            config = wandb.config
            wandb_name = ""
            for param in config.keys():
                if config[param] == "None":
                    param_value = None
                else:
                    param_value = config[param]
                    wandb_name = wandb_name + str(param)[:5] + str(param_value)

                if param == "max_num_nodes" and param_value is None and "aggregation_method" in config and config[
                    "aggregation_method"] == "fixed_size":
                    continue  # ignore since it is manually overwritten below
                if param == "node_type" and "pretrained_model" in config and config[
                    "pretrained_model"] in enforced_node_type:
                    continue  # ignore since it is manually overwritten below
                if param == "transfer_learning_datasets" and isinstance(param_value, str):
                    if ";" in param_value:
                        sweep_args.__dict__[param] = param_value.split(";")
                    else:
                        sweep_args.__dict__[param] = [param_value]
                    continue

                sweep_args.__dict__[param] = param_value
                if param == "pretrained_model":
                    if config[param] in enforced_node_type:
                        sweep_args.__dict__["node_type"] = enforced_node_type[config[param]]
                        config["node_type"] = enforced_node_type[config[param]]

                if param == 'aggregation_method':
                    if param_value == "fixed_size" and "max_num_nodes" in config and config["max_num_nodes"] == "None":
                        max_num_nodes_values = [int(value) for value in
                                                sweep_args.config["HYPERPARAMETER_SEARCH"]["parameters"]["max_num_nodes"][
                                                    "values"] if value != "None"]
                        sweep_args.max_num_nodes = random.choice(max_num_nodes_values)
                        config["max_num_nodes"] = sweep_args.max_num_nodes

            # adapt hyperparameter based on node type
            if sweep_args.node_type == "atom" and sweep_args.max_num_nodes is not None:
                sweep_args.max_num_nodes = int(sweep_args.max_num_nodes * ATOM_NODES_MULTIPLIKATOR)

            # adapt batch size bases on node type
            if sweep_args.node_type == "atom":
                sweep_args.batch_size = int(sweep_args.batch_size / ATOM_NODES_MULTIPLIKATOR) + 1

            if "learning_rate" not in config.keys():
                # reset learning rate to original value
                sweep_args.learning_rate = ORIGINAL_LEARNINGRATE

            if sweep_args.pretrained_model in ["IPA", "Diffusion"]:
                logging.warning(
                    "Forcing batch_size to 1 for IPA model (learning-rate is reduced proportionally). Also forcing GNN type to 'identity' and fine_tuning.")
                sweep_args.__dict__[
                    "gnn_type"] = "identity"  # we could also test combination of IPA and GNN, but it adds combplexity
                # Adjusting learning rate for the reduced batch size
                sweep_args.__dict__["learning_rate"] = sweep_args.__dict__["learning_rate"] / sweep_args.__dict__["batch_size"]
                sweep_args.__dict__["batch_size"] = 1

                # Enforce fine-tuning
                if not sweep_args.__dict__["fine_tune"]:
                    sweep_args.__dict__["fine_tune"] = True
                    sweep_args.__dict__["max_epochs"] = sweep_args.__dict__["max_epochs"] / 2  # account for the duplication of epochs

            sweep_args.tqdm_output = False  # disable tqdm output to reduce log syncing
            logger.info(f"Performing {sweep_args.train_strategy}")
            sweep_args.wandb_name = wandb_name
            seed(sweep_args.seed)
            model, results, wandb_inst = training[sweep_args.train_strategy](sweep_args)
            run_and_log_benchmarks(model, sweep_args)
            wandb.finish(0)
        except Exception as e:
            # log errors before finishing job
            logger.error(e)
            logger.error(traceback.print_exc())
            wandb.finish(-1)

        if sweep_args.wandb_mode == "offline":
            command = f'wandb sync --id {run.id} {run_dir}'
            subprocess.run(command, shell=True)
            time.sleep(10)

    logger.info(f"Starting {args.sweep_runs} runs in this instance")
    wandb.agent(args.sweep_id, function=sweep_train, count=args.sweep_runs, project="abag_binding_affinity")


def start_debugger():
    import debugpy
    for port in range(5678, 5689):
        try:
            debugpy.listen(("0.0.0.0", port))
            print(f"Debugger listening on port {port}")
            # debugpy.wait_for_client()
            break
        except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
            pass
    else:
        logging.warning("No free port found for debugger")

def main() -> Dict:
    """ Main functionality of the abag_affinity module

    Provides train functionalities based on provided CLI arguments

    Returns:
        Dict: Results of training
    """
    args = parse_args()
    logger = logging_setup(args)
    if args.debug:
        start_debugger()

    from guided_protein_diffusion.utils.interact import init_interactive_environment
    init_interactive_environment(
        ["--dataset", "abdb", "--openfold_time_injection_alpha", "0.0", "--antigen_conditioning"]
    )  # implies --testing

    if args.init_sweep:
        if args.sweep_config:
            logging.info(f"Using sweep config from dedicated file {args.sweep_config}")
            with open(args.sweep_config, "r") as f:
                sweep_configuration = yaml.safe_load(f)
        else:
            logging.info("Using sweep config from config.yaml")
            sweep_configuration = args.config["HYPERPARAMETER_SEARCH"]
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='abag_binding_affinity')
        args.sweep_id = sweep_id
        logger.info(f"W&B Sweep initialized with ID: {args.sweep_id}")

    if args.sweep_id is not None and args.sweep_runs != 0:
        run_sweep(args, logger)
    else:
        logger.info(f"Performing {args.train_strategy}")

        seed(args.seed)
        if args.cross_validation:
            model, results = cross_validation(args)
        else:
            model, results, wandb_inst = training[args.train_strategy](args)
            run_and_log_benchmarks(model, args)

            # Save model
            if args.model_path is not None:
                path = Path(args.model_path)
            else:
                path = Path(args.config["model_path"]) / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.wandb_name.replace(" ", "")) / "model.pt"
            path.parent.mkdir(parents=True, exist_ok=True)

            # Minor hack to exploit PyTorch Lightnings model+argument-saving mechanism
            trainer = pl.Trainer()
            trainer.fit(model, DataLoader([]))
            trainer.save_checkpoint(path)
            # TODO make sure (when loading) that the model is initialized with the same seed. <- why did I write this comment? If no-one finds a reason, delete the comment
        # return results  (leads to error code in bash)


if __name__ == "__main__":
    main()
