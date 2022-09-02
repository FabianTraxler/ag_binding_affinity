"""This module provides all training utilities for the model streams in table_header_standardization"""
import argparse
from datetime import datetime
import logging
import sys

from abag_affinity.train.training_strategies import model_train, cross_validation, pretrain_model, bucket_train
from abag_affinity.utils.config import read_yaml


training = {
    "bucket_train": bucket_train,
    "pretrain_model": pretrain_model,
    "model_train": model_train,
    "cross_validation": cross_validation
}

def logging_setup(args):
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING

    main_logger = logging.getLogger("abag_affinity")
    main_logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    main_logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description='CLI for using the abag_affinity module')
    parser.add_argument("-t", "--train_strategy", type=str, help='The training strategy to use', default="model_train",
                        choices=["bucket_train", "pretrain_model", "model_train", "cross_validation"])
    parser.add_argument("-m", "--model_type", type=str, help='The type of model to train', default="GraphConv",
                        choices=["GraphConv", "GraphAttention", "FixedSizeGraphConv", "DDGBackboneFC", "KpGNN"])
    parser.add_argument("-d", "--data_type", type=str, help='The type of dataset (graphs) to use for training',
                        default="SimpleGraphs", choices=["SimpleGraphs", "FixedSizeGraphs", "DDGBackboneInputs", "HeteroGraphs", "InterfaceGraphs"])
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size used for training", default=1)
    parser.add_argument("-e", "--max_epochs", type=int, help="Max number of training epochs", default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Initial learning rate", default=1e-4)
    parser.add_argument("-p", "--patience", type=int, help="Number of epochs with no improvement until end of training",
                        default=10)
    parser.add_argument("-n", "--node_type", type=str, help="Type of nodes in the graphs", default="residue",
                        choices=["residue", "atom"])
    parser.add_argument("--max_num_nodes", type=int, help="Maximal number of nodes for fixed sized graphs", default=50)
    parser.add_argument("-w", "--num_workers", type=int, help="Number of workers to use for data loading", default=0)
    parser.add_argument("-wdb", "--use_wandb", type=bool, help="Use Weight&Bias to log training process", default=False)
    parser.add_argument("--wandb_name", type=str, help="Name of the Weight&Bias logs", default=str(datetime.utcnow()))
    parser.add_argument("-v", "--validation_set", type=int, help="Which validation set to use", default=1,
                        choices=[1,2,3])
    parser.add_argument("-c", "--config_file", type=str, help="Path to config file for datasets and training strategies",
                        default="../config.yaml")
    parser.add_argument("--verbose", type=bool, help="Print verbose logging statements",
                        default=False)

    args = parser.parse_args()
    args.config = read_yaml(args.config_file)

    logging_setup(args)

    results = training[args.train_strategy](args)



if __name__ == "__main__":
    main()


