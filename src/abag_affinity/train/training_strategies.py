"""Module providing implementations of different training modalities"""

import logging
import torch
from argparse import Namespace
import numpy as np
from typing import Dict, Tuple
import random

from abag_affinity.utils.config import read_config
from abag_affinity.train.utils import load_model, load_datasets, train_loop, finetune_backbone, bucket_learning

random.seed(123)

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system') # cluster mulitple dataloader

logger = logging.getLogger(__name__) # setup module logger


def model_train(args:Namespace, validation_set: int = None) -> Tuple[torch.nn.Module, Dict]:
    """ Model training functionality

    1. Get random initialized model
    2. Get dataloader
    3. Train model
    4. Return results

    Args:
        args: CLI arguments
        validation_set: Validation set to use

    Returns:
        Tuple: Trained model and Dict with results and statistics of training
    """

    config = read_config(args.config_file)

    dataset_name = config["TRAIN"]["standard"]["dataset"]

    if validation_set is None:
        validation_set = args.validation_set

    train_data, val_data = load_datasets(config, dataset_name, args.data_type, validation_set, args)

    logger.info("Val Set:{} | Train Size:{} | Test Size: {}".format(str(validation_set), len(train_data), len(val_data)))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(args.model_type, train_data.num_features, train_data.num_edge_features, args, device)

    results, model = train_loop(model, train_data, val_data, args)

    if args.model_type in ["DDGBackboneFC", "DeepRefineBackbone"]:
        results, model = finetune_backbone(model, train_data, val_data)
    return model, results


def cross_validation(args:Namespace) -> Tuple[None, Dict]:
    """ Perform a Cross Validation based on predefined splits of the datas

    Args:
        args: CLI arguments

    Returns:
        Tuple: None and the results and statistics of training
    """
    losses = []
    correlations = []
    all_results = {}

    for i in range(1, 4):
        logger.info("Validation on split {} and training with all other splits\n".format(i))
        results = model_train(args, validation_set=i)
        losses.append(results["best_loss"])
        correlations.append(results["best_correlation"])
        all_results[i] = results

    logger.info("Average Loss: {} ({})".format(np.mean(losses), np.std(losses)))
    logger.info("Average Pearson Correlation: {} ({})".format(np.mean(correlations), np.std(correlations)))

    return None, all_results


def pretrain_model(args:Namespace) -> Tuple[torch.nn.Module, Dict]:
    """ Pretrain model and then finetune model based on information in config file

    Args:
        args: CLI arguments

    Returns:
        Tuple: Trained model and Dict with results and statistics of training
    """
    config = read_config(args.config_file)

    datasets = config["TRAIN"]["transfer"]["datasets"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = None

    all_results = {}

    logger.debug("Loading Datasets")

    for dataset_name in datasets:
        logger.info("Training with {} starting ...".format(dataset_name))
        logger.debug(f"Loading  {dataset_name}")
        train_data, val_data = load_datasets(config, dataset_name, args.data_type, args.validation_set, args)

        if model is None: # only load model for first dataset
            logger.debug(f"Loading  {args.model_type}")
            model = load_model(args.model_type, train_data.num_features, train_data.num_edge_features, args, device)
            logger.debug(f"Model Memory usage: {torch.cuda.max_memory_allocated()}")
        logger.debug(f"Training with  {dataset_name}")
        logger.debug(f"Training done on GPU: {next(model.parameters()).is_cuda}")
        results, model = train_loop(model, train_data, val_data, args)

        logger.info("Training with {} completed".format(dataset_name))
        logger.debug(results)
        all_results[dataset_name] = results

    if args.model_type == "DDGBackboneFC" and datasets[-1] == "Dataset_v1":
        train_data, val_data = load_datasets(config, datasets[-1], args.data_type, args.validation_set, args)
        model, results = finetune_backbone(model, train_data, val_data, args)
        all_results["finetuning"] = results

    return model, all_results


def bucket_train(args:Namespace) -> Tuple[torch.nn.Module, Dict]:
    """ Train on multiple datasets in parallel but downsample larger datasets to the smallest in ever epoch

    Args:
        args: CLI arguments

    Returns:
        Tuple: Trained model and Dict with results and statistics of training
    """
    config = read_config(args.config_file)

    datasets = config["TRAIN"]["bucket"]["datasets"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_datasets = []
    val_datasets = []

    for dataset_name in datasets:
        train_data, val_data = load_datasets(config, dataset_name, args.data_type, args.validation_set, args)
        train_datasets.append(train_data)
        val_datasets.append(val_data)

    model = load_model(args.model_type, train_datasets[0].num_features, train_datasets[0].num_edge_features, args, device)
    logger.debug(f"Training done on GPU = {next(model.parameters()).is_cuda}")

    logger.info("Training with {} starting ...".format(datasets))
    results, model = bucket_learning(model, train_datasets, val_datasets, args)
    logger.info("Training with {} completed".format(datasets))

    logger.debug(results)
    return model, results
