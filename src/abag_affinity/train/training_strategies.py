import logging
import torch
from argparse import Namespace
import numpy as np

from abag_affinity.utils.config import read_yaml
from abag_affinity.train.utils import load_model, load_datasets, train_loop, finetune_backbone, bucket_learning

#torch.cuda.empty_cache()
#print(torch.cuda.is_available())

logger = logging.getLogger(__name__)


def model_train(args:Namespace, validation_set: int = None):
    """
    Instantiate model training
    """

    config = read_yaml(args.config_file)

    dataset_name = config["TRAIN"]["standard"]["dataset"]

    if validation_set is None:
        validation_set = args.validation_set

    train_data, val_data = load_datasets(config, dataset_name, args.data_type, validation_set, args)

    logger.info("Val Set:", str(validation_set), " | Train Size:", len(train_data), " | Test Size:", len(val_data))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(args.model_type, train_data.num_features, args, device)

    results, model = train_loop(model, train_data, val_data, args)

    if args.model_type == "DDGBackboneFC":
        results, model = finetune_backbone(model, train_data, val_data)
    return results


def cross_validation(args:Namespace):
    losses = []
    correlations = []
    for i in range(1, 4):
        logger.info("Validation on split {} and training with all other splits\n".format(i))
        results = model_train(args, validation_set=i)
        losses.append(results["best_loss"])
        correlations.append(results["best_correlation"])

    logger.info("Average Loss: {} ({})".format(np.mean(losses), np.std(losses)))
    logger.info("Average Pearson Correlation: {} ({})".format(np.mean(correlations), np.std(correlations)))


def pretrain_model(args:Namespace):
    import random
    random.seed(123)
    config = read_yaml(args.config_file)

    datasets = config["TRAIN"]["transfer"]["datasets"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = None

    logger.debug("Loading Datasets")

    for dataset_name in datasets:
        logger.debug(f"Loading  {dataset_name}")
        train_data, val_data = load_datasets(config, dataset_name, args.data_type, args.validation_set, args)

        if model is None:
            logger.debug(f"Loading  {args.model_type}")
            model = load_model(args.model_type, train_data.num_features, args, device)
        logger.debug(f"Training with  {dataset_name}")
        logger.debug(f"Training done on GPU: {next(model.parameters()).is_cuda}")
        results, model = train_loop(model, train_data, val_data, args)

        logger.info("Training with {} completed".format(dataset_name))
        logger.debug(results)

    if args.model_type == "DDGBackboneFC" and datasets[-1] == "Dataset_v1":
        finetune_backbone(model, train_data, val_data)

    return results


def bucket_train(args:Namespace):
    import os
    import random
    random.seed(123)
    config = read_yaml(args.config_file)

    datasets = config["TRAIN"]["bucket"]["datasets"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_datasets = []
    val_datasets = []

    for dataset_name in datasets:
        train_data, val_data = load_datasets(config, dataset_name, args.data_type, args.validation_set, args)
        train_datasets.append(train_data)
        val_datasets.append(val_data)

    model = load_model(args.model_type, train_datasets[0].num_features, args, device)
    logger.debug(f"Training done on GPU = {next(model.parameters()).is_cuda}")

    results, model = bucket_learning(model, train_datasets, val_datasets, args)

    logger.info("Training with {} completed".format(datasets))
    logger.info(results)
    return results
