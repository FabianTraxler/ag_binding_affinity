"""Module providing implementations of different training modalities"""
import logging
import os
from pathlib import Path
import torch
from argparse import Namespace
import numpy as np
from typing import Dict, Tuple
import random
from collections import Counter

from ..utils.config import read_config, get_data_paths
from .utils import get_skempi_corr, load_model, load_datasets, train_loop, finetune_pretrained, bucket_learning, get_benchmark_score, \
    get_abag_test_score
from ..model.gnn_model import AffinityGNN

# TODO: create global seeding mechanism
random.seed(125)
np.random.seed(125)
torch.manual_seed(125)


torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system') # cluster mulitple dataloader

logger = logging.getLogger(__name__) # setup module logger


def model_train(args:Namespace, validation_set: int = None) -> Tuple[AffinityGNN, Dict]:
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

    dataset_name = args.target_dataset

    if validation_set is None:
        validation_set = args.validation_set

    train_data, val_data = load_datasets(args.config, dataset_name, validation_set, args)

    logger.info("Val Set:{} | Train Size:{} | Test Size: {}".format(str(validation_set), len(train_data), len(val_data)))

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(train_data.num_features, train_data.num_edge_features, args, device)

    logger.debug(f"Training with  {dataset_name}")
    logger.debug(f"Training done on GPU: {next(model.parameters()).is_cuda}")

    results, best_model = train_loop(model, train_data, val_data, args)

    if args.pretrained_model in ["Binding_DDG", "DeepRefine", "IPA", "Diffusion"]:
        results, best_model = finetune_pretrained(best_model, train_data, val_data, args)
    return best_model, results


def pretrain_model(args:Namespace) -> Tuple[AffinityGNN, Dict]:
    """ Pretrain model and then finetune model based on information in config file

    Args:
        args: CLI arguments

    Returns:
        Tuple: Trained model and Dict with results and statistics of training
    """
    config = read_config(args.config_file)

    datasets = args.transfer_learning_datasets + [args.target_dataset]

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = None

    all_results = {}

    logger.debug("Loading Datasets")

    for dataset_name in datasets:
        logger.info("Training with {} starting ...".format(dataset_name))
        logger.debug(f"Loading  {dataset_name}")
        train_data, val_data = load_datasets(config, dataset_name, args.validation_set, args)

        if model is None: # only load model for first dataset
            logger.debug(f"Loading  Model")
            model = load_model(train_data.num_features, train_data.num_edge_features, args, device)
            logger.debug(f"Model Memory usage: {torch.cuda.max_memory_allocated()/(1<<20):,.0f} MB")
        logger.debug(f"Training with  {dataset_name}")
        logger.debug(f"Training done on GPU: {next(model.parameters()).is_cuda}")

        results, model = train_loop(model, train_data, val_data, args)

        logger.info("Training with {} completed".format(dataset_name))
        logger.debug(results)
        all_results[dataset_name] = results

    if args.pretrained_model in ["Binding_DDG", "DeepRefine", "IPA", "Diffusion"]:
        train_data, val_data = load_datasets(config, datasets[-1], args.validation_set, args)
        results, model = finetune_pretrained(model, train_data, val_data, args)
        all_results["finetuning"] = results

    return model, all_results


def bucket_train(args:Namespace) -> Tuple[AffinityGNN, Dict]:
    """ Train on multiple datasets in parallel but downsample larger datasets to the smallest in ever epoch

    Args:
        args: CLI arguments

    Returns:
        Tuple: Trained model and Dict with results and statistics of training
    """
    config = read_config(args.config_file)

    datasets = args.transfer_learning_datasets + [args.target_dataset]

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_datasets = []
    val_datasets = []

    double_dataset = set() # check if dataset occours in multiple datalaoders (eg. relative and absolute)
    dataset_counter = Counter([dataset_name.split(":")[0] for dataset_name in datasets])
    for name, count in dataset_counter.items():
        if count > 1:
            double_dataset.add(name)

    for dataset_type in datasets:
        train_data, val_data = load_datasets(config, dataset_type, args.validation_set, args)

        data_name, data_type = dataset_type.split(":")
        if data_type == "absolute" and data_name in double_dataset:
            # set the force_recomputation to False for absolute dataset because relative loader already preprocesses graphs
            train_data.force_recomputation = False
            val_data.force_recomputation = False

        if len(train_data) > 0:
            train_datasets.append(train_data)
        if len(val_data) > 0:
            val_datasets.append(val_data)

    model = load_model(train_datasets[0].num_features, train_datasets[0].num_edge_features, args, device)
    logger.debug(f"Training done on GPU = {next(model.parameters()).is_cuda}")

    logger.info("Training with {}".format(", ".join([dataset.dataset_name for dataset in train_datasets])))
    logger.info("Evaluating on {}".format(", ".join([dataset.dataset_name for dataset in val_datasets])))
    results, model = bucket_learning(model, train_datasets, val_datasets, args)
    logger.info("Training with {} completed".format(datasets))

    if args.pretrained_model in ["Binding_DDG", "DeepRefine", "IPA", "Diffusion"]:
        results, model = finetune_pretrained(model, train_datasets, val_datasets, args)

    logger.debug(results)
    return model, results


def cross_validation(args:Namespace) -> Tuple[None, Dict]:
    """ Perform a Cross Validation based on predefined splits of the data

    Args:
        args: CLI arguments

    Returns:
        Tuple: None and the results and statistics of training
    """
    import pandas as pd

    experiment_name = "CV_experiment_transfer_learning"

    Path(args.config["model_path"]).mkdir(exist_ok=True, parents=True)
    losses = []
    correlations = []
    all_results = {}
    benchmark_losses = []
    benchmark_correlation = []
    test_losses = []
    test_correlation = []
    skempi_losses = []
    skempi_correlation = []
    skempi_grouped_correlation = []

    training = {
        "bucket_train": bucket_train,
        "pretrain_model": pretrain_model,
        "model_train": model_train
    }

    #args.max_epochs = 1
    summary_path, _ = get_data_paths(args.config, "abag_affinity")
    summary_df = pd.read_csv(summary_path, index_col=0)
    n_splits = summary_df["validation"].max() + 1

    Path(os.path.join(args.config["prediction_path"], experiment_name)).mkdir(exist_ok=True, parents=True)

    for i in range(1, n_splits):
        logger.info("\nValidation on split {} and training with all other splits".format(i))
        args.validation_set = i
        best_model, results = training[args.train_strategy](args)
        torch.save(best_model.state_dict(), os.path.join(args.config["model_path"], f"best_model_val_set_{i}.pt"))

        if args.target_dataset in results:
            losses.append(results[args.target_dataset]["best_loss"])
            correlations.append(results[args.target_dataset]["best_correlation"])
            logger.info(f"Results for split {i} >>> {results[args.target_dataset]['best_correlation']}")
        else:
            losses.append(results["best_loss"])
            correlations.append(results["best_correlation"])
            logger.info(f"Results for split {i} >>> {results['best_correlation']}")

        all_results[i] = results

        # Benchmark results
        benchmark_plot_path = os.path.join(args.config["plot_path"], experiment_name, f"benchmark_cv{args.validation_set}.png")
        benchmark_pearson, benchmark_loss, benchmark_df = get_benchmark_score(best_model, args, tqdm_output=args.tqdm_output, plot_path=benchmark_plot_path)
        benchmark_df.to_csv(os.path.join(args.config["prediction_path"], experiment_name, f"benchmark_cv{args.validation_set}.csv"))

        benchmark_losses.append(benchmark_loss)
        benchmark_correlation.append(benchmark_pearson)
        logger.info(f"Benchmark results >>> {benchmark_pearson}")

        # SKEMPI results
        skempi_test_plot_path = os.path.join(args.config["plot_path"], experiment_name,
                                            f"skempi_score_test_cv{args.validation_set}.png")
        test_skempi_grouped_corrs, test_skempi_score, test_loss_skempi, test_skempi_df = get_skempi_corr(best_model, args, tqdm_output=args.tqdm_output,
                                                              plot_path=skempi_test_plot_path)
        test_skempi_df.to_csv(os.path.join(args.config["prediction_path"], experiment_name, f"skempi_score_test_cv{args.validation_set}.csv"))

        skempi_grouped_correlation.append(test_skempi_grouped_corrs)
        skempi_correlation.append(test_skempi_score)
        skempi_losses.append(test_loss_skempi)
        logger.info(f"SKEMPI testset results >>> {test_skempi_score}")

        # ABAG Test set results
        abag_test_plot_path = os.path.join(args.config["plot_path"], experiment_name,
                                           f"abag_affinity_test_cv{args.validation_set}.png")
        test_pearson, test_loss, test_df = get_abag_test_score(best_model, args, tqdm_output=args.tqdm_output,
                                                      plot_path=abag_test_plot_path,
                                                      validation_set=i)
        test_df.to_csv(os.path.join(args.config["prediction_path"], experiment_name, f"abag_affinity_test_cv{args.validation_set}.csv"))
        test_losses.append(test_loss)
        test_correlation.append(test_pearson)
        logger.info(f"AbAg-Affinity testset results >>> {test_pearson}")


    logger.info("Average Loss: {} ({})".format(np.mean(losses), np.std(losses)))
    logger.info("Losses: {}".format(" - ".join([ str(loss) for loss in losses ])))
    logger.info("Average Pearson Correlation: {} ({})".format(np.mean(correlations), np.std(correlations)))
    logger.info("Correlations: {}".format(" - ".join([ str(corr) for corr in correlations ])))

    logger.info("Average Pearson Correlation Benchmark: {} ({})".format(np.mean(benchmark_correlation), np.std(benchmark_correlation)))
    logger.info("Benchmark correlations: {}".format(" - ".join([ str(corr) for corr in benchmark_correlation])))

    logger.info("Average Pearson Correlation AbAg testset: {} ({})".format(np.mean(test_correlation), np.std(test_correlation)))
    logger.info("AbAg testset correlations: {}".format(" - ".join([ str(corr) for corr in test_correlation])))

    logger.info("Average PDB-grouped Pearson Correlation SKEMPI testset: {} ({})".format(np.mean(skempi_grouped_correlation), np.std(skempi_grouped_correlation)))
    logger.info("Average Pearson Correlation SKEMPI testset: {} ({})".format(np.mean(skempi_correlation), np.std(skempi_correlation)))
    logger.info("Skempi testset correlations: {}".format(" - ".join([ str(corr) for corr in skempi_correlation])))

    return None, all_results

