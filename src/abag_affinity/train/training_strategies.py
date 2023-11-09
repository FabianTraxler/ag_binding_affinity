"""Module providing implementations of different training modalities"""
import logging
import os
from pathlib import Path
import pandas as pd
import torch
from argparse import Namespace
import numpy as np
from typing import Dict, Tuple, Optional
import random
from collections import Counter

from ..utils.config import read_config, get_data_paths
from .utils import get_skempi_corr, load_model, bucket_learning, get_benchmark_score, \
    get_abag_test_score
from ..dataset.advanced_data_utils import load_datasets
from ..model.gnn_model import AffinityGNN



torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system') # cluster mulitple dataloader

logger = logging.getLogger(__name__) # setup module logger

def bucket_train(args:Namespace) -> Tuple[AffinityGNN, Dict]:
    """ Train on multiple datasets in parallel but downsample larger datasets to the smallest in ever epoch

    Args:
        args: CLI arguments

    Returns:
        Tuple: Trained model and Dict with results and statistics of training
    """
    config = read_config(args.config_file)

    dataset_names = args.transfer_learning_datasets + [args.target_dataset]

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #### Preparing Dataset ######
    train_datasets = []
    val_datasets = []

    double_dataset = set() # check if dataset occours in multiple datalaoders (eg. relative and absolute)
    dataset_counter = Counter([dataset_name.split("#")[0] for dataset_name in dataset_names])
    for name, count in dataset_counter.items():
        if count > 1:
            double_dataset.add(name)

    neglogkd_datasets = []
    for dataset_name in dataset_names:
        train_data, val_datas = load_datasets(config, dataset_name, args.validation_set, args)

        if args.add_neglogkd_labels_dataset and train_data.affinity_type == "E":  # exclude phillips21
            neglogkd_data, _ = load_datasets(config, dataset_name, args.validation_set, args=args,
                                             validation_size=0, only_neglogkd_samples=True)
            if len(neglogkd_data) > 0:
                neglogkd_datasets.append(neglogkd_data)

        data_name, loss_type = dataset_name.split("#")
        if not train_data.relative_data and data_name in double_dataset:
            # set the force_recomputation to False for absolute dataset because relative loader already preprocesses graphs
            train_data.force_recomputation = False
            for val_data in val_datas:
                val_data.force_recomputation = False

        if len(train_data) > 0:
            train_datasets.append(train_data)

        for val_data in val_datas:
            if len(val_data) > 0:
                val_datasets.append(val_data)

    if args.add_neglogkd_labels_dataset:
        concat_neglogkd_datasets = True
        if concat_neglogkd_datasets:
            neglogkd_dataset = torch.utils.data.ConcatDataset(neglogkd_datasets)
            neglogkd_dataset.full_dataset_name = "DMS-neglogkd-mixed"
            neglogkd_dataset.dataset_name = "DMS"
            neglogkd_dataset.relative_data = False
            neglogkd_dataset.loss_criterion = args.add_neglogkd_labels_dataset
            train_datasets.append(neglogkd_dataset)
        else:
            train_datasets.extend(neglogkd_datasets)

    #### Preparing Model ######

    model = load_model(train_datasets[0].num_features, train_datasets[0].num_edge_features, dataset_names, args, device)
    logger.debug(f"Training done on GPU = {next(model.parameters()).is_cuda}")

    logger.info("Training with {}".format(", ".join([dataset.full_dataset_name for dataset in train_datasets])))
    logger.info("Evaluating on {}".format(", ".join([dataset.full_dataset_name for dataset in val_datasets])))
    
    #### Actual Training ######
    results, model, wandb_inst = bucket_learning(model, train_datasets, val_datasets, args)

    logger.info("Training with {} completed".format(dataset_names))

    logger.debug(results)
    return model, results, wandb_inst


def train_transferlearnings_validate_target(args: Namespace):
    """
    Train on the transfer learning datasets, validate on the target dataset, making sure that the target dataset is excluded from the training

    This function is dedicated to the DMS cross-validation and slightly abuses the CLI argument names
    """
    config = read_config(args.config_file)

    # make sure that the loss functions are not in the way
    if args.target_dataset.split("#")[1] != "L2":
        logging.warning(f"target_dataset was set to ({args.target_dataset}), but L2/RMSE loss will be enforced in the validation")

    training_set_names = [dataset for dataset in args.transfer_learning_datasets if dataset.split("#")[0] != args.target_dataset.split("#")[0]]

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_datasets = []
    val_datasets = []

    for training_set_name in training_set_names:
        train_data, val_datas = load_datasets(config, training_set_name, args.validation_set, args, 0.0)  # For simplicity, we only load training data. (makes the DMS cross-validation sweep simpler)

        if len(train_data) > 0:
            train_datasets.append(train_data)

    # Load the main validation dataset
    target_train_spikein_set, target_val_datas = load_datasets(config, args.target_dataset, args.validation_set, args, 1.0-args.training_set_spikein)
    if len(target_train_spikein_set) > 0:
        train_datasets.append(target_train_spikein_set)

    for val_data in target_val_datas:
        if len(val_data) > 0:
            val_datasets.append(val_data)

    model = load_model(train_datasets[0].num_features, train_datasets[0].num_edge_features, training_set_names + [args.target_dataset], args, device)
    logger.debug(f"Training done on GPU = {next(model.parameters()).is_cuda}")

    logger.info("Training with {}".format(", ".join([dataset.full_dataset_name for dataset in train_datasets])))
    logger.info("Evaluating on {}".format(", ".join([dataset.full_dataset_name for dataset in val_datasets])))
    results, model, wandb_inst = bucket_learning(model, train_datasets, val_datasets, args)
    logger.info("Training with {} completed".format(training_set_names))

    logger.debug(results)
    return model, results, wandb_inst


def cross_validation(args:Namespace) -> Tuple[None, Dict]:
    """ Perform a Cross Validation based on predefined splits of the data

    Args:
        args: CLI arguments

    Returns:
        Tuple: None and the results and statistics of training
    """


    experiment_name = "CV_experiment_transfer_learning"

    Path(args.config["model_path"]).mkdir(exist_ok=True, parents=True)
    losses = []
    correlations = []
    spearman_correlations = []
    all_results = {}
    benchmark_losses = []
    benchmark_correlation = []
    benchmark_spearman_correlation = []
    test_losses = []
    test_correlation = []
    test_spearman_correlation = []
    skempi_losses = []
    skempi_correlation = []
    skempi_spearman_correlation = []
    skempi_grouped_correlation = []
    skempi_grouped_spearman_correlation = []

    training = {
        "bucket_train": bucket_train,
    }

    #args.max_epochs = 1
    summary_path, _ = get_data_paths(args.config, "abag_affinity")
    summary_df = pd.read_csv(summary_path, index_col=0)
    n_splits = summary_df["validation"].max() + 1

    Path(os.path.join(args.config["prediction_path"], experiment_name)).mkdir(exist_ok=True, parents=True)

    for i in range(1, n_splits):
        logger.info("\nValidation on split {} and training with all other splits".format(i))
        args.validation_set = i
        best_model, results, wandb_inst = training[args.train_strategy](args)
        torch.save(best_model.state_dict(), os.path.join(args.config["model_path"], f"best_model_val_set_{i}.pt"))

        if args.target_dataset in results:
            losses.append(results[args.target_dataset]["best_loss"])
            correlations.append(results[args.target_dataset]["best_correlation"])
            spearman_correlations.append(results[args.target_dataset]["best_spearman_correlation"])
            logger.info(f"Pearson for split {i} >>> {results[args.target_dataset]['best_correlation']}")
            logger.info(f"Spearman for split {i} >>> {results[args.target_dataset]['best_spearman_correlation']}")
        else:
            losses.append(results["best_loss"])
            correlations.append(results["best_correlation"])
            spearman_correlations.append(results["best_spearman_correlation"])
            logger.info(f"Pearson for split {i} >>> {results['best_correlation']}")
            logger.info(f"Spearman for split {i} >>> {results['best_spearman_correlation']}")

        all_results[i] = results

        # Benchmark results
        benchmark_plot_path = os.path.join(args.config["plot_path"], experiment_name, f"benchmark_cv{args.validation_set}.png")
        benchmark_pearson, benchmark_spearman, benchmark_loss, benchmark_rmse, benchmark_df = get_benchmark_score(best_model, args, tqdm_output=args.tqdm_output, plot_path=benchmark_plot_path)

        benchmark_df.to_csv(os.path.join(args.config["prediction_path"], experiment_name, f"benchmark_cv{args.validation_set}.csv"))

        benchmark_losses.append(benchmark_loss)
        benchmark_correlation.append(benchmark_pearson)
        benchmark_spearman_correlation.append(benchmark_spearman)

        logger.info(f"Benchmark pearson >>> {benchmark_pearson}")
        logger.info(f"Benchmark spearman >>> {benchmark_spearman}")

        # SKEMPI results
        skempi_test_plot_path = os.path.join(args.config["plot_path"], experiment_name,
                                            f"skempi_score_test_cv{args.validation_set}.png")
        test_skempi_grouped_corrs, test_skempi_grouped_spearman_corrs, test_skempi_score, test_skempi_spearman_score, test_loss_skempi, rmse_skempi, test_skempi_df = get_skempi_corr(best_model, args, tqdm_output=args.tqdm_output,
                                                              plot_path=skempi_test_plot_path)
        test_skempi_df.to_csv(os.path.join(args.config["prediction_path"], experiment_name, f"skempi_score_test_cv{args.validation_set}.csv"))

        skempi_grouped_correlation.append(test_skempi_grouped_corrs)
        skempi_grouped_spearman_correlation.append(test_skempi_grouped_spearman_corrs)
        skempi_correlation.append(test_skempi_score)
        skempi_spearman_correlation.append(test_skempi_spearman_score)
        skempi_losses.append(test_loss_skempi)
        logger.info(f"SKEMPI testset results >>> {test_skempi_score}")

        # ABAG Test set results
        abag_test_plot_path = os.path.join(args.config["plot_path"], experiment_name,
                                           f"abag_affinity_test_cv{args.validation_set}.png")
        test_pearson, test_spearman, test_loss, test_rmse, test_df = get_abag_test_score(best_model, args, tqdm_output=args.tqdm_output,
                                                      plot_path=abag_test_plot_path,
                                                      validation_set=i)
        test_df.to_csv(os.path.join(args.config["prediction_path"], experiment_name, f"abag_affinity_test_cv{args.validation_set}.csv"))
        test_losses.append(test_loss)
        test_correlation.append(test_pearson)
        test_spearman_correlation.append(test_spearman)
        logger.info(f"AbAg-Affinity testset results >>> {test_pearson}")

        wandb_benchmark_log = {"abag_test_pearson": test_pearson, "abag_test_spearman": test_spearman, "abag_test_loss": test_loss, "abag_rmse": test_rmse,
                               "skempi_test_pearson": test_skempi_score, "skempi_test_spearman": test_skempi_spearman_score, "skempi_test_loss": test_loss_skempi, "skempi_rmse": rmse_skempi,
                               "benchmark_test_pearson": benchmark_pearson, "benchmark_test_spearman": benchmark_spearman, "benchmark_test_loss": benchmark_loss, "benchmark_rmse": benchmark_rmse}
        wandb_inst.log(wandb_benchmark_log, commit=True)

    # TODO need to add logging for spearman in case we use this function
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

