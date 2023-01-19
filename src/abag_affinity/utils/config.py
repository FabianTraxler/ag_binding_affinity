"""Utilities to read config file and extract relevant paths"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def read_config(file_path: str, use_relaxed: bool = False) -> Dict:
    """ Read a yaml file, join paths and return content as dict

    Args:
        file_path: Path to file
        use_relaxed: Boolean indicator if relaxed pdb should be used

    Returns:
        Dict: Modified content of yaml file
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)


    folder_path = (Path(__file__).parents[3]).resolve()
    config['PROJECT_ROOT'] = folder_path

    if folder_path is not None:
        config["DATASETS"]["path"] = os.path.join(folder_path, config["DATASETS"]["path"])
        config["RESOURCES"]["path"] = os.path.join(folder_path, config["RESOURCES"]["path"])

    config["plot_path"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["plot_path"])
    config["prediction_path"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["prediction_path"])
    config["model_path"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["model_path"])
    config["processed_graph_path"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["processed_graph_path"])
    config["cleaned_pdbs"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["cleaned_pdbs"])
    config["force_field_results"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["force_field_results"])

    for model in config["MODELS"].keys():
        config["MODELS"][model]["model_path"] = os.path.join(folder_path, config["MODELS"][model]["model_path"])

    if use_relaxed:
        for dataset in config["DATASETS"].keys():
            if "relaxed_pdb_path" in config["DATASETS"][dataset]:
                config["DATASETS"][dataset]["pdb_path"] = config["DATASETS"][dataset]["relaxed_pdb_path"]
            elif "relaxed_mutated_pdb_path" in config["DATASETS"][dataset]:
                config["DATASETS"][dataset]["mutated_pdb_path"] = config["DATASETS"][dataset]["relaxed_mutated_pdb_path"]

        config["cleaned_pdbs"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["cleaned_pdbs"], "relaxed")

    return config


def get_data_paths(config: dict, dataset: str) -> Tuple[str, List[str]]:
    """ Get the path to the meta-data file and to the PDB Folders

    Args:
        config: Config dict
        dataset: Name of the dataset to load

    Returns:
        str: Path to summary file
        List: Paths to the pdb folders
    """
    path = os.path.join(config["DATASETS"]["path"], config["DATASETS"][dataset]["folder_path"])
    if "summary" in config["DATASETS"][dataset]:
        summary = os.path.join(path, config["DATASETS"][dataset]["summary"])
    else:
        summary = ""
    if "pdb_path" in config["DATASETS"][dataset]:
        pdb_paths = [os.path.join(path, config["DATASETS"][dataset]["pdb_path"])]
    elif "pdb_paths" in config["DATASETS"][dataset]:
        pdb_paths = [os.path.join(path, folder) for folder in config["DATA"][dataset]["pdb_paths"]]
    else:
        pdb_paths = []

    return summary, pdb_paths


def get_resources_paths(config: dict, dataset: str) -> Tuple[str, List[str]]:
    """ Get the path to the meta-data files and to the PDB Folder

    Args:
        config: Config dict
        dataset: Name of the dataset to load

    Returns:
        str: Path to summary file
        List: Paths to the pdb folders
    """
    path = os.path.join(config["RESOURCES"]["path"], config["RESOURCES"][dataset]["folder_path"])
    if "summaries" in config["RESOURCES"][dataset]:
        summary = [os.path.join(path, folder) for folder in config["RESOURCES"][dataset]["summaries"]]
    elif "summary" in config["RESOURCES"][dataset]:
        summary = os.path.join(path, config["RESOURCES"][dataset]["summary"])
    else:
        summary = ""
    pdb_path = os.path.join(path, config["RESOURCES"][dataset]["pdb_path"])

    return summary, pdb_path
