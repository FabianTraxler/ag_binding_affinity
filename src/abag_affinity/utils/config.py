"""Utilities to read config file and extract relevant paths"""
import yaml
import os
from typing import Dict, Tuple, List


def read_yaml(file_path: str) -> Dict:
    """ Read a yaml file, join paths and return content as dict

    Args:
        file_path: Path to file

    Returns:
        Dict: Modified content of yaml file
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    folder_path = config["PROJECT_ROOT"]
    if folder_path is not None:
        config["RESOURCES"]["path"] = os.path.join(folder_path, config["RESOURCES"]["path"])
        config["DATA"]["path"] = os.path.join(folder_path, config["DATA"]["path"])

    config["plot_path"] = os.path.join(folder_path, config["RESULTS"]["path"], config["RESULTS"]["plot_path"])

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
    path = os.path.join(config["DATA"]["path"], config["DATA"][dataset]["folder_path"])
    if "summary" in config["DATA"][dataset]:
        summary = os.path.join(path, config["DATA"][dataset]["summary"])
    else:
        summary = ""
    if "pdb_path" in config["DATA"][dataset]:
        pdb_paths = [os.path.join(path, config["DATA"][dataset]["pdb_path"])]
    elif "pdb_paths" in config["DATA"][dataset]:
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
