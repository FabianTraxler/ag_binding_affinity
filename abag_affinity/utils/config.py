import yaml
import os


def read_yaml(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    folder_path = os.getenv('ABAG_PATH')
    if folder_path is not None:
        config["RESOURCES"]["path"] = os.path.join(folder_path, config["RESOURCES"]["path"])
        config["DATA"]["path"] = os.path.join(folder_path, config["DATA"]["path"])

    return config


def get_data_paths(config: dict, dataset: str):
    path = os.path.join(config["DATA"]["path"], config["DATA"][dataset]["folder_path"])
    summary = os.path.join(path, config["DATA"][dataset]["summary"])
    if "pdb_path" in config["DATA"][dataset]:
        pdb_paths = [os.path.join(path, config["DATA"][dataset]["pdb_path"])]
    elif "pdb_paths" in config["DATA"][dataset]:
        pdb_paths = [os.path.join(path, folder) for folder in config["DATA"][dataset]["pdb_paths"]]
    else:
        pdb_paths = []

    return summary, pdb_paths


def get_resources_paths(config: dict, dataset: str):
    path = os.path.join(config["RESOURCES"]["path"], config["RESOURCES"][dataset]["folder_path"])
    summary = os.path.join(path, config["RESOURCES"][dataset]["summary"])
    pdb_path = os.path.join(path, config["RESOURCES"][dataset]["pdb_path"])

    return summary, pdb_path
