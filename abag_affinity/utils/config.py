import yaml
import os


def read_yaml(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    folder_path = os.getenv('ABAG_PATH')
    if folder_path is not None:
        config["DATA"]["path"] = os.path.join(folder_path, config["DATA"]["path"])

    return config


def get_data_paths(config: dict, dataset: str):
    path = os.path.join(config["DATA"]["path"], config["DATA"][dataset]["folder_path"])
    summary = os.path.join(path, config["DATA"][dataset]["summary"])
    pdb_path = os.path.join(path, config["DATA"][dataset]["pdb_path"])

    return summary, pdb_path
