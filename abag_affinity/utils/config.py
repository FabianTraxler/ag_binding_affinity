import yaml
import os


def read_yaml(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_data_paths(config: dict, dataset: str):
    path = os.path.join(config["DATA"]["path"], config["DATA"][dataset]["folder_path"])
    summary = os.path.join(path, config["DATA"][dataset]["summary"])
    pdb_path = os.path.join(path, config["DATA"][dataset]["pdb_path"])

    return summary, pdb_path