import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import warnings

warnings.filterwarnings("ignore")
from abag_affinity.utils.config import read_yaml, get_data_paths
from abag_affinity.utils.pdb_processing import get_distances_and_info, get_residue_encodings, get_edge_encodings, read_file


alphabet_letters = set(string.ascii_lowercase)


def generate_dataset_v1(config_path: str):
    config = read_yaml(config_path)

    summary_path, pdb_path = get_data_paths(config, "Dataset_v1")

    summary_df = pd.read_csv(summary_path)

    path = os.path.join(config["DATA"]["path"], config["DATA"]["Dataset_v1"]["folder_path"])
    output_folder = os.path.join(path, config["DATA"]["Dataset_v1"]["dataset_path"])

    summary_df.drop_duplicates('pdb', inplace=True)
    summary_df.reset_index(inplace=True)
    error_while_loading = []
    for i, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
        try:
            pdb_id = row["pdb"]

            chain_id2protein = {"l": "antibody", "h": "antibody"} # heavy and light chains of antibodies are often called l and h
            for chain_id in row["light_chains"]:
                chain_id2protein[chain_id.lower()] = "antibody"
            for chain_id in row["heavy_chains"]:
                chain_id2protein[chain_id.lower()] = "antibody"
            for chain_id in row["antigen_chains"]:
                chain_id2protein[chain_id.lower()] = "antigen"
            for letter in alphabet_letters - set(chain_id2protein.keys()):
                chain_id2protein[letter] = "antigen"

            file_path = os.path.join(path,config["DATA"]["Dataset_v1"]["pdb_path"], row["abdb_file"])


            ###### TEST ######
            from abag_affinity.binding_ddg_predictor.utils.data import load_wt_mut_pdb_pair

            test = load_wt_mut_pdb_pair(file_path, file_path)

            structure, header = read_file(pdb_id, file_path)

            distances, residue_infos, residue_atom_coordinates, structure_info, closest_residues = get_distances_and_info(structure, header, chain_id2protein)

            res_features = get_residue_encodings(residue_infos, structure_info, chain_id2protein)
            adj_tensor = get_edge_encodings(distances, residue_infos, chain_id2protein, distance_cutoff=10)

            delta_g = row["delta_g"]
            assert len(residue_infos) > 0
            assert res_features.shape[0] == len(residue_infos)
            assert adj_tensor[0, :, :].shape == (len(residue_infos), len(residue_infos))

        except Exception as e:
            print("error")
            import traceback
            aa = traceback.format_exc()
            error_while_loading.append((row["pdb"], e))
            continue

        out_file = os.path.join(output_folder, row["pdb"] + ".npz")
        np.savez_compressed(out_file, residue_features=res_features, residue_infos=residue_infos,
                            residue_atom_coordinates=residue_atom_coordinates, adjacency_tensor=adj_tensor,
                            affinity=delta_g, closest_residues=closest_residues)

    with open(os.path.join(path, "dataset_errors.txt"), "w") as f:
        for error in error_while_loading:
            f.write(error[0] + " - " + str(error[1]))
            f.write("\n")

if __name__ == "__main__":
    generate_dataset_v1("../../abag_affinity/config.yaml")