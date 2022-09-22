# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import io
import logging
import os
from typing import Dict, List

import pandas as pd
import pyrosetta
from abag_affinity.utils.config import get_data_paths, read_config
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.protocols.moves import Mover
from pyrosetta.toolbox.mutants import mutate_residue
from tqdm import tqdm

logger = logging.getLogger("Skempi-Structures-Generation")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

folder_path = os.getenv('ABAG_PATH')
log_file = logging.FileHandler(os.path.join(folder_path, "logs/data_generation_scripts.log"))
log_file.setLevel(logging.INFO)
log_file.setFormatter(formatter)
logger.addHandler(log_file)

pyrosetta.init(extra_options="-mute all")


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    pose = pyrosetta.pose_from_pdb(pdb_path)
    testPose = pyrosetta.Pose()
    testPose.assign(pose)
    return testPose


def load_relax_function_and_packer():
    from pyrosetta.rosetta.protocols.relax import FastRelax

    relax = FastRelax()
    scorefxn = pyrosetta.get_fa_scorefxn()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)

    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)

    return relax, packer


def get_mutations(summary_df: pd.DataFrame, location: int, cleaned_file: bool = True):
    row = summary_df.iloc[location]

    if cleaned_file:
        mutation_code = row["Mutation(s)_cleaned"]
    else:
        mutation_code = row["Mutation(s)_PDB"]

    if len(mutation_code) == 0:
        return []

    mutation_codes = mutation_code.split(",")

    decoded_mutations = []
    for code in mutation_codes:
        mutation_info = {
            "original_amino_acid": code[0],
            "chain": code[1],
            "index": int(code[2:-1]),
            "new_amino_acid": code[-1],
            "code": mutation_code
        }
        decoded_mutations.append(mutation_info)

    return decoded_mutations


def mutate_and_relax(pose: Pose, mutations: List[Dict], relax_fn: Mover):
    all_codes = []
    for mutation in mutations:
        mutate_residue(pose, pose.pdb_info().pdb2pose(mutation["chain"], mutation["index"]), mutation["new_amino_acid"], pack_radius=10, pack_scorefxn=pyrosetta.get_fa_scorefxn())
        all_codes.append(mutation["code"])

    # relax pose
    relax_fn.apply(pose)

    return all_codes


def create_relaxed_mutations(config: Dict):
    summary_path, pdb_folder = get_data_paths(config, "SKEMPI.v2")
    summary_df = pd.read_csv(summary_path, sep=";")

    out_path = os.path.join(config["DATA"]["path"], config["DATA"]["SKEMPI.v2"]["folder_path"], config["DATA"]["SKEMPI.v2"]["mutated_pdb_path"])

    relaxed_wt_pose = None
    relaxed_wt_pose_pdb = None
    relaxed_wt_pose_score = None

    relax_fn, packer = load_relax_function_and_packer()
    score_fn = pyrosetta.get_fa_scorefxn()

    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
        try:
            pdb = summary_df.iloc[idx, 0].split("_")[0]
            path = os.path.join(pdb_folder, pdb + ".pdb")

            if pdb == relaxed_wt_pose_pdb:
                pose = relaxed_wt_pose.clone()
                wt_score = relaxed_wt_pose_score
            else:
                pose = load_pose(path)
                relax_fn.apply(pose)
                relaxed_wt_pose_pdb = pdb
                relaxed_wt_pose = pose.clone()
                wt_score = score_fn(pose)
                relaxed_wt_pose_score = wt_score

            mutations = get_mutations(summary_df, idx)

            mutation_codes = mutate_and_relax(pose, mutations, relax_fn)

            # store pdb file
            out_path = os.path.join(out_path, pdb + "_".join(mutation_codes) + ".pdb")
            pose.dump_pdb(out_path)

            mutated_score = score_fn(pose)

            summary_df.loc[idx, 'ddG (REU)'] = wt_score - mutated_score
            summary_df.to_csv(summary_path, index=False)

        except Exception as e:
            logger.error("Error in row {} - Error Message: {}".format(idx, e))
            pass


def realx_with_openfold(pose: Pose):
    tmp_file = "./tmp.pdb"
    pose.dump_pdb(tmp_file)

    with open(tmp_file) as f:
        pdb_string = f.read()


def main():
    folder_path = os.getenv('ABAG_PATH')
    config_path = os.path.join(folder_path, "abag_affinity/config.yaml")
    config = read_config(config_path)
    create_relaxed_mutations(config)


if __name__ == "__main__":
    logger.info('Started program')
    main()
    logger.info('Finished program')
