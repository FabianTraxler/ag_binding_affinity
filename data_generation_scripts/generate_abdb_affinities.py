# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import logging
import os
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import ast
import pyrosetta
from pyrosetta.rosetta.protocols import docking, rigid

from abag_affinity.utils.config import read_yaml, get_data_paths


# init logger
logger = logging.getLogger("AbDb-Affinity-Generation")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = logging.FileHandler("logs/data_generation_scripts.log")
log_file.setLevel(logging.INFO)
log_file.setFormatter(formatter)
logger.addHandler(log_file)

pyrosetta.init(extra_options="-mute all")


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    pose = pyrosetta.pose_from_pdb(pdb_path)
    testPose = pyrosetta.Pose()
    testPose.assign(pose)
    return testPose


def load_relax_and_score_function():
    from pyrosetta.rosetta.protocols.relax import FastRelax

    relax = FastRelax()
    scorefxn = pyrosetta.get_fa_scorefxn()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)

    return relax, scorefxn


def get_partners(summary_df: pd.DataFrame, location: int, cleaned_file: bool = True):
    row = summary_df.iloc[location]

    antibods_chains = "".join(ast.literal_eval(row["antibody_chains"])).upper()
    antigen_chains = "".join(ast.literal_eval(row["antigen_chains"])).upper()

    partners = antibods_chains + "_" + antigen_chains
    return partners


def unbind(pose, partners):
    STEP_SIZE = 100
    JUMP = 2
    docking.setup_foldtree(pose, partners, pyrosetta.Vector1([-1,-1,-1]))
    trans_mover = rigid.RigidBodyTransMover(pose,JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose)


def calculate_affinities(config: Dict):
    summary_path, pdb_folder = get_data_paths(config, "AbDb")
    summary_df = pd.read_csv(summary_path)

    relax, scorefxn = load_relax_and_score_function()

    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
        try:
            pdb_file = summary_df.loc[idx, "abdb_filename"]
            path = os.path.join(pdb_folder, pdb_file)

            pose = load_pose(path)

            #relax.apply(pose)
            # calculate original score
            original_score = scorefxn(pose)

            partners = get_partners(summary_df, idx)
            unbind(pose, partners)
            relax.apply(pose)

            # calculate unbound score
            unbound_score = scorefxn(pose)

            summary_df.loc[idx, 'dG (REU) - no relax'] = original_score - unbound_score
        except Exception as e:
            logger.error("Error in row {} - Error Message: {}".format(idx, e))
            pass

        summary_df.to_csv(summary_path, index=False)


def main():
    config_path = "abag_affinity/config.yaml"
    config = read_yaml(config_path)
    calculate_affinities(config)


if __name__ == "__main__":
    logger.info('Started program')
    main()
    logger.info('Finished program')

