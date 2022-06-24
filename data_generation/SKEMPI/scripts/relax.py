# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import os
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import pyrosetta
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.protocols.moves import Mover
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax


pyrosetta.init(extra_options="-mute all")

relax = FastRelax()
scorefxn = pyrosetta.get_fa_scorefxn()
relax.set_scorefxn(scorefxn)
relax.constrain_relax_to_start_coords(True)

packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    pose = pyrosetta.pose_from_pdb(pdb_path)
    testPose = pyrosetta.Pose()
    testPose.assign(pose)
    return testPose


def add_score(pose: Pose):
    score = scorefxn(pose)
    add_comment(pose, "rosetta_energy_score", str(score))


out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]

pose = load_pose(file_path)
relax.apply(pose)
add_score(pose)

dump_comment_pdb(out_path, pose)
