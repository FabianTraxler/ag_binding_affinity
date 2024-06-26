# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import os
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, add_comment, get_all_comments
from pyrosetta.rosetta.protocols.relax import FastRelax


pyrosetta.init(extra_options="")

relax = FastRelax()
scorefxn = pyrosetta.get_fa_scorefxn()
relax.set_scorefxn(scorefxn)
relax.constrain_relax_to_start_coords(True)

packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)


def load_pose_without_header(pdb_path: str):
    with open(pdb_path) as f:
        lines = f.readlines()

    i = 0
    while True:
        if lines[i][:4] != "ATOM":
            i += 1
        else:
            break

    new_path = pdb_path.split(".")[0] + "_tmp.pdb"

    with open(new_path, "w") as f:
        f.writelines(lines[i:])

    pose = pyrosetta.pose_from_pdb(new_path)

    os.remove(new_path)

    return pose


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    try:
        pose = pyrosetta.pose_from_pdb(pdb_path)
    except RuntimeError:
        pose = load_pose_without_header(pdb_path)
    testPose = pyrosetta.Pose()
    testPose.assign(pose)
    return testPose


file_path = snakemake.input[0]

pose = load_pose(file_path)
relax.apply(pose)

Path(snakemake.output.pdb_comments).write_text("\n".join(get_all_comments(pose)))
pose.dump_pdb(snakemake.output.pdb)
