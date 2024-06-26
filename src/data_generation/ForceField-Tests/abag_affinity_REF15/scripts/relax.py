# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
from pyrosetta.rosetta.protocols.moves import Mover
from pyrosetta.rosetta.protocols.relax import FastRelax

# if "snakemake" not in globals(): # use fake snakemake object for debugging
#     import os

#     from abag_affinity.utils.config import get_data_paths, read_config
#     config = read_config("../../../abag_affinity/config.yaml")
#     _, pdb_path = get_data_paths(config, "AbDb_REF15")
#     abdb_folder_path = os.path.join(config["DATA"]["path"], config["DATA"]["AbDb_REF15"]["folder_path"])

#     sample_pdb_id = "2YPV_1.pdb"
#     snakemake = type('', (), {})()
#     snakemake.input = [os.path.join(pdb_path, sample_pdb_id)]
#     snakemake.output = [os.path.join(abdb_folder_path + "/bound_relaxed/" + sample_pdb_id)]


def relax(file_path, out_path):
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

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    pose = load_pose(file_path)
    relax.apply(pose)
    add_score(pose)

    dump_comment_pdb(out_path, pose)


try:
    relax(snakemake.input[0], snakemake.output[0])
except NameError:
    pass
