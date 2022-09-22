# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import os
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax

if "snakemake" not in globals(): # use fake snakemake object for debugging
    from abag_affinity.utils.config import get_data_paths, read_config
    config = read_config("../../../abag_affinity/config.yaml")
    skempi_df_path, pdb_path = get_data_paths(config, "SKEMPI.v2")
    data_path = config["DATA"]["path"]

    skempi_folder_path = os.path.join(data_path, config["DATA"]["SKEMPI.v2"]["folder_path"])

    sample_pdb_id = "2NYY"
    mutation_code = "IA928A"
    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(skempi_folder_path + "/relaxed_mutated/" + sample_pdb_id, mutation_code + ".pdb")]
    snakemake.output = [os.path.join(skempi_folder_path + "/relaxed_mutated_relaxed/", sample_pdb_id, mutation_code + ".pdb")]
    snakemake.wildcards = type('', (), {})()
    snakemake.wildcards.mutation = mutation_code


pyrosetta.init(extra_options="")

relax = FastRelax()
relax.max_iter(1)
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
