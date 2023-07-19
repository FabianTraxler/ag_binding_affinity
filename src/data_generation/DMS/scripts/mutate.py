# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import logging
import yaml

from pathlib import Path
from typing import Dict, List

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
import os
from parallel import submit_jobs
from common import mutate

pyrosetta.init(extra_options="-mute all")

scorefxn = pyrosetta.get_fa_scorefxn()
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


def convert_mutations(mutation_code: str):
    if len(mutation_code) == 0 or mutation_code == "original" or mutation_code == "WT":
        return []

    mutation_codes = mutation_code.split(";")

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


def add_score(pose: Pose):
    score = scorefxn(pose)
    add_comment(pose, "rosetta_energy_score", str(score))


publication = snakemake.input.pdb.split("/")[-2]
antibody, antigen = snakemake.input.pdb.split("/")[-1].split(".")[0].split("_")

original_pose = load_pose(snakemake.input.pdb)

pose = pyrosetta.Pose()
pose.detached_copy(original_pose)

mutation_code = snakemake.wildcards.mut
if mutation_code == "original":
    logging.warning("don't use original! renaming to 'WT'")
    mutation_code = "WT"

decoded_mutation = convert_mutations(mutation_code)
mutate(pose, decoded_mutation)
add_score(pose)
dump_comment_pdb(snakemake.output.mutated_pdb, pose)
