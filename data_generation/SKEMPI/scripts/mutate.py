# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
from typing import Dict, List
import pyrosetta
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
import os


if "snakemake" not in globals(): # use fake snakemake object for debugging
    from abag_affinity.utils.config import read_yaml, get_data_paths
    config = read_yaml("../../../abag_affinity/config.yaml")
    skempi_df_path, pdb_path = get_data_paths(config, "SKEMPI.v2")
    data_path = config["DATA"]["path"]

    skempi_folder_path = os.path.join(data_path, config["DATA"]["SKEMPI.v2"]["folder_path"])

    sample_pdb_id = "3BN9"
    mutation_code = "RB51A"
    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(skempi_folder_path + "/relaxed_wildtype/" + sample_pdb_id + ".pdb")]
    snakemake.output = [os.path.join(skempi_folder_path + "/relaxed_mutated/", sample_pdb_id, mutation_code + ".pdb")]
    snakemake.wildcards = type('', (), {})()
    snakemake.wildcards.mutation = mutation_code


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


def mutate(pose: Pose, mutations: List[Dict]):
    for mutation in mutations:
        mutate_residue(pose, pose.pdb_info().pdb2pose(mutation["chain"], mutation["index"]), mutation["new_amino_acid"], pack_radius=10, pack_scorefxn=pyrosetta.get_fa_scorefxn())


def add_score(pose: Pose):
    score = scorefxn(pose)
    add_comment(pose, "rosetta_energy_score", str(score))

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]
mutation_code = snakemake.wildcards.mutation

pose = load_pose(file_path)
decoded_mutation = convert_mutations(mutation_code)
mutate(pose, decoded_mutation)
add_score(pose)

dump_comment_pdb(out_path, pose)
