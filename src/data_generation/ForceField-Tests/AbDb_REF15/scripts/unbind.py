# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path

import pyrosetta
from Bio.PDB.PDBParser import PDBParser
from pyrosetta.rosetta.core.pose import (Pose, add_comment, dump_comment_pdb,
                                         get_chain_from_chain_id)
from pyrosetta.rosetta.protocols import docking, rigid
from pyrosetta.rosetta.protocols.moves import Mover

if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os

    from abag_affinity.utils.config import get_resources_paths, read_config
    config = read_config("../../../abag_affinity/config.yaml")
    _, pdb_path = get_resources_paths(config, "AbDb")
    abdb_folder_path = os.path.join(config["DATA"]["path"], config["DATA"]["AbDb"]["folder_path"])

    sample_pdb_id = "2YPV_1.pdb"

    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(pdb_path, sample_pdb_id)]#[os.path.join(abdb_folder_path + "/bound_relaxed/" + sample_pdb_id)]
    snakemake.output = [os.path.join(abdb_folder_path + "/unbound/" + sample_pdb_id)]


pyrosetta.init(extra_options="-mute all")

scorefxn = pyrosetta.get_fa_scorefxn()

packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    pose = pyrosetta.pose_from_pdb(pdb_path)
    testPose = pyrosetta.Pose()
    testPose.assign(pose)
    return testPose


def add_score(pose: Pose):
    score = scorefxn(pose)
    add_comment(pose, "rosetta_energy_score", str(score))


def get_partners(structure: Pose):

    chains = structure.get_chains()

    chains = [ chain.id for chain in chains]

    antibody_chains = ""
    antigen_chains = ""

    if "L" in chains:
        antibody_chains += "L"
        chains.remove("L")
    if "H" in chains:
        antibody_chains += "H"
        chains.remove("H")

    for chain in chains:
        antigen_chains += chain

    partners = antibody_chains + "_" + antigen_chains
    return partners


def unbind(pose, partners):
    STEP_SIZE = 100
    JUMP = 1
    docking.setup_foldtree(pose, partners, pyrosetta.Vector1([-1,-1,-1]))
    trans_mover = rigid.RigidBodyTransMover(pose, JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose)

parser = PDBParser(PERMISSIVE=3)

parser = PDBParser(PERMISSIVE=3)

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]

pose = load_pose(file_path)

structure = parser.get_structure("", file_path)
partners = get_partners(structure)

unbind(pose, partners)

add_score(pose)

dump_comment_pdb(out_path, pose)
