# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import os
from pathlib import Path

import pyrosetta
from Bio.PDB import PDBIO
from Bio.PDB.PDBParser import PDBParser
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
from pyrosetta.rosetta.protocols.moves import Mover
from pyrosetta.rosetta.protocols.relax import FastRelax

parser = PDBParser(PERMISSIVE=3)

if "snakemake" not in globals(): # use fake snakemake object for debugging
    from abag_affinity.utils.config import (get_data_paths,
                                            get_resources_paths, read_config)
    config = read_config("../../../abag_affinity/config.yaml")
    _, pdb_path = get_resources_paths(config, "AntibodyBenchmark")
    benchmark_folder_path = os.path.join(config["DATA"]["path"], config["DATA"]["AbDb"]["folder_path"])

    sample_pdb_id = "1WEJ"
    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(pdb_path, sample_pdb_id + "_r_b.pdb"), os.path.join(pdb_path, sample_pdb_id + "_l_b.pdb")]
    snakemake.output = [os.path.join(relaxed_pdbs + sample_pdb_id)]


pyrosetta.init(extra_options="-load_PDB_components false")#-mute all")

relax = FastRelax()
scorefxn = pyrosetta.get_fa_scorefxn()
relax.set_scorefxn(scorefxn)
relax.constrain_relax_to_start_coords(True)

#relax.max_iter(1)

packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)


def bind_chains(receptor_path, ligand_path):
    receptor = parser.get_structure("", receptor_path)
    ligand = parser.get_structure("", ligand_path)
    for chain in ligand.get_chains():
        chain.detach_parent()
        receptor[0].add(chain)

    return receptor


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    pose = pyrosetta.pose_from_pdb(pdb_path)
    testPose = pyrosetta.Pose()
    testPose.assign(pose)
    return testPose


def add_score(pose: Pose):
    score = scorefxn(pose)
    add_comment(pose, "rosetta_energy_score", str(score))


relax_out_path = snakemake.output[0]
unrelax_out_path = snakemake.output[1]
Path(relax_out_path).parent.mkdir(parents=True, exist_ok=True)
Path(unrelax_out_path).parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input[0]

bound_complex = bind_chains(snakemake.input[0], snakemake.input[1])


pdb_id = snakemake.input[0].split("/")[-1].split("_")[0]
# save to temp file
temp_file = pdb_id + "_temp.pdb"
io = PDBIO()
io.set_structure(bound_complex)
io.save(temp_file)

pose = load_pose(temp_file)
unrelax_pose = pose.clone()
add_score(unrelax_pose)
dump_comment_pdb(unrelax_out_path, pose)


relax.apply(pose)
add_score(pose)
os.remove(temp_file)

dump_comment_pdb(relax_out_path, pose)
