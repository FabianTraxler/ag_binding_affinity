# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
import pyrosetta
from pyrosetta.rosetta.protocols.moves import Mover
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax


if "snakemake" not in globals(): # use fake snakemake object for debugging
    import os
    from abag_affinity.utils.config import read_yaml, get_resources_paths, get_data_paths
    config = read_yaml("../../../abag_affinity/config.yaml")
    _, pdb_path = get_resources_paths(config, "AntibodyBenchmark")
    abdb_folder_path = os.path.join(config["DATA"]["path"], config["DATA"]["AntibodyBenchmark"]["folder_path"])
    benchmark_scores_path, relaxed_pdbs = get_data_paths(config, "AntibodyBenchmark")

    sample_pdb_id = "3EOA_r_u.pdb"
    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(pdb_path, sample_pdb_id)]
    snakemake.output = [os.path.join(relaxed_pdbs[0], sample_pdb_id)]


pyrosetta.init(extra_options="-load_PDB_components false -no_optH false")#-mute all")

relax = FastRelax()
scorefxn = pyrosetta.get_fa_scorefxn()
relax.set_scorefxn(scorefxn)
relax.constrain_relax_to_start_coords(True)

packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)


def load_pose(pdb_path: str) -> pyrosetta.Pose:
    #from pyrosetta.toolbox.cleaning import cleanATOM
    #cleanATOM(pdb_path)
    pose = pyrosetta.pose_from_pdb( pdb_path)
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

pose = load_pose(file_path)
unrelax_pose = pose.clone()
add_score(unrelax_pose)
dump_comment_pdb(unrelax_out_path, pose)

relax.apply(pose)
add_score(pose)

dump_comment_pdb(relax_out_path, pose)
