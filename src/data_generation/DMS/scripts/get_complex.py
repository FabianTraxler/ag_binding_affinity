# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
import tempfile
from typing import Dict, Tuple
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose, dump_comment_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
import os
import yaml
import tarfile
import shutil
import subprocess

from common import substitute_chain, order_substitutions, mutate

pyrosetta.init()#extra_options="-mute all")

scorefxn = pyrosetta.get_fa_scorefxn()
packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)

relax = FastRelax()
scorefxn = pyrosetta.get_fa_scorefxn()
relax.set_scorefxn(scorefxn)
relax.constrain_relax_to_start_coords(True)

def get_complex_metadata(publication:str, antibody: str, antigen: str, metadata: Dict) -> Dict:
    publication_data = metadata[publication]
    for complex in publication_data["complexes"]:
        if complex["antigen"]["name"] == antigen and complex["antibody"]["name"] == antibody:
            return complex["pdb"]

    raise RuntimeError(f"Complex not found in Metadata: {publication}, {antibody}, {antigen}")


def load_large_file(pdb_id: str, download_folder: str):
    filename = PDBList().retrieve_pdb_file(pdb_id, pdir=download_folder, file_format="bundle")
    tar_file = tarfile.open(filename)
    tar_file.extractall(path=f"{download_folder}/{pdb_id}")
    tar_file.close()
    complete_file = ""
    i = 1
    filename_part = f"{download_folder}/{pdb_id}/{pdb_id}-pdb-bundle{i}.pdb"
    while os.path.exists(filename_part):
        with open(filename_part) as f:
            complete_file += f.read()

        i += 1
        filename_part = f"{download_folder}/{pdb_id}/{pdb_id}-pdb-bundle{i}.pdb"

    shutil.rmtree(f"{download_folder}")

    filename = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    with open(filename, "w") as f:
        f.write(complete_file)

    return filename


def clean_tidy_pdb(pdb_file: str, out_file: str):
    """
    Apparently, the dms_curated.csv mutations are defined on insert-fixed residue indices

    TODO could/should use the function in pdb_processing.py
    Clean temporary PDB file and then save its cleaned version as the original PDB file
    retry 3 times because these commands sometimes do not properly write to disc
    """
    retries = 0
    while not os.path.exists(out_file):
        command = f'pdb_sort "{pdb_file}" | pdb_tidy | pdb_fixinsert | pdb_delhetatm > "{out_file}"'
        print(command)
        subprocess.run(command, shell=True)
        retries += 1
        if retries >= 3:
            raise RuntimeError(f"Error in PDB Utils commands to clean PDB {out_file}")

    return out_file

def get_pdb(publication:str, antibody: str, antigen: str, metadata: Dict, project_root: str) -> Tuple[pyrosetta.Pose, str]:
    complex_metadata = get_complex_metadata(publication, antibody, antigen, metadata)
    #keep_chains = complex_metadata["chains"]["antibody"] + complex_metadata["chains"]["antigen"]
    if "id" in complex_metadata:
        # load file from Protein Data Bank and perform mutations

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = PDBList().retrieve_pdb_file(complex_metadata["id"], file_format="pdb", pdir=tmpdirname)
            if not os.path.exists(filename): # download did not work in pdb Format
                filename = load_large_file(complex_metadata["id"].lower(), tmpdirname)  # f"{publication}:{antibody}:{antigen}")
            # remove irrelevant chains
            #filename = clean_tidy_pdb(filename, keep_chains)
            chain_renaming = {v: k for rename_dict in complex_metadata["chains"].values() for k,v in rename_dict.items()}
            filename = remove_and_rename_chains(filename, chain_renaming)

    elif "file" in complex_metadata:


        # load pdb from disc and return
        filename = os.path.join(project_root, complex_metadata["file"])
        #filename = clean_tidy_pdb(filepath, keep_chains)
    else:
        raise RuntimeError(f"Neither 'id' nor 'file' found in complex metadata: {publication}, {antibody}, {antigen}")

    pose = load_pose(filename)
    mutations = ""
    if "mutations" in complex_metadata:
        mutations = complex_metadata["mutations"]
    return pose, mutations


def remove_and_rename_chains(filename: str, chain_renaming: Dict) -> str:
    renaming_commands = " | ".join([
        f"pdb_rplchain -{chain}:{new_id}"
        for chain, new_id in order_substitutions(chain_renaming).items()
    ])

    tmp_pdb_file = tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
    command = f"pdb_sort {filename} | pdb_tidy | pdb_selchain -{','.join(chain_renaming.keys())} | {renaming_commands} > {tmp_pdb_file.name}"
    subprocess.run(command, shell=True, check=True)

    return tmp_pdb_file.name


def load_pose_without_header(pdb_path: str):
    raise NotImplementedError("code is broken")

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

    #os.remove(new_path)

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

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

publication = snakemake.wildcards.publication

if "mason21" in publication:
    publication = "mason21_optim_therap_antib_by_predic"

with open(snakemake.input["metadata_file"], "r") as f:
    metadata = yaml.safe_load(f)
project_root = snakemake.params["project_root"]

# load pose and get mutations
pose, mutation_code = get_pdb(publication, snakemake.wildcards.antibody, snakemake.wildcards.antigen, metadata, project_root)
mutation_code = substitute_chain(metadata, mutation_code, snakemake.wildcards.publication, snakemake.wildcards.antibody, snakemake.wildcards.antigen)

if mutation_code != "":
    # Mutate
    decoded_mutation = convert_mutations(mutation_code)
    mutate(pose, decoded_mutation)

# Relax
# relax.apply(pose)  # TODO re-enable!

pose.dump_pdb(out_path + ".tmp")
clean_tidy_pdb(out_path + ".tmp", out_path)

