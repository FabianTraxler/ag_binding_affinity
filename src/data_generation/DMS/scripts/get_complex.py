# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
from typing import Dict, List, Tuple
import pyrosetta
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.toolbox.rcsb import pose_from_rcsb
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
import os
import yaml

three2one_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


if "snakemake" not in globals(): # use fake snakemake object for debugging
    project_root = (Path(__file__).parents[4]).resolve()
    out_folder = os.path.join(project_root, "data/DMS/")

    complexes = [("phillips21_bindin","cr9114", "h1newcal99")] #("starr21_prosp_covid","lycov016", "cov2rbd")

    file = out_folder + "prepared_pdbs/{}/{}_{}.pdb"

    metadata_file = os.path.join(project_root, "data/metadata_dms_studies.yaml")


    snakemake = type('', (), {})()
    snakemake.output = [file.format(complex, antibody, antigen) for (complex, antibody, antigen) in complexes ]
    snakemake.params = {}
    snakemake.params["metadata_file"] = metadata_file
    snakemake.params["project_root"] = project_root


pyrosetta.init(extra_options="-mute all")

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


def get_pdb(publication:str, antibody: str, antigen: str, metadata: Dict, project_root: str) -> Tuple[pyrosetta.Pose, str]:
    complex_metadata = get_complex_metadata(publication, antibody, antigen, metadata)
    if "id" in complex_metadata:
        # load file from Protein Data Bank and perform mutations
        filename = PDBList().retrieve_pdb_file(complex_metadata["id"], file_format="pdb")
        # remove irrelevant chains
        pose = remove_chains(filename, complex_metadata["chains"])
        mutations = ""
        if "mutations" in complex_metadata:
            mutations = complex_metadata["mutations"]
        return pose, mutations
    elif "file" in complex_metadata:
        # load pdb from disc and return
        filepath = os.path.join(project_root, complex_metadata["file"])
        pose = load_pose(filepath)

        return pose, ""

    else:
        raise RuntimeError(f"Neither 'id' nor 'file' found in complex metadata: {publication}, {antibody}, {antigen}")


def remove_chains(filename: str, keep_chains: Dict) -> Pose:
    # use BioPython for easier handling

    structure = PDBParser().get_structure("_", filename)

    remove_chains = []
    # get all chains and mark redundant ones
    all_chains = list(structure.get_chains())
    for i, chain in enumerate(all_chains):
        if chain.id not in keep_chains["antibody"] and chain.id not in keep_chains["antigen"]:
            remove_chains.append((chain.id, chain.parent.id))

    for (redundant_chain, model) in remove_chains:
        structure[model].detach_child(redundant_chain)

    class ModelSelect(Select):
        def accept_residue(self, res):
            if res.parent.parent.id == 0:
                return True
            else:
                return False

    io = PDBIO()
    io.set_structure(structure)
    io.save(filename, ModelSelect())

    pose = load_pose(filename)
    if os.path.exists(filename):
        os.remove(filename)
    return pose


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


def mutate(pose: Pose, mutations: List[Dict]):
    for mutation in mutations:
        try:
            original_residue = pose.pdb_rsd((mutation["chain"], mutation["index"])).name()
            # check if mutation is correct
            assert three2one_code[original_residue.upper()] == mutation["original_amino_acid"]

            mutate_residue(pose, pose.pdb_info().pdb2pose(mutation["chain"], mutation["index"]), mutation["new_amino_acid"], pack_radius=10, pack_scorefxn=pyrosetta.get_fa_scorefxn())
        except:
            pass # Mutation not in PDB File


out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

antibody, antigen = out_path.split("/")[-1].split(".")[0].split("_")
publication = out_path.split("/")[-2]

with open(snakemake.params["metadata_file"], "r") as f:
    metadata = yaml.safe_load(f)

project_root = snakemake.params["project_root"]

# load pose and get mutations
pose, mutation_code = get_pdb(publication, antibody, antigen, metadata, project_root)

if mutation_code != "": # mutate and relax
    # mutation
    decoded_mutation = convert_mutations(mutation_code)
    mutate(pose, decoded_mutation)

    # relax pose
    relax.apply(pose)

# save to disc
dump_comment_pdb(out_path, pose)
