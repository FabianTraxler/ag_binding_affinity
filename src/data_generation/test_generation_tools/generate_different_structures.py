# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from typing import Dict, List, Tuple, Union
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
import numpy as np
import pyrosetta
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
import os
from pyrosetta.rosetta.protocols.relax import FastRelax
import shutil
import tarfile
from pathlib import Path
from biopandas.pdb import PandasPdb
import subprocess
from Bio.PDB.Structure import Structure


from abag_affinity.binding_ddg_predictor.utils.protein import  RESIDUE_SIDECHAIN_POSTFIXES

three2one_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


pyrosetta.init()#extra_options="-mute all")

scorefxn = pyrosetta.get_fa_scorefxn()
packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)

relax = FastRelax()
scorefxn = pyrosetta.get_fa_scorefxn()
relax.set_scorefxn(scorefxn)
relax.constrain_relax_to_start_coords(True)



def load_large_file(pdb_id: str, download_fodler: str, out_path: str):
    filename = PDBList().retrieve_pdb_file(pdb_id, pdir=download_fodler, file_format="bundle")
    tar_file = tarfile.open(filename)
    tar_file.extractall(path=f"./{download_fodler}/{pdb_id}")
    tar_file.close()
    complete_file = ""
    i = 1
    filename_part = f"{download_fodler}/{pdb_id}/{pdb_id}-pdb-bundle{i}.pdb"
    while os.path.exists(filename_part):
        with open(filename_part) as f:
            complete_file += f.read()

        i += 1
        filename_part = f"{download_fodler}/{pdb_id}/{pdb_id}-pdb-bundle{i}.pdb"

    shutil.rmtree(f"{download_fodler}")

    with open(out_path, "w") as f:
        f.write(complete_file)

    return out_path



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
    #if os.path.exists(filename):
    #    os.remove(filename)
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


def mutate(pose: Pose, mutations: List[Dict]) -> str:
    perform_mutations = []
    for mutation in mutations:
        original_residue = pose.pdb_rsd((mutation["chain"], mutation["index"]))
        if original_residue is None:
            # Residue to mutate not in interface_hull
            # Pass this mutation and do not add it to the list
            continue
        original_residue_name = original_residue.name()
        a = 0

        # check if mutation is correct
        assert three2one_code[original_residue_name.upper()[:3]] == mutation["original_amino_acid"], \
            f"Residue {mutation['original_amino_acid']} at chain {mutation['chain']} and index {mutation['index']} is " \
            f"different from found residue {three2one_code[original_residue_name.upper()]}"

        mutate_residue(pose, pose.pdb_info().pdb2pose(mutation["chain"], mutation["index"]), mutation["new_amino_acid"], pack_radius=10, pack_scorefxn=pyrosetta.get_fa_scorefxn())
        perform_mutations.append(mutation["original_amino_acid"] + mutation["chain"] +
                                 str(mutation["index"]) + mutation["new_amino_acid"])

    return ";".join(perform_mutations)


def perform_mutation(mutation_code: str, original_pose, pdb_out_path):
    try:
        pose = pyrosetta.Pose()
        pose.detached_copy(original_pose)
        if mutation_code == "original" or mutation_code == "WT" or (not isinstance(mutation_code, str) and np.isnan(mutation_code)):
            mutation_code = "WT"
            performed_mutations = []
        else:
            decoded_mutation = convert_mutations(mutation_code)
            performed_mutations = mutate(pose, decoded_mutation)

        if not os.path.exists(pdb_out_path):
            dump_comment_pdb(pdb_out_path, pose)

        error = False
        error_msg = ""
    except Exception as e:
        error = True
        error_msg = e
        performed_mutations = ""

    return error, error_msg, mutation_code, performed_mutations, pdb_out_path


def clean_and_tidy_pdb(pdb_id: str, pdb_file_path: Union[str, Path], cleaned_file_path: Union[str, Path]):
    Path(cleaned_file_path).parent.mkdir(exist_ok=True, parents=True)

    tmp_pdb_filepath = f'{pdb_file_path}.tmp'
    shutil.copyfile(pdb_file_path, tmp_pdb_filepath)

    # remove additional models - only keep first model
    structure, _ = read_file(pdb_id, tmp_pdb_filepath)
    model = structure[0]
    io = PDBIO()
    io.set_structure(model)
    io.save(tmp_pdb_filepath)

    # Clean temporary PDB file and then save its cleaned version as the original PDB file
    # retry 3 times because these commands sometimes do not properly write to disc
    retries = 0
    while not os.path.exists(cleaned_file_path):
        command = f'pdb_sort {tmp_pdb_filepath} | pdb_tidy | pdb_fixinsert | pdb_delhetatm  > {cleaned_file_path}'
        subprocess.run(command, shell=True)
        retries += 1
        if retries >= 3:
            raise RuntimeError(f"Error in PDB Utils commands to clean PDB {tmp_pdb_filepath}")

    cleaned_pdb = PandasPdb().read_pdb(cleaned_file_path)
    input_atom_df = cleaned_pdb.df['ATOM']

    # remove all duplicate (alternate location residues)
    filtered_df = input_atom_df.drop_duplicates(subset=["atom_name", "chain_id", "residue_number"])

    # remove all residues that do not have at least N, CA, C atoms
    filtered_df = filtered_df.groupby(["chain_id", "residue_number", "residue_name"]).filter(
        lambda x: x["atom_name"].values[:3].tolist() == ["N", "CA", "C"])

    # drop H atoms
    filtered_df = filtered_df[filtered_df['element_symbol'] != 'H']

    # remove all non-standard atoms - used in Binding_DDG preprocessing
    all_postfixes = [ "" ]
    for postfixes in RESIDUE_SIDECHAIN_POSTFIXES.values():
        all_postfixes += postfixes
    atom_name_postfix = filtered_df['atom_name'].apply(get_atom_postfixes)
    filtered_df = filtered_df[atom_name_postfix.isin(all_postfixes)]

    assert len(filtered_df) > 0, f"No atoms in pdb file after cleaning: {pdb_file_path}"

    cleaned_pdb.df['ATOM'] = filtered_df.reset_index(drop=True)

    cleaned_pdb.to_pdb(path=str(cleaned_file_path),
                       records=["ATOM"],
                       gz=False,
                       append_newline=True)

    # Clean up from using temporary PDB file for tidying
    if os.path.exists(tmp_pdb_filepath):
        os.remove(tmp_pdb_filepath)


def read_file(structure_id: str, path: Union[str, Path]) -> Tuple[Structure, Dict]:
    """ Read a PDB file and return the structure and header

    Args:
        structure_id: PDB ID
        path: Path of the PDB file

    Returns:
        Tuple: Structure (Bio.PDB object), header (Dict)
    """
    parser = PDBParser(PERMISSIVE=3)

    try:
        structure = parser.get_structure(structure_id, str(path))
        header = parser.get_header()
    except Exception as e:
        raise RuntimeError(f"Could not load pdb_file {path}: {e}")

    return structure, header


def get_atom_postfixes(atom_name: str):
    # very similar to binding_ddg preprocessing
    if atom_name in ('N', 'CA', 'C', 'O'):
        return ""
    if atom_name[-1].isnumeric():
        return atom_name[-2:]
    else:
        return atom_name[-1:]


if __name__ == "__main__":
    filename = "wu17_in-c05-h3perth09"
    pdb_id = "4FP8" #"6CDI"
    start_structure_folder = "./pdbs"
    start_structure_path = os.path.join(start_structure_folder, pdb_id + ".pdb")
    out_path = "./pdbs"
    mutation_code = "KA50E;NA53D;NA54S;RA57Q;IA62K;DA63N;HA75Q;VA78G;EA82K;TA83K;FA94Y;IA121N;TA122N;GA124S;TA126N;GA135T;NA137S;KA140I;GA142R;PA143S;GA144K;SA145N;GA146S;KA156H;SA157L;GA158N;SA159F;TA160K;VA163A;DA172E;NA173Q;SA186G;NA188D;QA189K;EA190D;TA192I;SA193F;VA196A;VA202I;RA207K;IA213V;IA214S;WA222R;GA225N;LA226I;SA227P;VA242I;VA244L;NA248T;MA260I;TA262S;DA275G;TA276K;IA278N;KA299R;KA307R;GA310K;HA311Q;HA312N" #Lc520C;Fl55P" # "Lc520C;Sl43K" #"Lc520C"
    extra_mutations = "VH111P;SH112G;AH113S"
    perfrom_relax = True
    perfrom_deeprefine = True
    keep_chains = {
        "antibody": {"L", "H"},
        "antigen": {"A"}
    }

    if not os.path.exists(start_structure_path) and pdb_id != "":
        out_name = PDBList().retrieve_pdb_file(pdb_id, file_format="pdb")
        if not os.path.isfile(out_name): # download did not work in pdb Format
            out_name = load_large_file(pdb_id.lower(), "downloads", start_structure_path)
        shutil.move(out_name, start_structure_path)

        remove_chains(start_structure_path, keep_chains)
    pose = load_pose(start_structure_path)

    mut_out_path = os.path.join(out_path, filename + "initial_mutations.pdb")
    error, error_msg, mutation_code, performed_mutations, pdb_out_path = perform_mutation(mutation_code, pose, mut_out_path)

    if extra_mutations != "":
        # Clean temporary PDB file and then save its cleaned version as the original PDB file
        # retry 3 times because these commands sometimes do not properly write to disc
        retries = 0
        cleaned = os.path.join(out_path, filename + "initial_mutations_cleaned.pdb")
        while not os.path.exists(cleaned):
            command = f'pdb_sort {mut_out_path} | pdb_tidy | pdb_fixinsert | pdb_delhetatm  > {cleaned}'
            subprocess.run(command, shell=True)
            retries += 1
            if retries >= 3:
                raise RuntimeError(f"Error in PDB Utils commands to clean PDB {mut_out_path}")

        pose = load_pose(cleaned)
        mut_out_path = os.path.join(out_path, "4FP8to5UMN_all_mutations.pdb")
        error, error_msg, mutation_code, performed_mutations, pdb_out_path = perform_mutation(extra_mutations, pose,
                                                                                              mut_out_path)
        pose = load_pose(mut_out_path)

    if perfrom_relax:
        relax_out_path = os.path.join(out_path, filename + "_relax.pdb")
        relax.apply(pose)
        dump_comment_pdb(relax_out_path, pose)

    if perfrom_deeprefine:
        deeprefine_out_path = os.path.join(out_path, "4FP8to5UMN_all_mutations_cleaned.pdb")
        clean_and_tidy_pdb(filename, mut_out_path, deeprefine_out_path)