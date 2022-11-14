# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyrosetta
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.core.pose import Pose, add_comment, dump_comment_pdb
import os
from parallel import submit_jobs

three2one_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


if "snakemake" not in globals(): # use fake snakemake object for debugging
    import pandas as pd

    project_root =  "../../../../" # three directories above
    folder_path = os.path.join(project_root, "data/DMS")

    publication = "phillips21_bindin"
    sample_pdb_id = "cr6261_h1newcal99"

    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(folder_path, "interface_hull_pdbs", publication, sample_pdb_id + ".pdb")]
    snakemake.output = [os.path.join(folder_path,  "mutated", publication, sample_pdb_id + ".log"), os.path.join(folder_path, "mutated", publication, sample_pdb_id + ".json")]
    snakemake.params = {}
    dms_info_file = os.path.join(project_root, "data/DMS/dms_curated.csv")
    info_df = pd.read_csv(dms_info_file)

    # limit to 5 mutations
    antibody, antigen = sample_pdb_id.split("_")
    groups = info_df.groupby(["publication", "antibody", "antigen"]).groups
    info_df = info_df.iloc[groups[(publication, antibody, antigen)]][:100].reset_index()

    snakemake.params["info_df"] = info_df
    snakemake.params["mutation_out_folder"] = folder_path + "/mutated"
    snakemake.threads = 2



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
    if len(mutation_code) == 0 or mutation_code == "original":
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

        # check if mutation is correct
        assert three2one_code[original_residue_name.upper()] == mutation["original_amino_acid"], \
            f"Residue {mutation['original_amino_acid']} at chain {mutation['chain']} and index {mutation['index']} is " \
            f"different from found residue {three2one_code[original_residue.upper()]}"

        mutate_residue(pose, pose.pdb_info().pdb2pose(mutation["chain"], mutation["index"]), mutation["new_amino_acid"], pack_radius=10, pack_scorefxn=pyrosetta.get_fa_scorefxn())
        perform_mutations.append(mutation["original_amino_acid"] + mutation["chain"] +
                                 str(mutation["index"]) + mutation["new_amino_acid"])

    return ";".join(perform_mutations)


def add_score(pose: Pose):
    score = scorefxn(pose)
    add_comment(pose, "rosetta_energy_score", str(score))


def perform_mutation(mutation_code: str):
    try:
        pose = pyrosetta.Pose()
        pose.detached_copy(original_pose)

        if not isinstance(mutation_code, str) and np.isnan(mutation_code):
            mutation_code = "original"
            pdb_out_path = os.path.join(mutated_pdb_folder, mutation_code + ".pdb")
            performed_mutations = []
        else:
            decoded_mutation = convert_mutations(mutation_code)
            performed_mutations = mutate(pose, decoded_mutation)

            pdb_out_path = os.path.join(mutated_pdb_folder, performed_mutations + ".pdb")

        if not os.path.exists(pdb_out_path):
            add_score(pose)
            dump_comment_pdb(pdb_out_path, pose)

        error = False
        error_msg = ""
    except Exception as e:
        error = True
        error_msg = e
        performed_mutations = ""

    return error, error_msg, mutation_code, performed_mutations


def get_existing_mutations(existing_mutations: set, given_mutations: str):
    mutations = given_mutations.split(";")

    remaining_mutations = []
    for mutation in mutations:
        if mutation[1:-1] in existing_mutations:
            remaining_mutations.append(mutation)

    return ";".join(remaining_mutations)


log_path = snakemake.output[0]
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
mutation_mapping_path = snakemake.output[1]
Path(mutation_mapping_path).parent.mkdir(parents=True, exist_ok=True)

file_path = snakemake.input[0]

publication = file_path.split("/")[-2]
antibody, antigen = file_path.split("/")[-1].split(".")[0].split("_")

mutated_pdb_folder = os.path.join(snakemake.params["mutation_out_folder"], publication, antibody + "_" + antigen)
Path(mutated_pdb_folder).mkdir(parents=True, exist_ok=True)

if os.path.exists(file_path.replace("interface_hull_pdbs", "prepared_pdbs").replace(".pdb", ".err")):
    # do not perform mutations if data generation process has thrown an error
    results = []
else:
    info_df = snakemake.params["info_df"]
    groups = info_df.groupby(["publication", "antibody", "antigen"]).groups
    mutations = info_df.iloc[groups[(publication, antibody, antigen)]]["mutation_code"].tolist()

    original_pose = load_pose(file_path)

    all_resi = set()
    for mutation in mutations:
        all_resi.update([mut[1:-1] for mut in mutation.split(";")])

    existing_resi = set()
    for resi in all_resi:
        chain = resi[0]
        index = int(resi[1:])
        original_residue = original_pose.pdb_rsd((chain, index))
        if original_residue is None:
            # Residue to mutate not in interface_hull
            # Pass this mutation and do not add it to the list
            continue
        else:
            existing_resi.add(chain + str(index))

    mutations2perform = info_df.iloc[groups[(publication, antibody, antigen)]]["mutation_code"].apply(lambda x: get_existing_mutations(existing_resi, x)).values.tolist()

    # skip existing mutations
    existing_mutations = set( [ code.split(".")[0] for code in os.listdir(mutated_pdb_folder) ])
    #existing_mutations = set()

    print(len(existing_mutations))
    print(len(mutations2perform))

    mutations = [(mutation_code, ) for mutation_code, mutation2perform in zip(mutations, mutations2perform) if mutation2perform not in existing_mutations]
    print(len(mutations))

    print(f"Performing {len(mutations)} mutations with {snakemake.threads} number of threads")

    results = submit_jobs(perform_mutation, mutations, snakemake.threads)
    #results = Parallel(n_jobs=snakemake.threads)(delayed(perform_mutation)(mutation) for mutation in tqdm(mutations))

mutation_mapping = {}
with open(log_path, "w") as f:
    f.write("mutation_code,performed_mutations,status,error_msg\n")
    for result in results:
        error, error_msg, mutation_code, performed_mutations = result
        mutation_mapping[mutation_code] = performed_mutations
        if error:
            f.write(f"{mutation_code},{performed_mutations},error,{error_msg}\n")
        else:
            f.write(f"{mutation_code},{performed_mutations},processed,\n")

with open(mutation_mapping_path, "w") as f:
    json.dump(mutation_mapping, f)

