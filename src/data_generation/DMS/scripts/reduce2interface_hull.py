# Inspiration from PyRosetta tutorial notebooks
# https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
from pathlib import Path
from typing import Dict, List
import yaml
import os
import numpy as np
from biopandas.pdb import PandasPdb
import scipy.spatial as sp


three2one_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


if "snakemake" not in globals(): # use fake snakemake object for debugging
    import pandas as pd

    project_root =  "../../../../" # three directories above
    folder_path = os.path.join(project_root, "results/DMS")

    publication = "madan21_mutat_hiv"
    sample_pdb_id = "vfp1602_fp8v1"

    snakemake = type('', (), {})()
    snakemake.input = [os.path.join(folder_path, "prepared_pdbs", publication, sample_pdb_id + ".pdb")]
    snakemake.output = [os.path.join(folder_path,  "interface_hull_pdbs", publication, sample_pdb_id + ".pdb")]

    metadata_file = os.path.join(project_root, "data/metadata_dms_studies.yaml")

    snakemake.params = {}
    snakemake.params["metadata_file"] = metadata_file
    snakemake.params["project_root"] = project_root
    snakemake.params["interface_size"] = 5
    snakemake.params["interface_hull_size"] = 10
    snakemake.threads = 2


def get_complex_metadata(publication:str, antibody: str, antigen: str, metadata: Dict) -> Dict:
    publication_data = metadata[publication]
    for complex in publication_data["complexes"]:
        if complex["antigen"]["name"] == antigen and complex["antibody"]["name"] == antibody:
            return complex["pdb"]

    raise RuntimeError(f"Complex not found in Metadata: {publication}, {antibody}, {antigen}")


def get_chain_infos(complex_metadata: Dict) -> List:
    chain_infos = []
    for chains in complex_metadata["chains"].values():
        chain_infos.append(chains)

    assert len(chain_infos) == 2, "More than 2 proteins found in complex"
    return chain_infos


def reduce2interface_hull(pdb_filepath: str, out_path: str,
                          prot_chains: List,
                          interface_size: int, interface_hull_size: int):
    """ Reduce PDB file to only contain residues in interface-hull

    Interface hull defines as class variable

    1. Get distances between atoms
    2. Get interface atoms
    3. get all atoms in hull around interface
    4. expand to all resiudes that have at least 1 atom in interface hull

    Args:
        file_name: Name of the file
        pdb_filepath: Path of the original pdb file
        chain_infos: Dict with information which chain belongs to which protein (necessary for interface detection)

    Returns:
        str: path to interface pdb file
    """

    pdb = PandasPdb().read_pdb(pdb_filepath)
    atom_df = pdb.df['ATOM']

    #atom_df["chain_id"] = atom_df["chain_id"].str.upper()

    # calcualte distances
    coords = atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    distances = sp.distance_matrix(coords, coords)

    prot_1_idx = atom_df[atom_df["chain_id"].isin(prot_chains[0])].index.to_numpy().astype(int)
    prot_2_idx = atom_df[atom_df["chain_id"].isin(prot_chains[1])].index.to_numpy().astype(int)

    # get interface
    abag_distance = distances[prot_1_idx, :][:, prot_2_idx]
    interface_connections = np.where(abag_distance < interface_size)
    prot_1_interface = prot_1_idx[np.unique(interface_connections[0])]
    prot_2_interface = prot_2_idx[np.unique(interface_connections[1])]

    # get interface hull
    interface_atoms = np.concatenate([prot_1_interface, prot_2_interface])
    interface_hull = np.where(distances[interface_atoms, :] < interface_hull_size)[1]
    interface_hull = np.unique(interface_hull)

    # use complete residues if one of the atoms is in hull
    interface_residues = atom_df.iloc[interface_hull][["chain_id", "residue_number"]].drop_duplicates()
    interface_df = atom_df.merge(interface_residues)

    assert len(interface_df) > 0, f"No atoms after cleaning in file: {pdb_filepath}"

    pdb.df['ATOM'] = interface_df
    pdb.to_pdb(path=out_path,
               records=["ATOM"],
               gz=False,
               append_newline=True)



out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

file_path = snakemake.input[0]
interface_size = snakemake.params["interface_size"]
interface_hull_size = snakemake.params["interface_hull_size"]

antibody, antigen = out_path.split("/")[-1].split(".")[0].split("_")
publication = out_path.split("/")[-2]
if "mason21" in publication:
    publication = "mason21_optim_therap_antib_by_predic"

with open(snakemake.params["metadata_file"], "r") as f:
    metadata = yaml.safe_load(f)

complex_metadata = get_complex_metadata(publication, antibody, antigen, metadata)
chain_infos = get_chain_infos(complex_metadata)

reduce2interface_hull(file_path, out_path, chain_infos, interface_size, interface_hull_size)

