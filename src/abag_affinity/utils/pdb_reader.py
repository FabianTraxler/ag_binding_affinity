"""Utilities to read PDB files"""
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBList import PDBList
from Bio.PDB.Structure import Structure
from typing import Tuple, Dict

parser = PDBParser(PERMISSIVE=3)

pdb_list = PDBList()

def read_file(structure_id: str, path: str) -> Tuple[Structure, Dict]:
    """ Read a PDB file and return the structure and header

    Args:
        structure_id: PDB ID
        path: Path of the PDB file

    Returns:
        Tuple: Structure (Bio.PDB object), header (Dict)
    """
    structure = parser.get_structure(structure_id, path)
    header = parser.get_header()
    return structure, header
