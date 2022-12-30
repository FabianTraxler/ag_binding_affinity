"""Utilities to read PDB files"""
from pathlib import Path
from typing import Dict, Tuple, Union

from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure

parser = PDBParser(PERMISSIVE=3)

pdb_list = PDBList()

def read_file(structure_id: str, path: Union[str, Path]) -> Tuple[Structure, Dict]:
    """ Read a PDB file and return the structure and header

    Args:
        structure_id: PDB ID
        path: Path of the PDB file

    Returns:
        Tuple: Structure (Bio.PDB object), header (Dict)
    """
    try:
        structure = parser.get_structure(structure_id, str(path))
        header = parser.get_header()
    except Exception as e:
        raise RuntimeError(f"Could not load pdb_file {path}: {e}")

    return structure, header

