from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBList import PDBList

parser = PDBParser(PERMISSIVE=0)

pdb_list = PDBList()

def read_file(structure_id: str, path: str):
    structure = parser.get_structure(structure_id, path)
    header = parser.get_header()
    return structure, header


def get_pdb_info(pdb_id: str):
    pdb_list.retrieve_pdb_file(pdb_id)