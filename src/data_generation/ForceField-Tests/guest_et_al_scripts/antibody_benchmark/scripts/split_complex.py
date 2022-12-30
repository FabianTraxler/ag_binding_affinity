from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO,Select
import sys


class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = chains

    def accept_chain(self, chain):
        if chain.get_id().lower() in self.chains:
            return 1
        else:
            return 0


pdb_file_path = sys.argv[1]
ab_chains = sys.argv[2]
ag_chains = sys.argv[3]

pdb_id = pdb_file_path.split("/")[-1].split("_")[0]

parser = PDBParser()
io = PDBIO()
structure = parser.get_structure(pdb_id, f"{pdb_file_path}.pdb")
io.set_structure(structure)

io.save(pdb_id + ".rec.pdb", ChainSelect(ab_chains))

io.save(pdb_id + ".lig.pdb", ChainSelect(ag_chains))