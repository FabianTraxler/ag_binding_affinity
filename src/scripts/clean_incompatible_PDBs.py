from Bio.PDB.PDBParser import PDBParser
import os
import sys
import shutil

pdbs_dir = sys.argv[1]
clean_pdbs_dir = sys.argv[2]

if not os.path.exists(clean_pdbs_dir):
    os.mkdir(clean_pdbs_dir)

# Create a PDB parser object
parser = PDBParser(QUIET=True)

files = os.listdir(pdbs_dir)
pdb_ids = [f[:4] for f in files]
for i in range(len(files)):
    fname = os.path.join(pdbs_dir, files[i])
    structure = parser.get_structure(pdb_ids[i], fname)
    # Get the chains from the structure object
    chains = {chain.id for chain in structure.get_chains()}
    # Print the chains
    if {"L", "H", "A"} == chains:
        shutil.copyfile(os.path.join(pdbs_dir, pdb_ids[i] + '.pdb'), os.path.join(clean_pdbs_dir, pdb_ids[i] + '.pdb'))

