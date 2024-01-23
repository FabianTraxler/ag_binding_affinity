from Bio import PDB
from pathlib import Path

def pdb_to_fasta(pdb_filenames, output_filename):
    pdb_parser = PDB.PDBParser(QUIET=True)
    with open(output_filename, 'w') as fasta_file:
        for pdb_filename in pdb_filenames:
            structure_id = pdb_filename.stem
            structure = pdb_parser.get_structure(structure_id, pdb_filename)
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    sequence = ''
                    for residue in chain:
                        if PDB.is_aa(residue):
                            try:
                                sequence += PDB.Polypeptide.protein_letters_3to1.get(residue.get_resname(), "X")
                            except AttributeError:
                                try:
                                    sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                                except KeyError:
                                    sequence += "X"
                    fasta_header = f'>{structure_id}_{chain_id}'
                    fasta_file.write(f'{fasta_header}\n')
                    fasta_file.write(f'{sequence}\n')

# List of PDB filenames
pdb_files = Path(snakemake.input.wt_pdb_dir).glob("????.pdb")

# Convert and save to FASTA
pdb_to_fasta(pdb_files, snakemake.output.pdb_seqs)
