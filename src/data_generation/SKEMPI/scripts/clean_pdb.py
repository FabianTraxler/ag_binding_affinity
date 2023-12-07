import sys

from Bio.PDB import PDBParser, PDBIO, Select
import pandas as pd


from abag_affinity.utils.pdb_processing import clean_and_tidy_pdb


# Initialize a parser
parser = PDBParser()
io = PDBIO()


def truncate_ab(pdb_fn, pdb_out_fn, max_l=110, max_h=120):
    # Read the input PDB file
    structure = parser.get_structure("pdb_fn", pdb_fn)

    # Create an output PDB file
    with open(pdb_out_fn, "w") as output_file:

        # Initialize Select class
        class FvSelect(Select):
            def accept_residue(self, residue):
                residue_id = residue.get_id()[1]
                chain = residue.parent.id

                if chain == "H":
                    if residue_id <= max_h:
                        return True
                    else:
                        return False
                elif chain == "L":
                    if residue_id <= max_l:
                        return True
                    else:
                        return False
                else:
                    return True


        io.set_structure(structure)
        io.save(output_file, FvSelect())


# load skempi datasets
skempi_df = pd.read_csv(sys.argv[3], index_col=0)
chain_infos = eval(skempi_df.loc[f"{sys.argv[4].lower()}-original", "chain_infos"])
# Control whether there is a chain overwrite

for i, target in enumerate(chain_infos.values()):
    if target in list(chain_infos.keys())[i+1:]:
        raise ValueError(f"Chain {target} is overwritten: {chain_infos}")

clean_and_tidy_pdb("_", sys.argv[1], sys.argv[2], rename_chains=chain_infos)
truncate_ab(sys.argv[2], sys.argv[2])
