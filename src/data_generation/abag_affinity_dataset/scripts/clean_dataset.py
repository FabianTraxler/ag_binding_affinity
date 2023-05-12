from abag_affinity.utils.pdb_processing import clean_and_tidy_pdb
import tqdm


from pathlib import Path

pdb_folder = Path(snakemake.input[0])
cleaned_pdb_folder = Path(snakemake.output[0])


for pdb_file in tqdm.tqdm(pdb_folder.glob("*.pdb")):
    clean_and_tidy_pdb("", pdb_file, cleaned_pdb_folder)  # fix_insert is applied for now
