rule trimmed_csvs:
    input:
        expand(out_folder / "trimmed_csvs" / "{publication}.csv", publication=publications)

rule remove_dms_without_pdb:
    """
    This is a workaround to trim the CSV files to only contain data points for which the corresponding PDB exists
    """
    input:
        out_folder / "{publication}.csv"
    output:
        out_folder / "trimmed_csvs" / "{publication}.csv"
    run:
        df = pd.read_csv(input[0], index_col=0)

        file_exists = df.filename.apply(lambda fn: (MUTATED_PDB_FOLDER / fn).exists())
        df.loc[file_exists].to_csv(output[0])

