
rule import_testing_mutations:
    """
    Call this manually using snakemake to import mutations into PDB

    This is an analysis to test whether our generated mutants resemble mutants for which we have ground truth structures.
    """
    output:
        wu17_5umn='input_pdb/wu17_in/c05_h3perth09:5UMN_VPGSGW_clean.pdb',
        madan21_6wwc='input_pdb/madan21_mutat_hiv/vfp1602_fp8v1:6WWC_S48K_clean.pdb',
        madan21_6wx2='input_pdb/madan21_mutat_hiv/vfp1602_fp8v1:6WX2_F60P_clean.pdb',
        # wu20_6np='input_pdb/wu20_differ_ha_h3_h1/cr9114_h3hk68/'
    shell: '''
        pdb_fetch 5UMN | pdb_tidy | pdb_selchain -A,E,F | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > {output.wu17_5umn}
        pdb_fetch 6WWC | pdb_tidy | pdb_selchain -A,B,C | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > {output.madan21_6wwc}
        pdb_fetch 6WX2 | pdb_tidy | pdb_selchain -A,B,C | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > {output.madan21_6wx2}
    '''
    # pdb_fetch 6NHP | pdb_tidy | pdb_selchain -TODO | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > {output.wu20_6nhp}
    # pdb_fetch 6NHQ | pdb_tidy | pdb_selchain -TODO | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > {output.wu20_6nhq}
    # pdb_fetch 6NHR | pdb_tidy | pdb_selchain -TODO | pdb_fixinsert | pdb_delhetatm | pdb_seg | pdb_chainbows > {output.wu20_6nhr}
