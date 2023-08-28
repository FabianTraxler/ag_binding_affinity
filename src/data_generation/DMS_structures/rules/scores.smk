SCORE_PATH = dms_results / "structure_prediction_matching" / "{publication}" / "{complex}_{rank}.score"
rule match_scores:
    input:
        initial_pdb=dms_results / "prepared_renamed_pdbs" / "{publication}" / "{complex}.pdb"
        af_pdb=str(AF_OUTPUT_PATH).format(rank="{rank}"),
    output:
        output=SCORE_PATH
    run:
        from Bio.PDB.PDBParser import PDBParser
        from Bio.PDB.Superimposer import Superimposer

        # Parse the PDB files
        parser = PDBParser(PERMISSIVE=1)
        structure1 = parser.get_structure("input", input.initial_pdb)
        structure2 = parser.get_structure("output", input.af_pdb)

        # Align the structures using the alpha-carbon atoms
        model1 = structure1[0]
        model2 = structure2[0]
        atoms1 = [atom for atom in model1.get_atoms() if atom.name == "CA"]
        atoms2 = [atom for atom in model2.get_atoms() if atom.name == "CA"]

        # Calculate the RMSD
        superimposer = Superimposer()
        superimposer.set_atoms(atoms1, atoms2)
        superimposer.apply(model2)

        with open(output[0], 'w') as f:
            f.write(str(superimposer.rms))

rule combine_scores:
    input:
        scores=[str(SCORE_PATH).format(publication=publication, complex=complex, rank=rank) for (publication, complex), rank in itertools.product(PUBLICATION_COMPLEXES, range(1, 6))],
        structure_prediction=[str(AF_OUTPUT_PATH).format(rank=rank).format(publication=publication, complex=complex) for (publication, complex), rank in itertools.product(PUBLICATION_COMPLEXES, range(1, 6))]
    output:
        dms_results / "structure_prediction_matching" / 'all_scores.csv'
    run:
        from pathlib import Path
        import pandas as pd
        import subprocess
        df = []
        for score_f, model_f in zip(input.scores, input.structure_prediction):
            # Get the underlying model
            path = Path(score_f)
            df.append(dict(
                publication=path.parent.name,
                complex=path.stem.rsplit('_', maxsplit=1)[0],
                model_i=int(subprocess.run(f'realpath {model_f} | grep -o model_.', shell=True, capture_output=True).stdout.strip().decode('utf-8')[-1]),
                rank=path.stem.rsplit('_', maxsplit=1)[1],
                score=float(path.read_text())
            ))
        df = pd.DataFrame(df)
        df.to_csv(output[0])

rule plot_scores:
    input:
        scores=rules.combine_scores.output[0]
    output:
        png=dms_results / "structure_prediction_matching" / 'all_scores.png',
        svg=dms_results / "structure_prediction_matching" / 'all_scores.svg',
    run:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd


        fig, ax = plt.subplots(figsize=(3, 3))

        df = pd.read_csv(input.scores, index_col=0)
        sns.boxplot(data=df, x='rank', y='score')
        fig.savefig(output["png"], bbox_inches='tight')
        fig.savefig(output["svg"], bbox_inches='tight')
