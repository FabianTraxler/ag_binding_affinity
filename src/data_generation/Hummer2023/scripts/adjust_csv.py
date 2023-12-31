import pandas as pd

summary_path = '../../../../results/synthethic_ddg/ddg.csv'
df = pd.read_csv(summary_path)
df.index = df.complex
df["filename"] = df.apply(lambda L: L.complex.replace('_', '_HL_A_')+'.pdb', axis=1)
df["mutation_code"] = df.apply(lambda L: L.complex.split('_')[-1], axis=1)
df["test"] = df.apply(lambda L: False, axis=1)

# Next we need to ensure that the column "-log(Kd) is there and contains correct values
# As Moritz is doing the convergence, I just hack here and compute the difference to a mean kd of 9
df["-log(Kd)"] = df.apply(lambda L: 9. + L.labels, axis=1)
df.to_csv(summary_path.replace(".csv","modified.csv"))
