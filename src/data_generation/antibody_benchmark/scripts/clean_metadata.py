from guided_protein_diffusion.datasets.abdb import ENUM_ANTIGEN
from guided_protein_diffusion.datasets.dataset import load_protein
import pandas as pd
from pathlib import Path
import string
import numpy as np
from typing import Dict
from collections import defaultdict, deque

if "snakemake" not in globals(): # use fake snakemake object for debugging
    snakemake = type('', (), {})()
    snakemake.input = {"xlsx": "~/guided-protein-diffusion/modules/ag_binding_affinity/resources/antibody_benchmark/antibody_benchmark_cases.xlsx",
                       "pdb_dir": "~/guided-protein-diffusion/modules/ag_binding_affinity/resources/AbDb/pdbs"}
    snakemake.output = {"csv": "~/guided-protein-diffusion/modules/ag_binding_affinity/results/antibody_benchmark/benchmark.csv"}


from guided_protein_diffusion.utils.interact import init_interactive_environment
init_interactive_environment(
    ["--dataset", "abdb", "--openfold_time_injection_alpha", "0.0", "--antigen_conditioning"]
)  # implies --testing
out_path = Path(snakemake.output["csv"]).expanduser()
out_path.parent.mkdir(parents=True, exist_ok=True)
file_path = snakemake.input["xlsx"]

summary_df = pd.read_excel(file_path)
print(file_path)
summary_df = summary_df.replace(" ", np.nan)
summary_df = summary_df[summary_df["Kd (nM)"].notnull()]

summary_df["pdb"] = summary_df["Complex PDB"].apply(lambda x: x.split("_")[0].lower())

summary_df["filename"] = summary_df["pdb"].apply(lambda x: x.upper() + ".pdb")

summary_df["-log(Kd)"] = summary_df["Kd (nM)"].apply(lambda x: -np.log10(x * 1e-9))

summary_df["delta_g"] = summary_df["ΔG (kcal/mol)"]


def get_chains(complex_name: str) -> Dict:
    ab_chains, ag_chains = complex_name.split("_")[-1].split(":")
    chain_info = {}
    for ab_chain, curated_chain in zip(ab_chains.strip(), "HL"):
        chain_info[ab_chain] = curated_chain
    for ag_chain, curated_chain in zip(ag_chains.strip(), string.ascii_uppercase):
        chain_info[ag_chain] = curated_chain
    if "L" in chain_info and "H" in chain_info:
        del chain_info["L"]
        del chain_info["H"]

    # manually add chains since there are errors in the metadata file
    if complex_name.split("_")[0].lower() == "5kov":
        chain_info["L"] = "L"

    def order_substitutions(substitutions):
        """
        Order substiutions to avoid chain overlaps (and thereby loss of chain information)
        """
        # Create a dependency graph with nodes as keys and values
        graph = defaultdict(list)
        for src, dest in substitutions.items():
            graph[src].append(dest)

        # Perform a topological sorting on the graph
        sorted_nodes = []
        visited = set()
        stack = deque()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    visit(neighbor)
                stack.appendleft(node)

        for node in list(graph.keys()):
            visit(node)

        # Apply substitutions in the sorted order
        result = {}
        for node in reversed(stack):
            if node in substitutions:
                result[node] = substitutions[node]

        return result

    return order_substitutions(chain_info)

summary_df["chain_infos"] = summary_df["Complex PDB"].apply(get_chains)
summary_df["validation"] = 0
summary_df["test"] = True

summary_df = summary_df[["pdb", "filename", "-log(Kd)", "Kd (nM)", "delta_g", "chain_infos", "validation", "test"]]
summary_df.index = summary_df["pdb"]

# Check for which PDBs we are able to load antigens (only take those for the benchmark set)
valid_pdbs = []
for pdb_id in summary_df.index:
    try:
        prot = load_protein(Path(snakemake.input["pdb_dir"]).expanduser() / f"{pdb_id.upper()}_1.pdb", max_antigen_length=None)  # allow any size
        if (prot["context_chain_type"] == ENUM_ANTIGEN).any():
            valid_pdbs.append(pdb_id)
    except FileNotFoundError:
        pass

print(f"{len(valid_pdbs)} for {len(summary_df)} were valid ((antibody+)antigen could be loaded)")
summary_df = summary_df.loc[valid_pdbs]

summary_df.index.name = ""
summary_df.to_csv(out_path)
