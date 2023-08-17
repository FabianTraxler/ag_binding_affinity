from typing import Dict, List
from pyrosetta.toolbox.mutants import mutate_residue
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose
from collections import defaultdict, deque

def get_chain_info(metadata, publication, antibody, antigen):
    # Set chain-translation
    publication_data = metadata[publication]
    for complex in publication_data["complexes"]:
        if complex["antigen"]["name"] == antigen and complex["antibody"]["name"] == antibody:
            # Inverted chain infos
            return {v: k for k,v in {**complex["pdb"]["chains"]["antibody"], **complex["pdb"]["chains"]["antigen"]}.items()}
    else:
        raise ValueError("antibody antigen complex not found")


# Chain conversion
def substitute_chain(metadata, mutation_str, publication, antibody, antigen):
    """
    Replace the chain ID according to metadata
    """
    chain_info = get_chain_info(metadata, publication, antibody, antigen)
    mutation_codes = [list(s) for s in mutation_str.split(";")]
    for mc in mutation_codes:
        mc[1] = chain_info[mc[1]]
    return ";".join(["".join(l) for l in mutation_codes])

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


def mutate(pose: Pose, mutations: List[Dict]):
    three2one_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


    for mutation in mutations:
        original_residue = pose.pdb_rsd((mutation["chain"], mutation["index"]))
        if original_residue is None:
            raise RuntimeError(f"Residue in chain {mutation['chain']} at index {mutation['index']} not found")
        original_residue_name = original_residue.name()
        if original_residue_name.upper()[:3] not in three2one_code:
            # check if mutation is correct
            raise RuntimeError(f"Residue name in chain {mutation['chain']} at index {mutation['index']} with name "
                          f"{original_residue_name.upper()} cannot be converted to one-letter-code")
        if three2one_code[original_residue_name.upper()[:3]] != mutation["original_amino_acid"]:
            # check if mutation is correct
            raise RuntimeError(f"Original residue in chain {mutation['chain']} at index {mutation['index']} does not match "
                          f"found residue {three2one_code[original_residue_name.upper()[:3]]} != {mutation['original_amino_acid']}")

        mutate_residue(pose, pose.pdb_info().pdb2pose(mutation["chain"], mutation["index"]), mutation["new_amino_acid"], pack_radius=10, pack_scorefxn=pyrosetta.get_fa_scorefxn())
