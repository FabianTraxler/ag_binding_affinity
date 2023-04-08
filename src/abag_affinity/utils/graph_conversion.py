import networkx as nx
import dgl
import torch_geometric as pyg
from typing import Union


def read_graph(file_path: str, type: str) -> Union[dgl.DGLGraph, pyg.data.Data]:
    with open(file_path, "rb") as f:
        dgl_graph = pickle.load(f)

    if type == "DGL":
        return dgl_graph
    elif type == "PyG":
        return ""
    else:
        raise ValueError("Please specify an available Graph Format (DGL, PyG)")


def save_graph(graph: Union[dgl.DGLGraph, pyg.data.Data], file_path: str):
    if isinstance(graph, dgl.DGLGraph):
        dgl_graph = graph
    elif isinstance(graph, pyg.data.Data):
        dgl_graph = dgl.DGLGraph()
    else:
        raise ValueError("Please input an available Graph Format (DGL, PyG)")

    with open(file_path, "wb") as f:
        pickle.dump(dgl_graph, f)


if __name__ == "__main__":
    import pickle
    from dgl.data.utils import save_graphs

    with open("/home/fabian/Desktop/Uni/Masterthesis/ag_binding_affinity/results/processed_graphs/atom/Dataset_v1/deeprefine/1p2c.pickle", "rb") as f:
        egr_dict = pickle.load(f)

    dgl_graph = egr_dict["graph"]

    for feat in ['atom_type', 'x_pred', 'labeled', 'interfacing', 'covalent_radius', 'chain_id', 'residue_number', 'surf_prox', 'is_ca_atom', 'dihedral_angles']:
        del dgl_graph.ndata[feat]

    for feat in ['pos_enc', 'rel_pos', 'r', 'w', 'bond_type', 'in_same_chain', 'rel_geom_feats']:
        del dgl_graph.edata[feat]

    save_graphs("./data.bin", [dgl_graph])
    save_graph(dgl_graph, "test.graph")

    dgl_graph2 = read_graph("test.graph", "DGL")
    a = 0