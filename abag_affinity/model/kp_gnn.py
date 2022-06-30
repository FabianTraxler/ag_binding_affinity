import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import HeteroData


class KpGNN(torch.nn.Module):

    def __init__(self, input_dim: int):
        super(KpGNN, self).__init__()
        # embed amino acids based on peptide bonded neighbors
        self.conv_1 = GATv2Conv(input_dim, 3, heads=3, dropout=0.5, edge_dim=1)
        # embed amino acids based on close residues from same protein
        self.conv_2 = GATv2Conv(3*3, 5, heads=3, dropout=0.5, edge_dim=1)
        self.conv_3 = GATv2Conv(3*5, 5, heads=5, dropout=0.5, edge_dim=1)

        self.edge_embed_1 = nn.Linear(2 * 5*5 + 1, 25)
        self.edge_embed_2 = nn.Linear(25, 10)
        self.edge_embed_3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.double()

    def forward(self, data: HeteroData):
        x = data["aa"].x

        peptide_bond_edge_idx = data["aa", "peptide_bond", "aa"].edge_index
        peptide_bond_edge_attr = data["aa", "peptide_bond", "aa"].edge_attr
        x = self.conv_1(x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        x = self.relu(x)

        same_protein_edge_idx = data["aa", "same_protein", "aa"].edge_index
        same_protein_edge_attr = data["aa", "same_protein", "aa"].edge_attr
        x = self.conv_2(x, same_protein_edge_idx, same_protein_edge_attr)
        x = self.relu(x)
        x = self.conv_3(x, same_protein_edge_idx, same_protein_edge_attr)
        x = self.relu(x)

        interface_edges = data["aa", "interface", "aa"].edge_index
        interface_distances = data["aa", "interface", "aa"].edge_attr.unsqueeze(1)

        edge_embeddings = torch.hstack([x[interface_edges[0]], x[interface_edges[1]], interface_distances])
        edge_embeddings = self.edge_embed_1(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_2(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_3(edge_embeddings)

        #affinity = global_mean_pool(edge_embeddings, torch.zeros(len(edge_embeddings)).long())
        affinity = torch.sum(edge_embeddings)

        return affinity