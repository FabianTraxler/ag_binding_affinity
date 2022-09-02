import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import HeteroData


class KpGNN(torch.nn.Module):

    def __init__(self, input_dim: int, device: torch.device):
        super(KpGNN, self).__init__()
        # embed amino acids based on peptide bonded neighbors
        self.peptide_conv_1 = GATv2Conv(input_dim, 5, heads=2, dropout=0.5, edge_dim=1)
        self.peptide_conv_2 = GATv2Conv(5*2, 5, heads=3, dropout=0.5, edge_dim=1)
        self.peptide_conv_3 = GATv2Conv(5*3, 10, heads=3, dropout=0.5, edge_dim=1)
        # embed amino acids based on close residues from same protein
        self.protein_conv_1 = GATv2Conv(10*3, 2, heads=2, dropout=0.5, edge_dim=1)
        self.protein_conv_2 = GATv2Conv(2*2, 2, heads=2, dropout=0.5, edge_dim=1)

        self.edge_embed_1 = nn.Linear(2 * (2 * 2 + 10 * 3) + 1, 25)
        self.edge_embed_2 = nn.Linear(25, 10)
        self.edge_embed_3 = nn.Linear(10, 1)

        self.relu = nn.LeakyReLU()
        self.double()
        self.device = device
        self.to(device)

    def forward(self, data: HeteroData):
        x = data["aa"].x

        peptide_bond_edge_idx = data["aa", "peptide_bond", "aa"].edge_index
        peptide_bond_edge_attr = data["aa", "peptide_bond", "aa"].edge_attr
        peptide_x = self.peptide_conv_1(x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)
        peptide_x = self.peptide_conv_2(peptide_x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)
        peptide_x = self.peptide_conv_3(peptide_x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)

        same_protein_edge_idx = data["aa", "same_protein", "aa"].edge_index
        same_protein_edge_attr = data["aa", "same_protein", "aa"].edge_attr
        protein_x = self.protein_conv_1(peptide_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)
        protein_x = self.protein_conv_2(protein_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)

        x = torch.hstack([peptide_x, protein_x])

        interface_edges = data["aa", "interface", "aa"].edge_index
        interface_distances = data["aa", "interface", "aa"].edge_attr.unsqueeze(1)

        edge_embeddings = torch.hstack([x[interface_edges[0]], x[interface_edges[1]], interface_distances])
        edge_embeddings = self.edge_embed_1(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_2(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_3(edge_embeddings)

        batch = torch.zeros(len(edge_embeddings), dtype=torch.int64)
        if hasattr(data, "num_graphs") and data.num_graphs > 1:
            i = 0
            j = 0
            for ii, data_graph in enumerate(data.to_data_list()):
                if data_graph["aa", "interface", "aa"].edge_index.shape[1] == 0:
                    batch = torch.hstack([batch, torch.zeros(1, dtype=torch.int64)])
                    edge_embeddings = torch.vstack([edge_embeddings[:i, :], torch.zeros((1,1)), edge_embeddings[i:, :]])
                    j += 1
                else:
                    j += data_graph["aa", "interface", "aa"].edge_index.shape[1]
                batch[i:j] = ii
                i = j

        #affinity = global_mean_pool(edge_embeddings, torch.zeros(len(edge_embeddings)).long())
        affinity = global_add_pool(edge_embeddings, batch.to(self.device))

        return affinity