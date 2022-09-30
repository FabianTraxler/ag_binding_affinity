"""Submodule for Knowledge primed networks (networks that derive its strucutre from previous knowledge)"""
import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import HeteroData


class ResidueKpGNN(torch.nn.Module):
    """Graph Convolutional Neural network with attention mechanism
    Utilize structure properties to get better node embeddings
    1. Embed residues first based on peptide bonded neighbors
    2. Embed residues based on close residues in same protein
    3. Get Interface edges (residues closer than 5A and on different proteins) and embed them based on residues and distance
    4. Sum over all those edges to get binding affinity
    """

    def __init__(self, input_dim: int, device: torch.device):
        super(ResidueKpGNN, self).__init__()
        # embed amino acids based on peptide bonded neighbors
        self.peptide_conv_1 = GATv2Conv(input_dim, 5, heads=2, dropout=0.5, edge_dim=1)
        self.peptide_conv_2 = GATv2Conv(5*2, 5, heads=3, dropout=0.5, edge_dim=1)
        self.peptide_conv_3 = GATv2Conv(5*3, 10, heads=3, dropout=0.5, edge_dim=1)
        # embed amino acids based on close residues from same protein
        self.protein_conv_1 = GATv2Conv(10*3, 2, heads=2, dropout=0.5, edge_dim=1)
        self.protein_conv_2 = GATv2Conv(2*2, 2, heads=2, dropout=0.5, edge_dim=1)
        # embed each interface edge
        self.edge_embed_1 = nn.Linear(2 * (2 * 2 + 10 * 3) + 1, 25)
        self.edge_embed_2 = nn.Linear(25, 10)
        self.edge_embed_3 = nn.Linear(10, 1)

        self.relu = nn.LeakyReLU()
        self.double()
        self.device = device
        self.to(device)

    def get_edge_batches(self, num_edges: int, data: HeteroData):
        batch = torch.zeros(num_edges, dtype=torch.int64)
        if hasattr(data, "num_graphs") and data.num_graphs > 1:
            i = 0
            j = 0
            for ii, data_graph in enumerate(data.to_data_list()):
                if data_graph["node", "interface", "node"].edge_index.shape[1] == 0:
                    batch = torch.hstack([batch, torch.zeros(1, dtype=torch.int64)])
                    edge_embeddings = torch.vstack([edge_embeddings[:i, :], torch.zeros((1,1)), edge_embeddings[i:, :]])
                    j += 1
                else:
                    j += data_graph["node", "interface", "node"].edge_index.shape[1]
                batch[i:j] = ii
                i = j

        return batch

    def forward(self, data: HeteroData):
        x = data["node"].x

        # first node embedding part
        peptide_bond_edge_idx = data["node", "peptide_bond", "node"].edge_index
        peptide_bond_edge_attr = data["node", "peptide_bond", "node"].edge_attr
        peptide_x = self.peptide_conv_1(x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)
        peptide_x = self.peptide_conv_2(peptide_x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)
        peptide_x = self.peptide_conv_3(peptide_x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)
        # second node embedding part
        same_protein_edge_idx = data["node", "same_protein", "node"].edge_index
        same_protein_edge_attr = data["node", "same_protein", "node"].edge_attr
        protein_x = self.protein_conv_1(peptide_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)
        protein_x = self.protein_conv_2(protein_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)

        x = torch.hstack([peptide_x, protein_x])
        # get interface edges
        interface_edges = data["node", "interface", "node"].edge_index
        interface_distances = data["node", "interface", "node"].edge_attr.unsqueeze(1)
        # get interface edge embeddings
        edge_embeddings = torch.hstack([x[interface_edges[0]], x[interface_edges[1]], interface_distances])
        edge_embeddings = self.edge_embed_1(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_2(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_3(edge_embeddings)

        # handle batches
        batch = self.get_edge_batches(len(edge_embeddings), data)

        #affinity = global_mean_pool(edge_embeddings, torch.zeros(len(edge_embeddings)).long())
        affinity = global_add_pool(edge_embeddings, batch.to(self.device))

        return affinity


class AtomEdgeModel(torch.nn.Module):
    """Graph Convolutional Neural network with attention mechanism
    Utilize structure properties to get better node embeddings
    1. Embed atoms first based on atoms in same residue
    2. Embed atoms based on close (<2A) atoms on same protein
    3. Get Interface edges (atoms closer than 5A and on different proteins) and embed them based on atom embeddings and distance
    4. Sum over all those edges to get binding affinity
    """

    def __init__(self, input_dim: int, edge_dim: int, device: torch.device):
        super(AtomEdgeModel, self).__init__()
        # embed amino acids based on peptide bonded neighbors
        self.residue_conv_1 = GATv2Conv(input_dim, 50, heads=3, dropout=0.5, edge_dim=1)

        # embed amino acids based on close residues from same protein
        self.protein_conv_1 = GATv2Conv(50*3, 75, heads=2, dropout=0.5, edge_dim=1)
        self.protein_conv_2 = GATv2Conv(75*2, 50, heads=2, dropout=0.5, edge_dim=1)
        self.protein_conv_3 = GATv2Conv(50*2, 100, heads=1, dropout=0.5, edge_dim=1)
        # embed each interface edge

        node_embedding_dim = input_dim + 50 * 3 + 100 # input embedding, after residue layer, after protein layers

        self.edge_embed_1 = nn.Linear(2 * node_embedding_dim + 1, 100)
        self.edge_embed_2 = nn.Linear(100, 10)
        self.edge_embed_3 = nn.Linear(10, 1)

        self.relu = nn.LeakyReLU()
        self.double()
        self.device = device
        self.to(device)

    def get_edge_batches(self, num_edges: int, data: HeteroData):
        batch = torch.zeros(num_edges, dtype=torch.int64)
        if hasattr(data, "num_graphs") and data.num_graphs > 1:
            i = 0
            j = 0
            for ii, data_graph in enumerate(data.to_data_list()):
                if data_graph["node", "interface", "node"].edge_index.shape[1] == 0:
                    batch = torch.hstack([batch, torch.zeros(1, dtype=torch.int64)])
                    edge_embeddings = torch.vstack([edge_embeddings[:i, :], torch.zeros((1,1)), edge_embeddings[i:, :]])
                    j += 1
                else:
                    j += data_graph["node", "interface", "node"].edge_index.shape[1]
                batch[i:j] = ii
                i = j

        return batch

    def forward(self, data: HeteroData):
        x = data["node"].x.double()

        # first node embedding part
        peptide_bond_edge_idx = data["node", "same_residue", "node"].edge_index
        peptide_bond_edge_attr = data["node", "same_residue", "node"].edge_attr
        peptide_x = self.residue_conv_1(x, peptide_bond_edge_idx, peptide_bond_edge_attr)
        peptide_x = self.relu(peptide_x)

        # second node embedding part
        same_protein_edge_idx = data["node", "same_protein", "node"].edge_index
        same_protein_edge_attr = data["node", "same_protein", "node"].edge_attr
        protein_x = self.protein_conv_1(peptide_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)
        protein_x = self.protein_conv_2(protein_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)
        protein_x = self.protein_conv_3(protein_x, same_protein_edge_idx, same_protein_edge_attr)
        protein_x = self.relu(protein_x)

        x = torch.hstack([x, peptide_x, protein_x])
        # get interface edges
        interface_edges = data["node", "interface", "node"].edge_index
        interface_distances = data["node", "interface", "node"].edge_attr.unsqueeze(1)
        # get interface edge embeddings
        edge_embeddings = torch.hstack([x[interface_edges[0]], x[interface_edges[1]], interface_distances])


        edge_embeddings = self.edge_embed_1(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_2(edge_embeddings)
        edge_embeddings = self.relu(edge_embeddings)
        edge_embeddings = self.edge_embed_3(edge_embeddings)

        # handle batches
        batch = self.get_edge_batches(len(edge_embeddings), data)

        #affinity = global_mean_pool(edge_embeddings, torch.zeros(len(edge_embeddings)).long())
        affinity = global_add_pool(edge_embeddings, batch.to(self.device))

        return affinity