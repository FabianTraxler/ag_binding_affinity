"""Submodule for Graph Convolutional Neural Networks trained from scratch using graph pooling to predict binding affinity"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool, GlobalAttention
from torch_geometric.data import Data


class GraphConv(torch.nn.Module):
    """Graph Convolutional Neural Network with attention mechanism for node embedding
    With graph max-pooling operation to predict binding affinity
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int):
        super(GraphConv, self).__init__()

        self.conv_1 = GATv2Conv(node_feat_dim, 25, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_2 = GATv2Conv(25*3, 50, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_3 = GATv2Conv(50*3, 100, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_4 = GATv2Conv(100*3, 300, heads=1, dropout=0.5, edge_dim=edge_feat_dim)

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(300*1, 10)
        self.fc_2 = nn.Linear(10, 1)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv_3(x, edge_index, edge_attr)
        x = self.conv_4(x, edge_index, edge_attr)
        x = self.relu(x)

        x = global_max_pool(x, data.batch)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x


class FSGraphConv(torch.nn.Module):
    """Graph Convolution with attention to get node embeddings
    Fixed sized input graphs and therefore no graph pooling necessary
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, num_nodes: int):
        super(FSGraphConv, self).__init__()

        self.num_nodes = num_nodes

        self.conv_1 = GATv2Conv(node_feat_dim, 3, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_2 = GATv2Conv(3*3, 5, heads=4, dropout=0.5, edge_dim=edge_feat_dim)


        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(4*5*num_nodes, 100)
        self.fc_2 = nn.Linear(100, 10)
        self.fc_3 = nn.Linear(10, 1)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.relu(x)

        x = x.view(data.num_graphs, self.num_nodes * 4 * 5)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        return x


class GraphConvAttention(torch.nn.Module):
    """Graph convolution with attention mechanism for node embedding
    Graph pooling done by global attention mechanism to predict binding affinity
    """
    def __init__(self, inode_feat_dim: int, edge_feat_dim: int):
        super(GraphConvAttention, self).__init__()

        self.conv_1 = GATv2Conv(inode_feat_dim, 3, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_2 = GATv2Conv(3*3, 5, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_3 = GATv2Conv(3*5, 5, heads=3, dropout=0.5, edge_dim=edge_feat_dim)
        self.conv_4 = GATv2Conv(3*5, 5, heads=3, dropout=0.5, edge_dim=edge_feat_dim)

        self.relu = nn.LeakyReLU()

        self.global_attention = GlobalAttention(nn.Linear(3*5, 1))

        self.fc_1 = nn.Linear(3*5, 10)
        self.fc_2 = nn.Linear(10, 1)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv_3(x, edge_index, edge_attr)
        x = self.conv_4(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.global_attention(x, batch=data.batch)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x