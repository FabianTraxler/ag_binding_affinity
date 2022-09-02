import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class FSGraphConv(torch.nn.Module):

    def __init__(self, input_dim: int, num_nodes: int):
        super(FSGraphConv, self).__init__()

        self.num_nodes = num_nodes

        self.conv_1 = GATv2Conv(input_dim, 3, heads=3, dropout=0.5, edge_dim=3)
        self.conv_2 = GATv2Conv(3*3, 5, heads=4, dropout=0.5, edge_dim=3)


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