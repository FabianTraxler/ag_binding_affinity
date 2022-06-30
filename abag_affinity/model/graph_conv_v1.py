import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool
from torch_geometric.data import Data


class GraphConv(torch.nn.Module):

    def __init__(self, input_dim: int):
        super(GraphConv, self).__init__()

        self.conv_1 = GATv2Conv(input_dim, 3, heads=3, dropout=0.5, edge_dim=3)
        self.conv_2 = GATv2Conv(3*3, 5, heads=4, dropout=0.5, edge_dim=3)

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(4*5, 10)
        self.fc_2 = nn.Linear(10, 1)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.relu(x)

        x = global_max_pool(x, data.batch)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x