import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool, GlobalAttention
from torch_geometric.data import Data


class GraphConvAttention(torch.nn.Module):

    def __init__(self, input_dim: int):
        super(GraphConvAttention, self).__init__()

        self.conv_1 = GATv2Conv(input_dim, 3, heads=3, dropout=0.5, edge_dim=3)
        self.conv_2 = GATv2Conv(3*3, 5, heads=3, dropout=0.5, edge_dim=3)
        self.conv_3 = GATv2Conv(3*5, 5, heads=3, dropout=0.5, edge_dim=3)
        self.conv_4 = GATv2Conv(3*5, 5, heads=3, dropout=0.5, edge_dim=3)

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