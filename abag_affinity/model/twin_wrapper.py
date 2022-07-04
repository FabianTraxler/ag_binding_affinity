import torch
from torch_geometric.data import Data


class TwinWrapper(torch.nn.Module):
    def __init__(self, backbone_net: torch.nn.Module):
        super(TwinWrapper, self).__init__()  # pre 3.3 syntax
        self.backbone_net = backbone_net

    def forward(self, data_1: Data, data_2: Data):
        out_1 = self.backbone_net.encode(data_1)
        out_2 = self.backbone_net.encode(data_2)

        return out_1, out_2
