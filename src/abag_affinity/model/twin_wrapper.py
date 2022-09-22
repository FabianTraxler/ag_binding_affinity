"""Submodule providing a wrapper class for training with relative binding affinities"""
import torch
from torch_geometric.data import Data


class TwinWrapper(torch.nn.Module):
    """Wrapper class to get relative binding affinity of two graphs"""
    def __init__(self, backbone_net: torch.nn.Module):
        super(TwinWrapper, self).__init__()
        self.backbone_net = backbone_net

    def forward(self, data_1: Data, data_2: Data):
        out_1 = self.backbone_net(data_1)
        out_2 = self.backbone_net(data_2)
        return out_1 - out_2
