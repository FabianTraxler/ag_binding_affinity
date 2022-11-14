"""Submodule providing a wrapper class for training with relative binding affinities"""
import torch
from typing import List, Dict

from .gnn_model import AffinityGNN

class TwinWrapper(torch.nn.Module):
    """Wrapper class to get relative binding affinity of two graphs"""
    def __init__(self, backbone_net: AffinityGNN):
        super(TwinWrapper, self).__init__()
        self.backbone_net = backbone_net
        self.device = self.backbone_net.device

    def forward(self, data: Dict) -> Dict:
        data_1, data_2 = data["input"]

        # put data on device before pass and then delete again
        # load to device
        data_1["graph"] = data_1["graph"].to(self.device)
        if "deeprefine_graph" in data_1:
            data_1["deeprefine_graph"] = data_1["deeprefine_graph"].to(self.device)
        out_1 = self.backbone_net(data_1)

        data_2["graph"] = data_2["graph"].to(self.device)
        if "deeprefine_graph" in data_2:
            data_2["deeprefine_graph"] = data_2["deeprefine_graph"].to(self.device)
        out_2 = self.backbone_net(data_2)

        output = {
            "relative": data["relative"],
            "affinity_type": data["affinity_type"],
            "x1": out_1["x"].flatten(),
            "x2": out_2["x"].flatten(),
        }

        if data["affinity_type"] == "-log(Kd)":
            output["x"] = out_1["x"] - out_2["x"]
        elif data["affinity_type"] == "E":
            diff_1 = output["x1"] - output["x2"]
            diff_2 = output["x2"] - output["x1"]
            class_preds = torch.stack((diff_1, diff_2)).T
            output["x_prob"] = torch.nn.functional.softmax(class_preds)
            output["x"] = torch.argmax(output["x_prob"], dim=1)
        else:
            raise ValueError(f"Wrong affinity type given - expected one of (-log(Kd), E) but got {data['affinity_type']}")
        return output
