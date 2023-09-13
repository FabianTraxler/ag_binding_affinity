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

    def forward(self, data: Dict, rel_temperature: float=0.2) -> Dict:
        """
        Relative temperature of 0.2, because in the scale [0-1], softmax leads to weak probabilities (close to 0.5)
        """
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
            "x": out_1["x"].flatten(),
            "x2": out_2["x"].flatten(),
            "difference": out_1["x"] - out_2["x"]
        }
        # Better always compute both x_prob and x as the difference and do not differ depending on affinity type!
        diff_1 = output["x"] - output["x2"]
        diff_2 = output["x2"] - output["x"]
        class_preds = torch.stack((diff_1, diff_2), dim=-1)
        # Computing the Probability based on Gaussian cdf:
        # We interpret rel_temperature as the standard deviation
        prob_1_ge_2 = torch.special.ndtr(diff_1 / 2**0.5 / rel_temperature)
        output["x_prob_cdf"] = torch.stack((prob_1_ge_2, 1-prob_1_ge_2), dim=-1)
        output["x_prob"] = torch.nn.functional.softmax(class_preds/rel_temperature, dim=-1)
        output["x_logit"] = torch.nn.functional.log_softmax(class_preds / rel_temperature, dim=-1)
        return output
