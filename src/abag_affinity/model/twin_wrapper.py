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

        output = {"relative": data["relative"]}

        if "uncertainty" in out_1.keys():
            # Uncertainty is part of model prediction (might be fixed)
            rel_temperature = (out_1["uncertainty"] + out_2["uncertainty"]).flatten() /2.
            output["uncertainty"] = out_1["uncertainty"]
            output["uncertainty2"] = out_2["uncertainty"]

        for output_type in ["E", "-log(Kd)"]:
            output[output_type] = out_1[output_type]#.flatten()
            output[f"{output_type}2"] = out_2[output_type]#.flatten()
            output[f"{output_type}_difference"] = out_1[output_type] - out_2[output_type]

            diff_1 = output[output_type] - output[f"{output_type}2"]
            diff_2 = output[f"{output_type}2"] - output[output_type]
            class_preds = torch.stack((diff_1.flatten() / rel_temperature, diff_2.flatten() / rel_temperature), dim=-1)
            prob_1_ge_2 = torch.special.ndtr(diff_1.flatten() / 2**0.5 / rel_temperature)
            # Calculating the log_ndtr is necessary for stable training
            log_prob_1_ge_2 = torch.special.log_ndtr(diff_1.flatten() / 2 ** 0.5 / rel_temperature)
            log_prob_2_ge_1 = torch.special.log_ndtr(diff_2.flatten() / 2 ** 0.5 / rel_temperature)
            output[f"{output_type}_prob_cdf"] = torch.stack((prob_1_ge_2, 1-prob_1_ge_2), dim=-1)
            output[f"{output_type}_logit_cdf"] = torch.stack((log_prob_1_ge_2, log_prob_2_ge_1), dim=-1)
            output[f"{output_type}_prob"] = torch.nn.functional.softmax(class_preds, dim=-1)
            output[f"{output_type}_logit"] = torch.nn.functional.log_softmax(class_preds, dim=-1)

        return output
