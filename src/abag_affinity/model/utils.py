import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, global_sort_pool, GlobalAttention, GraphConv, GATv2Conv
from torch.nn import ReLU, LeakyReLU, GELU
from typing import Tuple, Dict

from .pretrained_models import DDGBackbone, DeepRefineBackbone, IPABindingEmbedder, DiffusionPipelinePredictor

class NoOpModel(torch.nn.Module):
    def __init__(self):
        super(NoOpModel, self).__init__()
        self.model_type = "NoOp"

    def forward(self, data: Dict):
        return data["graph"]["node"].x


class FixedSizeAggregation(torch.nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        graph_embeddings = []
        for i in range(torch.max(batch) + 1):
            graph_embeddings.append(x[torch.where(batch == i)].flatten())
        graph_embeddings = torch.stack(graph_embeddings)
        return graph_embeddings


aggregation_methods = {
    "max": global_max_pool,
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "attention": GlobalAttention,
    "fixed_size": FixedSizeAggregation(),
    "edge": lambda x: x,
    "interface_sum": global_add_pool,
}

layer_types = {
    "GCN": GraphConv,
    "GAT": GATv2Conv
}

layer_type_edge_dim = {
    "GCN": False,
    "GAT": True
}

nonlinearity_function = {
    "relu": ReLU,
    "leaky": LeakyReLU,
    "gelu": GELU
}

pretrained_models = {
    "Binding_DDG": DDGBackbone,
    "DeepRefine": DeepRefineBackbone,
    "IPA": IPABindingEmbedder,
    "Diffusion": DiffusionPipelinePredictor
}


def pretrained_embeddings(data: Dict, pretrained_model: torch.nn.Module) -> Dict:
    """ Get the embeddings of the backbone module

    Convert the input data to the correct format and then feed through the backbone model
    Return node and edge information as well as batch information

    Args:
        data: Input data of different formats
        pretrained_model: backbone model to use

    Returns:
        Tuple: node and edge information as well as batch information
    """

    if not hasattr(pretrained_model, "model_type"):
        raise ValueError("Pretrained model does not contain model_type variable")
    x = pretrained_model(data)

    diff = data["graph"]["node"].x.shape[0] - x.shape[0]
    if diff > 0: # max num nodes is greater than actual available nodes in pdb
        x = torch.vstack((x, torch.zeros((diff, x.shape[-1]), dtype=x.dtype, device=x.device)))
    data["graph"]["node"].x = x

    return data
