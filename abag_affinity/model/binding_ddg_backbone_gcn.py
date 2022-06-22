import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool
from torch import device

from abag_affinity.binding_ddg_predictor.models.predictor import  DDGPredictor


class DDGBackboneFC(torch.nn.Module):

    def __init__(self, pretrained_model_path: str, device: device = torch.device("cpu")):
        super(DDGBackboneFC, self).__init__()

        ckpt = torch.load(pretrained_model_path, map_location=torch.device(device))
        config = ckpt['config']
        weight = ckpt['model']

        model = DDGPredictor(config.model).to(device)
        model.load_state_dict(weight)

        encoding_feat_dim = config.model.node_feat_dim

        self.gat_encoder = model.encoder

        self.gat_encoder.requires_grad_(False)

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(encoding_feat_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

        self.device = device

        self.to(device)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.to(self.device)

        atom_pos = x[:, :14*3].reshape(1, -1, 14, 3)
        aa = x[:, 42].reshape(1, -1).long()
        seq = x[:, 43].reshape(1, -1).long()
        chain_seq = x[:, 44].reshape(1, -1).long()
        atom_pos_mask = x[:, 45:].reshape(1, -1, 14).bool()

        x = self.gat_encoder(atom_pos, aa, seq, chain_seq, atom_pos_mask).squeeze(0)

        x = global_max_pool(x, data.batch.to(self.device))

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        return x