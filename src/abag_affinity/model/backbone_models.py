import torch
import torch.nn as nn
from torch import device
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool, GlobalAttention


from abag_affinity.binding_ddg_predictor.models.predictor import DDGPredictor


class GraphConvAttentionModelWithBackbone(torch.nn.Module):
    def __init__(self, backbone_model: torch.nn.Module,  num_nodes: int = 25, device: device = torch.device("cpu")):
        super(GraphConvAttentionModelWithBackbone, self).__init__()
        self.num_nodes = num_nodes

        self.backbone_model = backbone_model
        self.backbone_model.requires_grad_(False)

        self.relu = nn.LeakyReLU()

        self.conv_1 = GATv2Conv(backbone_model.embedding_size, 32, heads=2, dropout=0.5, edge_dim=3)
        self.conv_2 = GATv2Conv(32 * 2, 10, heads=3, dropout=0.5, edge_dim=3)

        if num_nodes is None:
            num_nodes = 1

        self.embedding_size = 10 * 3 * num_nodes

        self.fc_1 = nn.Linear(self.embedding_size, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

        self.device = device

        self.to(device)

    def forward(self, data: Data):
        if hasattr(data, "num_graphs") and data.num_graphs > 1:
            x = torch.tensor([])
            for data_graph in data.to_data_list():
                x = torch.cat((x, self.backbone_model(data_graph).squeeze(0)))
        else:
            x = self.backbone_model(data).squeeze(0)
        _, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.relu(x)

        if self.num_nodes is None:
            x = global_max_pool(x, data.batch.to(self.device))
        else:
            x = x.view(data.num_graphs, self.embedding_size)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        return x


class ModelWithBackbone(torch.nn.Module):
    def __init__(self, backbone_model: torch.nn.Module, num_nodes: int = 25, device: device = torch.device("cpu")):
        super(ModelWithBackbone, self).__init__()
        self.max_nodes = num_nodes

        self.backbone_model = backbone_model
        self.backbone_model.requires_grad_(False)

        self.relu = nn.ReLU()
        if num_nodes is None:
            num_nodes = 1

        self.embedding_size = backbone_model.embedding_size * num_nodes
        self.fc_1 = nn.Linear(self.embedding_size, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

        self.device = device

        self.to(device)

    def forward(self, data: Data):
        x = torch.tensor([])
        x = self.backbone_model(data).squeeze(0)

        if self.num_nodes is None:
            x = global_max_pool(x, data.batch.to(self.device))
        else:
            x = x.view(data.num_graphs, self.embedding_size)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        return x


class DDGBackbone(torch.nn.Module):
    def __init__(self, pretrained_model_path: str, device: device = torch.device("cpu")):
        super(DDGBackbone, self).__init__()

        ckpt = torch.load(pretrained_model_path, map_location=torch.device(device))
        config = ckpt['config']
        weight = ckpt['model']

        model = DDGPredictor(config.model).to(device)
        model.load_state_dict(weight)

        self.embedding_size = config.model.node_feat_dim

        self.gat_encoder = model.encoder
        self.device = device
        self.to(device)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.to(self.device)

        atom_pos = x[:, :14*3].reshape(1, -1, 14, 3)
        aa = x[:, 42].reshape(1, -1).long()
        seq = x[:, 43].reshape(1, -1).long()
        chain_seq = x[:, 44].reshape(1, -1).long()
        atom_pos_mask = x[:, 45:].reshape(1, -1, 14).bool()

        x = self.gat_encoder(atom_pos, aa, seq, chain_seq, atom_pos_mask)

        return x