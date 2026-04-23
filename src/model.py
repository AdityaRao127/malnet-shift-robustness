# model definitions for malnet-shift-robustness
# baseline gcn for graph classification, more variants get added in later phases

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ref: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
class BaselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # two gcn layers with relu, then mean pool, then linear
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


# same architecture as BaselineGCN, just trained on the wider enriched feature tensor
# the missingness transform pre-mutates the inputs upstream
class EnrichedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
