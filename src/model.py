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


# missingness aware gating, the novel contribution
# takes a binary mask saying which feature groups are present, learns a gate over the pooled embedding
class GatedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_groups):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # small mlp turns the [B, num_groups] mask into a [B, hidden] gate
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_groups, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Sigmoid(),
        )
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, missingness_mask):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # [B, hidden]
        gate = self.gate_mlp(missingness_mask)  # [B, hidden]
        x = x * gate
        x = self.classifier(x)
        return x
