# node feature transforms for the malnet pipeline
# degree, ldp (vectorized), and helpers used as pre_transform in pyg

import torch
from torch_geometric.utils import degree, scatter


# ref: https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.OneHotDegree
def compute_degree_features(data, max_degree=128):
    # one hot of node degree, clipped at max_degree
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    deg = deg.clamp(max=max_degree)

    one_hot = torch.zeros(num_nodes, max_degree + 1)
    one_hot.scatter_(1, deg.unsqueeze(1), 1)
    data.x = one_hot
    return data


# ref: https://arxiv.org/abs/2003.00982 (LDP paper, Cai and Wang)
def compute_ldp_features(data):
    # local degree profile, 5 dims per node, vectorized via scatter
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    row, col = edge_index
    neighbor_degs = deg[col]

    features = torch.zeros(num_nodes, 5)
    features[:, 0] = deg

    # if no edges, just keep zeros for the neighbor stats
    if edge_index.numel() == 0:
        data.x = features
        return data

    features[:, 1] = scatter(neighbor_degs, row, dim=0, dim_size=num_nodes, reduce="min")
    features[:, 2] = scatter(neighbor_degs, row, dim=0, dim_size=num_nodes, reduce="max")
    mean_nd = scatter(neighbor_degs, row, dim=0, dim_size=num_nodes, reduce="mean")
    features[:, 3] = mean_nd

    # std via two pass scatter, var = E[(x-mean)^2]
    sq_diff = (neighbor_degs - mean_nd[row]) ** 2
    var = scatter(sq_diff, row, dim=0, dim_size=num_nodes, reduce="mean")
    features[:, 4] = torch.sqrt(var.clamp(min=0))

    # nodes with zero degree have nan stats from min/max scatter, zero them out
    no_neighbors = deg == 0
    features[no_neighbors, 1:] = 0.0

    data.x = features
    return data


def make_pre_transform(feature_type="degree", max_degree=128):
    # factory for pyg pre_transform argument
    if feature_type == "degree":
        return lambda data: compute_degree_features(data, max_degree=max_degree)
    if feature_type == "ldp":
        return compute_ldp_features
    raise ValueError(f"unknown feature type: {feature_type}")
