# node feature transforms for the malnet pipeline
# degree, ldp (vectorized), structural surrogate, all used as pre_transform in pyg

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


# ref: https://arxiv.org/abs/2003.00982 (LDP, Cai and Wang)
def compute_ldp_features(data):
    # local degree profile, 5 dims per node, vectorized via scatter
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    features = torch.zeros(num_nodes, 5)
    features[:, 0] = deg

    if edge_index.numel() == 0:
        data.x = features
        return data

    row, col = edge_index
    neighbor_degs = deg[col]

    features[:, 1] = scatter(neighbor_degs, row, dim=0, dim_size=num_nodes, reduce="min")
    features[:, 2] = scatter(neighbor_degs, row, dim=0, dim_size=num_nodes, reduce="max")
    mean_nd = scatter(neighbor_degs, row, dim=0, dim_size=num_nodes, reduce="mean")
    features[:, 3] = mean_nd

    # std via two pass scatter, var = E[(x-mean)^2]
    sq_diff = (neighbor_degs - mean_nd[row]) ** 2
    var = scatter(sq_diff, row, dim=0, dim_size=num_nodes, reduce="mean")
    features[:, 4] = torch.sqrt(var.clamp(min=0))

    no_neighbors = deg == 0
    features[no_neighbors, 1:] = 0.0

    data.x = features
    return data


def _pagerank(edge_index, num_nodes, damping=0.85, num_iters=20):
    # power iteration pagerank, vectorized, no networkx dep
    if edge_index.numel() == 0:
        return torch.full((num_nodes,), 1.0 / max(num_nodes, 1))

    row, col = edge_index
    out_deg = degree(row, num_nodes=num_nodes, dtype=torch.float).clamp(min=1.0)

    pr = torch.full((num_nodes,), 1.0 / num_nodes)
    base = (1 - damping) / num_nodes

    for _ in range(num_iters):
        contrib = pr[row] / out_deg[row]
        pr = base + damping * scatter(contrib, col, dim=0, dim_size=num_nodes, reduce="sum")
    return pr


def _clustering_coefficient(edge_index, num_nodes, max_dense_nodes=4000):
    # local clustering coef per node, triangle count / possible triangles
    # for big graphs we skip and return zeros, dense adj would oom on colab
    # ref: https://en.wikipedia.org/wiki/Clustering_coefficient
    if edge_index.numel() == 0 or num_nodes > max_dense_nodes:
        return torch.zeros(num_nodes)

    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    adj = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes)
    ).coalesce()
    a_dense = adj.to_dense()
    a_sym = ((a_dense + a_dense.t()) > 0).float()
    a2 = a_sym @ a_sym
    triangles = (a_sym * a2).sum(dim=1) / 2.0

    possible = deg * (deg - 1).clamp(min=0) / 2.0
    cc = torch.where(possible > 0, triangles / possible.clamp(min=1), torch.zeros_like(possible))
    return cc.clamp(min=0, max=1)


def _bfs_depth_from_root(edge_index, num_nodes):
    # bfs depth from highest degree node, returns depth per node
    if edge_index.numel() == 0 or num_nodes == 0:
        return torch.zeros(num_nodes)

    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    root = int(deg.argmax().item())

    # build adjacency lists once
    adj_list = [[] for _ in range(num_nodes)]
    row, col = edge_index
    for r, c in zip(row.tolist(), col.tolist()):
        adj_list[r].append(c)
        adj_list[c].append(r)

    depths = torch.full((num_nodes,), -1, dtype=torch.long)
    depths[root] = 0
    frontier = [root]
    d = 0
    while frontier:
        d += 1
        next_frontier = []
        for u in frontier:
            for v in adj_list[u]:
                if depths[v] == -1:
                    depths[v] = d
                    next_frontier.append(v)
        frontier = next_frontier

    # unreachable nodes get max depth + 1
    max_d = depths.max().item() if (depths >= 0).any() else 0
    depths[depths == -1] = max_d + 1
    return depths.float()


def compute_structural_features(data):
    # ldp (5) + in/out deg (2) + clustering (1) + pagerank (1) + bfs depth (1) + norm degree (1) = 11 dims
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    features = torch.zeros(num_nodes, 11)

    in_deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float) if edge_index.numel() else torch.zeros(num_nodes)
    out_deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float) if edge_index.numel() else torch.zeros(num_nodes)

    # ldp block (using out_deg as "the" degree, similar to original paper convention)
    features[:, 0] = out_deg
    if edge_index.numel() > 0:
        row, col = edge_index
        nd = out_deg[col]
        features[:, 1] = scatter(nd, row, dim=0, dim_size=num_nodes, reduce="min")
        features[:, 2] = scatter(nd, row, dim=0, dim_size=num_nodes, reduce="max")
        mean_nd = scatter(nd, row, dim=0, dim_size=num_nodes, reduce="mean")
        features[:, 3] = mean_nd
        sq = (nd - mean_nd[row]) ** 2
        var = scatter(sq, row, dim=0, dim_size=num_nodes, reduce="mean")
        features[:, 4] = torch.sqrt(var.clamp(min=0))
        no_nbrs = out_deg == 0
        features[no_nbrs, 1:5] = 0.0

    features[:, 5] = in_deg
    features[:, 6] = out_deg
    features[:, 7] = _clustering_coefficient(edge_index, num_nodes)
    features[:, 8] = _pagerank(edge_index, num_nodes)
    features[:, 9] = _bfs_depth_from_root(edge_index, num_nodes)
    max_d = max(out_deg.max().item(), 1.0)
    features[:, 10] = out_deg / max_d

    data.x = features
    return data


def _two_step_neighborhood_size(edge_index, num_nodes, max_dense_nodes=4000):
    # number of unique 2 hop neighbors per node, falls back to zero on big graphs
    if edge_index.numel() == 0 or num_nodes > max_dense_nodes:
        return torch.zeros(num_nodes)

    adj = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes)
    ).coalesce()
    a_dense = adj.to_dense()
    a_sym = ((a_dense + a_dense.t()) > 0).float()
    a2 = (a_sym @ a_sym > 0).float()
    return a2.sum(dim=1) - a_sym.diagonal()


def _motif_counts(edge_index, num_nodes, max_dense_nodes=4000):
    # closed triangles and 2-paths through each node (3 dims)
    if edge_index.numel() == 0 or num_nodes > max_dense_nodes:
        return torch.zeros(num_nodes, 3)

    adj = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes)
    ).coalesce()
    a_dense = adj.to_dense()
    a_sym = ((a_dense + a_dense.t()) > 0).float()
    a2 = a_sym @ a_sym

    triangles = (a_sym * a2).sum(dim=1) / 2.0
    two_paths = a2.sum(dim=1) - a_sym.diagonal()
    open_paths = two_paths - triangles * 2

    out = torch.zeros(num_nodes, 3)
    out[:, 0] = triangles
    out[:, 1] = two_paths
    out[:, 2] = open_paths.clamp(min=0)
    return out


def _degree_quantile_bins(deg, num_bins=5):
    # one hot of which quantile bin each node falls into for this graph
    if deg.numel() == 0:
        return torch.zeros(0, num_bins)
    qs = torch.linspace(0, 1, num_bins + 1)
    edges = torch.quantile(deg, qs[1:-1]) if deg.numel() > 1 else torch.zeros(num_bins - 1)
    bins = torch.bucketize(deg, edges)
    one_hot = torch.zeros(deg.size(0), num_bins)
    one_hot.scatter_(1, bins.unsqueeze(1).clamp(max=num_bins - 1), 1)
    return one_hot


def compute_enriched_features(data):
    # structural (11) + surrogate metadata block (9) = 20 dims
    # surrogate metadata: motifs (3) + degree quantile bins (5) + 2 hop nbhd size (1)
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # reuse structural features
    data = compute_structural_features(data)
    structural = data.x

    out_deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float) if edge_index.numel() else torch.zeros(num_nodes)
    motifs = _motif_counts(edge_index, num_nodes)
    bins = _degree_quantile_bins(out_deg)
    two_hop = _two_step_neighborhood_size(edge_index, num_nodes).unsqueeze(1)

    semantic = torch.cat([motifs, bins, two_hop], dim=1)  # [num_nodes, 9]
    data.x = torch.cat([structural, semantic], dim=1)
    return data


# feature group spec for the enriched feature set, used by missingness simulation
ENRICHED_GROUPS = {
    "structural": (0, 11),
    "semantic": (11, 20),
}


def make_pre_transform(feature_type="degree", max_degree=128):
    # factory for pyg pre_transform argument
    if feature_type == "degree":
        return lambda data: compute_degree_features(data, max_degree=max_degree)
    if feature_type == "ldp":
        return compute_ldp_features
    if feature_type == "structural":
        return compute_structural_features
    if feature_type == "enriched":
        return compute_enriched_features
    raise ValueError(f"unknown feature type: {feature_type}")
