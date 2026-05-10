# load tran et al. shifted split from local pt files into pyg dataset
# uses pyg's own separate() so num_nodes and other private storage attrs are preserved
# ref: https://pytorch-geometric.readthedocs.io/en/2.7.0/_modules/torch_geometric/data/in_memory_dataset.html

import glob
import os

import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader


def _strip_unwanted_attrs(data, keep=("edge_index", "y", "num_nodes")):
    # remove any heavy embedding fields that snuck in, keep only structure + label
    for k in list(data.keys()):
        if k not in keep:
            try:
                delattr(data, k)
            except AttributeError:
                pass
    return data


def _restore_data_object(raw_data, data_cls):
    # converts a pyg-collated payload back into a Data object, restoring _num_nodes
    if isinstance(raw_data, Data):
        # legacy tran format: already a collated Data
        return raw_data

    if not isinstance(raw_data, dict):
        raise TypeError(f"cannot restore Data from {type(raw_data)}")

    raw_data = dict(raw_data)
    num_nodes_list = raw_data.pop("_num_nodes", None)
    cls = data_cls if isinstance(data_cls, type) else Data
    if hasattr(cls, "from_dict"):
        data = cls.from_dict(raw_data)
    else:
        data = Data.from_dict(raw_data)

    if num_nodes_list is not None:
        counts = [int(v) for v in num_nodes_list]
        data._store._num_nodes = counts
        data.num_nodes = sum(counts)
    return data


def _unpack_pyg_save_format(loaded):
    # supports legacy (Data, slices), pyg 2.4 (dict, slices), pyg 2.5+ (dict, slices, cls)
    if isinstance(loaded, Data):
        return [loaded]
    if isinstance(loaded, list):
        return [d for d in loaded if isinstance(d, Data)]
    if not (isinstance(loaded, tuple) and len(loaded) in (2, 3)):
        if isinstance(loaded, dict) and "edge_index" in loaded:
            return [Data(**{k: v for k, v in loaded.items() if isinstance(v, torch.Tensor) or k == "num_nodes"})]
        return []

    if len(loaded) == 3:
        raw_data, slices, data_cls = loaded
    else:
        raw_data, slices = loaded
        data_cls = Data

    data = _restore_data_object(raw_data, data_cls)

    if slices is None:
        return [data]

    # any normal slice tensor tells us the graph count
    slice_tensors = [v for v in slices.values() if torch.is_tensor(v)]
    if not slice_tensors:
        return [data]
    num_graphs = int(slice_tensors[0].numel() - 1)

    # use pyg separate() instead of manual narrowing, handles _num_nodes and cat dims correctly
    return [
        separate(
            cls=data.__class__,
            batch=data,
            idx=i,
            slice_dict=slices,
            decrement=False,
        )
        for i in range(num_graphs)
    ]


def load_split_dict(common_root):
    # tran et al. ships a split_dict.pt with train/valid/test index lists for the common subset
    # using the test slice (1000 graphs) is the fair comparison to their reported numbers
    # weights_only=True is safe here, it is just a dict of int lists
    path = os.path.join(common_root, "split_dict.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def slice_to_split(graphs, split_indices):
    # returns a new list selecting just the graphs at the given indices
    return [graphs[int(i)] for i in split_indices]


def node_count_stats(graphs):
    # diagnostic, useful to report alongside results since our semantic features cap at 4000 nodes
    counts = torch.tensor([int(g.num_nodes) for g in graphs])
    return {
        "n": int(counts.numel()),
        "mean": float(counts.float().mean()),
        "median": int(counts.median().item()),
        "max": int(counts.max().item()),
        "pct_over_4000": float((counts > 4000).float().mean() * 100),
    }


def load_shifted_graphs(root, pre_transform=None, max_graphs=None, recursive=False, target_file="data.pt"):
    # loads target_file from root by default, applies pre_transform if given
    # set recursive=True to walk subdirs; target_file=None to load every .pt
    if recursive:
        pt_files = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
    elif target_file:
        candidate = os.path.join(root, target_file)
        pt_files = [candidate] if os.path.exists(candidate) else []
    else:
        pt_files = sorted(glob.glob(os.path.join(root, "*.pt")))

    if max_graphs is not None:
        pt_files = pt_files[:max_graphs]

    if not pt_files:
        raise FileNotFoundError(
            f"no .pt files found in {root} (recursive={recursive}, target_file={target_file}). "
            "did the huggingface download succeed? check download_common_split call and root path."
        )

    graphs = []
    for path in pt_files:
        # weights_only=False needed since pyg .pt files contain custom Data classes
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        items = _unpack_pyg_save_format(loaded)

        for d in items:
            if not hasattr(d, "y") or d.y is None:
                continue  # skip malformed graphs without labels
            d = _strip_unwanted_attrs(d)
            if pre_transform is not None:
                d = pre_transform(d)
            graphs.append(d)

    return graphs


def make_shifted_loader(graphs, batch_size=64):
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)


def compute_robustness_gap(standard_acc, shifted_acc):
    # higher gap means the model degrades more under shift
    return round(float(standard_acc) - float(shifted_acc), 4)


# tran et al. common loader uses alphabetical class order, pyg malnettiny uses
# first appearance from raw splits, so we have to remap labels before eval
# ref: https://raw.githubusercontent.com/ngoctnq/malnet-features/master/training/graphgps/loader/dataset/malnet_tiny_features.py
def make_label_remap(source_names, target_names):
    # returns a list where result[source_label_int] = target_label_int
    if set(source_names) != set(target_names):
        raise ValueError(
            f"label name sets differ. source={source_names}, target={target_names}"
        )
    return [target_names.index(name) for name in source_names]


def remap_labels(graphs, remap):
    # mutates each graph's y in place, idempotent so rerunning a cell wont double remap
    for g in graphs:
        if getattr(g, "_y_remapped", False):
            continue  # already done, skip silently
        old = int(g.y.flatten()[0].item())
        new = remap[old]
        g.y = g.y.new_tensor([new]).reshape(g.y.shape)
        g._y_remapped = True
    return graphs
