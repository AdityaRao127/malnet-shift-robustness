# dataset i/o for MalNetTiny, no feature transforms here (those live in features.py)
# pyg dataset: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MalNetTiny.html
# malnet paper: https://openreview.net/pdf?id=1xDTDk3XPW

import json
import os

import matplotlib
# colab picks its own inline backend, only force Agg if no display is set
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader import DataLoader

# re-export so old code still imports make_pre_transform from data_loader
from features import make_pre_transform  # noqa: F401


def load_malnet_splits(root="data/", pre_transform=None, force_reload=False):
    # MalNetTiny official train/val/test splits, force_reload if transform changes
    train_ds = MalNetTiny(root=root, split="train", pre_transform=pre_transform, force_reload=force_reload)
    val_ds = MalNetTiny(root=root, split="val", pre_transform=pre_transform, force_reload=force_reload)
    test_ds = MalNetTiny(root=root, split="test", pre_transform=pre_transform, force_reload=force_reload)
    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_dataset_stats(dataset):
    # iterates once, picks up node counts, edge counts, labels
    num_graphs = len(dataset)
    node_counts = []
    edge_counts = []
    labels = []

    for i in range(num_graphs):
        g = dataset[i]
        node_counts.append(g.num_nodes)
        edge_counts.append(g.num_edges)
        labels.append(g.y.item() if isinstance(g.y, torch.Tensor) else g.y)

    return {
        "num_graphs": num_graphs,
        "num_classes": dataset.num_classes,
        "avg_nodes": float(np.mean(node_counts)),
        "max_nodes": int(np.max(node_counts)),
        "min_nodes": int(np.min(node_counts)),
        "avg_edges": float(np.mean(edge_counts)),
        "max_edges": int(np.max(edge_counts)),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "labels": labels,
    }


def save_dataset_summary(stats, class_names=None, output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "num_graphs": stats["num_graphs"],
        "num_classes": stats["num_classes"],
        "avg_nodes": round(stats["avg_nodes"], 2),
        "max_nodes": stats["max_nodes"],
        "min_nodes": stats["min_nodes"],
        "avg_edges": round(stats["avg_edges"], 2),
        "max_edges": stats["max_edges"],
    }
    if class_names:
        summary["class_names"] = class_names

    path = os.path.join(output_dir, "dataset_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved dataset summary to {path}")
    return summary


def plot_class_distribution(labels, class_names=None, output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)

    unique, counts = np.unique(labels, return_counts=True)
    names = class_names if class_names else [str(u) for u in unique]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("MalNetTiny Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved class distribution plot to {path}")


def plot_node_histogram(node_counts, output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(node_counts, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Frequency")
    ax.set_title("MalNetTiny Node Count Distribution")
    plt.tight_layout()

    path = os.path.join(output_dir, "node_count_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved node count histogram to {path}")
