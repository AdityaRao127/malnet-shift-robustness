# one-shot generator for phase1_structural.ipynb
# regenerate via: python scripts/_gen_phase1_notebook.py

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(md("""# Phase 1: Structural baseline

Assignment 3 used one hot degree features and got 75.1%. Here we change those to Local Degree Profile (https://arxiv.org/abs/2003.00982) plus clustering, PageRank, and BFS depth, cause one number per node throws away a lot of info. Same GCN, just better inputs.

11 features per node total:
- LDP block, 5 dims: degree and min/max/mean/std of neighbor degrees
- in degree and out degree, 2 dims
- local clustering coefficient
- PageRank score
- BFS depth from the highest degree node
- normalized degree

Tran et al. (https://arxiv.org/abs/2508.06734) report 85.6% on the standard split with LDP and global max pooling. We use mean pooling so will probably land a bit lower than that.
"""))

cells.append(md("""## 1. Setup

Clone the repo so we can import from src/. No-op if you already have it.
"""))

cells.append(code("""# clone the repo on a fresh colab session, no-op if already there
import os
REPO = "malnet-shift-robustness"
if not os.path.exists(REPO):
    !git clone https://github.com/AdityaRao127/malnet-shift-robustness.git
%cd $REPO
"""))

cells.append(code("""# colab already has torch, grab the rest. pinning pyg avoids api drift
!pip install -q torch-geometric==2.7.0 scikit-learn pandas matplotlib tqdm numpy
"""))

cells.append(code("""import os, sys, time, random, json
sys.path.insert(0, "src")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from data_loader import load_malnet_splits, make_loaders
from features import make_pre_transform
from model import BaselineGCN
from train import train_model, predict, build_metrics, save_metrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
"""))

cells.append(md("""## 2. Hyperparameters
"""))

cells.append(code("""EPOCHS = 50
LR = 0.001
BATCH_SIZE = 64
HIDDEN_DIM = 64
FEATURE_TYPE = "structural"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
"""))

cells.append(md("""## 3. Load Dataset with Structural Features

Note the root="data/structural/" so this cache does not collide with the degree-features cache from the original baseline notebook.
"""))

cells.append(code("""pre_transform = make_pre_transform(feature_type=FEATURE_TYPE)

# separate root keeps caches isolated by feature config
train_ds, val_ds, test_ds = load_malnet_splits(
    root="data/structural/", pre_transform=pre_transform
)

print(f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
print(f"features per node: {train_ds.num_features}")
print(f"classes: {train_ds.num_classes}")
print(f"sample: {train_ds[0]}")

# class names in pyg label order, parsed from the raw split files
raw_type_dir = os.path.join("data", "structural", "raw", "split_info_tiny", "type")
y_map = {}
for split_file in ["train.txt", "val.txt", "test.txt"]:
    with open(os.path.join(raw_type_dir, split_file)) as f:
        for line in f.read().strip().split("\\n"):
            mtype = line.split("/")[0]
            y_map.setdefault(mtype, len(y_map))
CLASS_NAMES = [name for name, _ in sorted(y_map.items(), key=lambda x: x[1])]
print(f"classes (label order): {CLASS_NAMES}")
"""))

cells.append(md("""## 4. DataLoaders
"""))

cells.append(code("""train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

batch = next(iter(train_loader))
print(f"batch x shape: {batch.x.shape}")
print(f"batch y shape: {batch.y.shape}")
"""))

cells.append(md("""## 5. Model

Same BaselineGCN architecture from src/model.py, just with the wider 11-dim input.
"""))

cells.append(code("""model = BaselineGCN(
    in_channels=train_ds.num_features,
    hidden_channels=HIDDEN_DIM,
    num_classes=train_ds.num_classes,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"total params: {n_params:,}")
"""))

cells.append(md("""## 6. Training
"""))

cells.append(code("""t0 = time.time()
model, train_losses, val_accs, best_val_acc = train_model(
    model, train_loader, val_loader, EPOCHS, LR, device, verbose_every=10
)
train_time = time.time() - t0
print(f"training done in {train_time:.1f}s, best val acc {best_val_acc:.4f}")
"""))

cells.append(md("""## 7. Test Check
"""))

cells.append(code("""test_preds, test_labels = predict(model, test_loader, device)

report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES)
print(report)
test_acc = accuracy_score(test_labels, test_preds)
print(f"overall test accuracy: {test_acc:.4f}")

# delta vs assignment 3 baseline (one hot degree, 75.1%)
assn3_acc = 0.7510
print(f"assignment 3 baseline: {assn3_acc:.4f}")
print(f"this run (structural): {test_acc:.4f}")
print(f"delta: {test_acc - assn3_acc:+.4f}")
"""))

cells.append(md("""## 8. Save Results

Outputs land in results/baseline_structural_*.{json,csv,png} so they do not overwrite the original baseline.
"""))

cells.append(code("""cfg = {
    "epochs": EPOCHS, "learning_rate": LR, "batch_size": BATCH_SIZE,
    "hidden_dim": HIDDEN_DIM, "feature_type": FEATURE_TYPE,
    "optimizer": "Adam", "device": str(device),
}
metrics = build_metrics(test_preds, test_labels, CLASS_NAMES, cfg, best_val_acc, train_time)
json_path, csv_path = save_metrics(metrics, RESULTS_DIR, prefix="baseline_structural")
print("saved", json_path)
print("saved", csv_path)

# confusion matrix
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Structural GCN - Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_yticklabels(CLASS_NAMES)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
fig.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "baseline_structural_confusion_matrix.png"), dpi=150)
plt.show()

# training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, EPOCHS+1), train_losses, color="steelblue")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("CE Loss"); ax1.set_title("Training Loss"); ax1.grid(alpha=0.3)
ax2.plot(range(1, EPOCHS+1), val_accs, color="steelblue")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Validation Accuracy"); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "baseline_structural_training_curve.png"), dpi=150)
plt.show()
"""))

cells.append(md("""## Summary

If the structural features carry more signal than degree alone, this should beat the 75.1% baseline. Phase 2 enhances the feature set further and starts simulating missingness during training.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(REPO, "notebooks", "phase1_structural.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote notebook with {len(cells)} cells to {out_path}")
