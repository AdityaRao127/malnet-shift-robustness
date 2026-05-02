# one-shot generator for phase3_gating.ipynb

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

cells.append(md("""# Phase 3: Missingness aware gating (the new bit)

Phase 2's EnrichedGCN sees the zero filled inputs but does not know which groups were dropped. Here we give the model the binary missingness mask as an extra input, cause telling it what is missing should help more than letting it guess from the zeros. A small MLP turns the mask into a sigmoid gate over the pooled graph embedding. The hope is the gate learns to downweight the metadata pathway when the mask says metadata is missing, instead of just averaging through the zeros.

The model is basically the same EnrichedGCN from phase 2, but with two changes. First, the forward pass takes the missingness mask as an additional argument. Second, right before the classifier, we multiply the pooled embedding by the gate output. Everything else, features, optimizer, epoch count, stays the same as phase 2 for a fair comparison.
"""))

cells.append(md("## 1. Setup\n"))

cells.append(code("""import os
REPO = "malnet-shift-robustness"
if not os.path.exists(REPO):
    !git clone https://github.com/AdityaRao127/malnet-shift-robustness.git
%cd $REPO
"""))

cells.append(code("!pip install -q torch-geometric==2.7.0 scikit-learn pandas matplotlib tqdm numpy\n"))

cells.append(code("""import os, sys, time, random, json
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader

from data_loader import load_malnet_splits
from features import make_pre_transform, ENRICHED_GROUPS
from missingness import make_missingness_transform
from model import GatedGCN
from train import train_model_gated, predict_gated, build_metrics, save_metrics

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
"""))

cells.append(md("## 2. Hyperparameters\n"))

cells.append(code("""EPOCHS = 50
LR = 0.001
BATCH_SIZE = 64
HIDDEN_DIM = 64
DROP_PROB = 0.5
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_GROUPS = len(ENRICHED_GROUPS)
print(f"groups: {ENRICHED_GROUPS}")
"""))

cells.append(md("## 3. Load Enriched Dataset\n"))

cells.append(code("""pre_transform = make_pre_transform(feature_type="enriched")
train_ds, val_ds, test_ds = load_malnet_splits(root="data/enriched/", pre_transform=pre_transform)

raw_type_dir = os.path.join("data", "enriched", "raw", "split_info_tiny", "type")
y_map = {}
for split_file in ["train.txt", "val.txt", "test.txt"]:
    with open(os.path.join(raw_type_dir, split_file)) as f:
        for line in f.read().strip().split("\\n"):
            mtype = line.split("/")[0]
            y_map.setdefault(mtype, len(y_map))
CLASS_NAMES = [name for name, _ in sorted(y_map.items(), key=lambda x: x[1])]
print(f"classes: {CLASS_NAMES}")
print(f"feature dim: {train_ds.num_features}")
"""))

cells.append(md("""## 4. Loaders with Missingness

Train loader gets random group drops, val and test get all-ones masks (no drops).
"""))

cells.append(code("""miss_train = make_missingness_transform(ENRICHED_GROUPS, drop_prob=DROP_PROB, strategy="zero", seed=SEED)
miss_test = make_missingness_transform(ENRICHED_GROUPS, drop_prob=0.0, strategy="zero", seed=0)

train_ds.transform = miss_train
val_ds.transform = miss_test
test_ds.transform = miss_test

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

batch = next(iter(train_loader))
print(f"batch x: {batch.x.shape}, mask: {batch.missingness_mask.shape}")
"""))

cells.append(md("## 5. Build Gated Model\n"))

cells.append(code("""model = GatedGCN(
    in_channels=train_ds.num_features,
    hidden_channels=HIDDEN_DIM,
    num_classes=train_ds.num_classes,
    num_groups=NUM_GROUPS,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"total params: {n_params:,}")
"""))

cells.append(md("## 6. Train\n"))

cells.append(code("""t0 = time.time()
model, train_losses, val_accs, best_val_acc = train_model_gated(
    model, train_loader, val_loader, EPOCHS, LR, device, verbose_every=10
)
train_time = time.time() - t0
print(f"training done in {train_time:.1f}s, best val {best_val_acc:.4f}")
"""))

cells.append(md("## 7. Test Check\n"))

cells.append(code("""test_preds, test_labels = predict_gated(model, test_loader, device)
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
test_acc = accuracy_score(test_labels, test_preds)
print(f"overall test accuracy: {test_acc:.4f}")
"""))

cells.append(md("## 8. Gate Distribution Sanity Check\n"))

cells.append(code("""# quick sanity check on what the gate learned, for the paper figure
model.train(False)
with torch.no_grad():
    full_mask = torch.ones(1, NUM_GROUPS).to(device)
    # half_mask: only the first group present, derived from NUM_GROUPS so it scales
    half_mask = torch.zeros(1, NUM_GROUPS).to(device)
    half_mask[0, 0] = 1.0
    no_mask = torch.zeros(1, NUM_GROUPS).to(device)
    g_full = model.gate_mlp(full_mask).cpu().numpy().flatten()
    g_half = model.gate_mlp(half_mask).cpu().numpy().flatten()
    g_none = model.gate_mlp(no_mask).cpu().numpy().flatten()

# scalar means so the student can tell if the gate actually learned to differentiate
print(f"gate mean | all present: {g_full.mean():.4f}")
print(f"gate mean | semantic missing: {g_half.mean():.4f}")
print(f"gate mean | all missing: {g_none.mean():.4f}")
print(f"max abs diff (present vs all missing): {abs(g_full - g_none).max():.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(g_full, bins=20, alpha=0.6, label="all present", color="steelblue")
ax.hist(g_half, bins=20, alpha=0.6, label="semantic missing", color="coral")
ax.hist(g_none, bins=20, alpha=0.6, label="all missing", color="forestgreen")
ax.set_xlabel("Gate Value (sigmoid)"); ax.set_ylabel("Count")
ax.set_title("Learned Gate Distribution by Mask")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gating_weights_hist.png"), dpi=150)
plt.show()
"""))

cells.append(md("## 9. Save Results\n"))

cells.append(code("""cfg = {
    "epochs": EPOCHS, "learning_rate": LR, "batch_size": BATCH_SIZE,
    "hidden_dim": HIDDEN_DIM, "drop_prob": DROP_PROB,
    "missingness_strategy": "zero_with_gating", "feature_type": "enriched",
    "num_groups": NUM_GROUPS, "optimizer": "Adam", "device": str(device),
}
metrics = build_metrics(test_preds, test_labels, CLASS_NAMES, cfg, best_val_acc, train_time)
json_path, csv_path = save_metrics(metrics, RESULTS_DIR, prefix="gated")
print("saved", json_path)
print("saved", csv_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, EPOCHS+1), train_losses, color="steelblue")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("CE Loss"); ax1.set_title("Training Loss"); ax1.grid(alpha=0.3)
ax2.plot(range(1, EPOCHS+1), val_accs, color="steelblue")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Validation Accuracy"); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gated_training_curve.png"), dpi=150)
plt.show()

cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Gated GCN - Confusion Matrix")
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
plt.savefig(os.path.join(RESULTS_DIR, "gated_confusion_matrix.png"), dpi=150)
plt.show()
"""))

cells.append(md("""## Summary

GatedGCN is trained with random missingness and evaluated with all features present. The numbers and the gate histogram tell us whether the gate actually learned to differentiate. Phase 4 pulls the real shifted split and computes the robustness gap across all four models.
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

out_path = os.path.join(REPO, "notebooks", "phase3_gating.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote notebook with {len(cells)} cells to {out_path}")
