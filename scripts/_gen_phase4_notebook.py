# one-shot generator for phase4_shifted.ipynb

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

cells.append(md("""# Phase 4: Shifted evaluation

This is where we actually answer the research question. Pull Tran et al.'s MalNet-Tiny-Common dataset (the covariate shift benchmark) from HuggingFace (https://huggingface.co/datasets/nntvu/MalNet-Tiny-Features), retrain all four models from scratch, evaluate each on both the standard test set and the shifted test set.

The full HF repo is around 1 TB cause it includes huge LLM embedding variants. We only grab the structure only variant (none+none, 363 MB) so the download stays manageable on Colab.

Headline metric: robustness gap = standard accuracy minus shifted accuracy. Smaller gap means the model generalizes better. We want to see if the gating model from phase 3 has a smaller gap than the structural baseline and the zero fill or prune EnrichedGCN.
"""))

cells.append(md("## 1. Setup\n"))

cells.append(code("""import os
REPO = "malnet-shift-robustness"
if not os.path.exists(REPO):
    !git clone https://github.com/AdityaRao127/malnet-shift-robustness.git
%cd $REPO
"""))

cells.append(code("""!pip install -q torch-geometric==2.7.0 scikit-learn pandas matplotlib tqdm numpy huggingface_hub
"""))

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
from model import BaselineGCN, EnrichedGCN, GatedGCN
from train import train_model, train_model_gated, predict, predict_gated, build_metrics, save_metrics
from hf_download import list_repo_files, download_common_split, get_split_root
from shift_loader import (
    load_shifted_graphs, make_shifted_loader, compute_robustness_gap,
    make_label_remap, remap_labels,
    load_split_dict, slice_to_split, node_count_stats,
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
"""))

cells.append(md("""## 2. Inspect HuggingFace Repo

First, see what files are actually available so we know what patterns to download.
"""))

cells.append(code("""files = list_repo_files()
print(f"total files: {len(files)}")
for f in files[:30]:
    print(" ", f)
"""))

cells.append(md("""## 3. Download Shifted Common Split

This pulls graph structure files for the Common subset. The first run takes a few minutes depending on actual file size.
"""))

cells.append(code("""# only pulls the structure-only variant (none+none), avoids llm embedding files
local_path = download_common_split(local_dir="data/hf_malnet/", variant="none+none")
print(f"downloaded to: {local_path}")
common_root = get_split_root("data/hf_malnet/", variant="none+none")
print(f"common split root: {common_root}")
"""))

cells.append(md("""## 4. Load Shifted Test Set

Apply the same pre_transform we used during training so feature dimensions match. Loader is non-recursive on purpose so we only read the .pt in this exact directory.
"""))

cells.append(code("""# enriched features for the enriched and gated models
pre_enriched = make_pre_transform(feature_type="enriched")
shifted_enriched_all = load_shifted_graphs(common_root, pre_transform=pre_enriched)
print(f"shifted enriched graphs (all): {len(shifted_enriched_all)}")
assert len(shifted_enriched_all) > 0, "shifted dataset is empty, check the HuggingFace download"

# structural features for the structural baseline
pre_structural = make_pre_transform(feature_type="structural")
shifted_structural_all = load_shifted_graphs(common_root, pre_transform=pre_structural)
print(f"shifted structural graphs (all): {len(shifted_structural_all)}")

# tran et al. ship a split_dict.pt with 3500 train / 500 val / 1000 test indices
# for fair comparison to their reported common-shifted numbers, we use just the test slice
split_dict = load_split_dict(common_root)
if split_dict is not None:
    print(f"split_dict found: train={len(split_dict['train'])}, "
          f"valid={len(split_dict['valid'])}, test={len(split_dict['test'])}")
    shifted_enriched = slice_to_split(shifted_enriched_all, split_dict["test"])
    shifted_structural = slice_to_split(shifted_structural_all, split_dict["test"])
    print(f"using {len(shifted_enriched)} graphs from common test split for evaluation")
else:
    print("no split_dict.pt found, falling back to all graphs as the shifted test set")
    shifted_enriched = shifted_enriched_all
    shifted_structural = shifted_structural_all

# node count stats (semantic features cap at 4000 nodes for memory reasons)
stats = node_count_stats(shifted_structural)
print(f"shifted node counts: {stats}")

# label space sanity check
all_labels = set()
for d in shifted_structural:
    all_labels.add(int(d.y.item()) if d.y.numel() == 1 else int(d.y.flatten()[0].item()))
print(f"shifted label values (pre remap): {sorted(all_labels)}")
"""))

cells.append(md("""## 4b. Fix the label ordering

The Common dataset orders its classes differently than PyG's MalNetTiny, so without a remap the shifted accuracy comes out scrambled. We build the remap once and apply it to all shifted graphs before evaluation.
"""))

cells.append(code("""# tran et al. common loader class order, from their malnet_tiny_features.py
COMMON_CLASS_NAMES = ["addisplay", "adware", "benign", "downloader", "trojan"]
# our standard CLASS_NAMES are derived from pyg's first-appearance order, set later

# we cannot build the remap yet because CLASS_NAMES is defined in the next cell after
# load_malnet_splits runs, so the remap is computed inline at first use below
print("standard order will be derived from data/structural/raw splits")
print(f"common order (from tran public code): {COMMON_CLASS_NAMES}")
"""))

cells.append(md("""## 5. Retrain All Four Models on Standard Train

To compare fair, we retrain each model from scratch on the standard MalNet-Tiny train, then evaluate on both standard and shifted test sets.

(if you already trained these in earlier notebooks and saved checkpoints, you could load those instead)
"""))

cells.append(code("""EPOCHS = 50
LR = 0.001
BATCH_SIZE = 64
HIDDEN_DIM = 64
DROP_PROB = 0.5
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- structural baseline ---
print("=" * 50, "\\nstructural baseline")
ds_train, ds_val, ds_test = load_malnet_splits(root="data/structural/", pre_transform=pre_structural)
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

m_baseline = BaselineGCN(ds_train.num_features, HIDDEN_DIM, ds_train.num_classes).to(device)
t0 = time.time()
m_baseline, _, _, best_val = train_model(m_baseline, train_loader, val_loader, EPOCHS, LR, device)
t_baseline = time.time() - t0
print(f"baseline trained in {t_baseline:.1f}s, best val {best_val:.4f}")

# helper to read class order from a populated pyg root
def _read_class_names(root):
    raw_type_dir = os.path.join(root, "raw", "split_info_tiny", "type")
    y_map = {}
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        with open(os.path.join(raw_type_dir, split_file)) as f:
            for line in f.read().strip().split("\\n"):
                mtype = line.split("/")[0]
                y_map.setdefault(mtype, len(y_map))
    return [name for name, _ in sorted(y_map.items(), key=lambda x: x[1])]

CLASS_NAMES = _read_class_names(os.path.join("data", "structural"))
print(f"standard classes: {CLASS_NAMES}")

# now that we have CLASS_NAMES we can build the remap and apply to shifted graphs once
label_remap = make_label_remap(COMMON_CLASS_NAMES, CLASS_NAMES)
print(f"label remap (common -> standard): {label_remap}")
remap_labels(shifted_structural, label_remap)
remap_labels(shifted_enriched, label_remap)
print("shifted labels remapped to standard class order")
"""))

cells.append(md("""## 6. Evaluate Baseline on Standard and Shifted
"""))

cells.append(code("""# standard
preds_s, labels_s = predict(m_baseline, test_loader, device)
acc_standard = accuracy_score(labels_s, preds_s)

# shifted
shifted_loader = make_shifted_loader(shifted_structural, batch_size=BATCH_SIZE)
preds_sh, labels_sh = predict(m_baseline, shifted_loader, device)
acc_shifted = accuracy_score(labels_sh, preds_sh)

gap = compute_robustness_gap(acc_standard, acc_shifted)
print(f"baseline standard: {acc_standard:.4f}, shifted: {acc_shifted:.4f}, gap: {gap:.4f}")

# save shifted metrics
config = {"feature_type": "structural", "epochs": EPOCHS, "device": str(device)}
metrics = build_metrics(preds_sh, labels_sh, CLASS_NAMES, config, best_val, t_baseline)
metrics["robustness_gap"] = gap
metrics["accuracy_standard"] = round(float(acc_standard), 4)
save_metrics(metrics, RESULTS_DIR, prefix="baseline_shifted")

# free gpu memory before next training block
del m_baseline, train_loader, val_loader, test_loader, ds_train, ds_val, ds_test
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""))

cells.append(md("""## 7. EnrichedGCN with Zero-Fill, Both Splits
"""))

cells.append(code("""ds_e_train, ds_e_val, ds_e_test = load_malnet_splits(root="data/enriched/", pre_transform=pre_enriched)
miss_train = make_missingness_transform(ENRICHED_GROUPS, drop_prob=DROP_PROB, strategy="zero", seed=SEED)
miss_test = make_missingness_transform(ENRICHED_GROUPS, drop_prob=0.0, strategy="zero", seed=0)
ds_e_train.transform = miss_train
ds_e_val.transform = miss_test
ds_e_test.transform = miss_test

train_loader_e = DataLoader(ds_e_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader_e = DataLoader(ds_e_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader_e = DataLoader(ds_e_test, batch_size=BATCH_SIZE, shuffle=False)

m_enriched = EnrichedGCN(ds_e_train.num_features, HIDDEN_DIM, ds_e_train.num_classes).to(device)
t0 = time.time()
m_enriched, _, _, best_val_e = train_model(m_enriched, train_loader_e, val_loader_e, EPOCHS, LR, device)
t_enriched = time.time() - t0

preds_es, labels_es = predict(m_enriched, test_loader_e, device)
acc_es = accuracy_score(labels_es, preds_es)

# shifted: apply miss_test to keep mask attribute (model does not use it but loader needs consistent batches)
for d in shifted_enriched:
    d.missingness_mask = torch.ones(1, len(ENRICHED_GROUPS))
shifted_loader_e = make_shifted_loader(shifted_enriched, batch_size=BATCH_SIZE)
preds_esh, labels_esh = predict(m_enriched, shifted_loader_e, device)
acc_esh = accuracy_score(labels_esh, preds_esh)

gap_e = compute_robustness_gap(acc_es, acc_esh)
print(f"enriched_zero standard: {acc_es:.4f}, shifted: {acc_esh:.4f}, gap: {gap_e:.4f}")

config = {"feature_type": "enriched", "missingness_strategy": "zero", "epochs": EPOCHS, "device": str(device)}
metrics = build_metrics(preds_esh, labels_esh, CLASS_NAMES, config, best_val_e, t_enriched)
metrics["robustness_gap"] = gap_e
metrics["accuracy_standard"] = round(float(acc_es), 4)
save_metrics(metrics, RESULTS_DIR, prefix="enriched_zero_shifted")

# keep enriched dataset alive for prune and gated runs, just free this model
del m_enriched
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""))

cells.append(md("""## 7b. EnrichedGCN with Prune, Both Splits
"""))

cells.append(code("""miss_train_p = make_missingness_transform(ENRICHED_GROUPS, drop_prob=DROP_PROB, strategy="prune", seed=SEED)
ds_e_train.transform = miss_train_p

g = torch.Generator(); g.manual_seed(SEED)
train_loader_p = DataLoader(ds_e_train, batch_size=BATCH_SIZE, shuffle=True, generator=g)

m_enriched_p = EnrichedGCN(ds_e_train.num_features, HIDDEN_DIM, ds_e_train.num_classes).to(device)
t0 = time.time()
m_enriched_p, _, _, best_val_ep = train_model(m_enriched_p, train_loader_p, val_loader_e, EPOCHS, LR, device)
t_enriched_p = time.time() - t0

preds_eps, labels_eps = predict(m_enriched_p, test_loader_e, device)
acc_eps = accuracy_score(labels_eps, preds_eps)

shifted_loader_p = make_shifted_loader(shifted_enriched, batch_size=BATCH_SIZE)
preds_epsh, labels_epsh = predict(m_enriched_p, shifted_loader_p, device)
acc_epsh = accuracy_score(labels_epsh, preds_epsh)

gap_ep = compute_robustness_gap(acc_eps, acc_epsh)
print(f"enriched_prune standard: {acc_eps:.4f}, shifted: {acc_epsh:.4f}, gap: {gap_ep:.4f}")

config = {"feature_type": "enriched", "missingness_strategy": "prune", "epochs": EPOCHS, "device": str(device)}
metrics = build_metrics(preds_epsh, labels_epsh, CLASS_NAMES, config, best_val_ep, t_enriched_p)
metrics["robustness_gap"] = gap_ep
metrics["accuracy_standard"] = round(float(acc_eps), 4)
save_metrics(metrics, RESULTS_DIR, prefix="enriched_prune_shifted")

del m_enriched_p, train_loader_p
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""))

cells.append(md("""## 8. GatedGCN, Both Splits
"""))

cells.append(code("""# reset train transform back to zero-fill since the prune cell mutated it
# (train_loader_e holds a live reference to ds_e_train, so transform change leaks)
ds_e_train.transform = make_missingness_transform(ENRICHED_GROUPS, drop_prob=DROP_PROB, strategy="zero", seed=SEED)
g = torch.Generator(); g.manual_seed(SEED)
train_loader_g = DataLoader(ds_e_train, batch_size=BATCH_SIZE, shuffle=True, generator=g)

m_gated = GatedGCN(
    ds_e_train.num_features, HIDDEN_DIM, ds_e_train.num_classes, num_groups=len(ENRICHED_GROUPS)
).to(device)
t0 = time.time()
m_gated, _, _, best_val_g = train_model_gated(m_gated, train_loader_g, val_loader_e, EPOCHS, LR, device)
t_gated = time.time() - t0

preds_gs, labels_gs = predict_gated(m_gated, test_loader_e, device)
acc_gs = accuracy_score(labels_gs, preds_gs)

# defensive: re-attach all-ones missingness mask in case shifted_enriched got rebuilt
for d in shifted_enriched:
    if not hasattr(d, "missingness_mask"):
        d.missingness_mask = torch.ones(1, len(ENRICHED_GROUPS))
shifted_loader_g = make_shifted_loader(shifted_enriched, batch_size=BATCH_SIZE)
preds_gsh, labels_gsh = predict_gated(m_gated, shifted_loader_g, device)
acc_gsh = accuracy_score(labels_gsh, preds_gsh)

# confusion matrix for gated model on shifted set (paper figure)
cm = confusion_matrix(labels_gsh, preds_gsh)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Gated GCN on Common Shifted - Confusion Matrix")
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
plt.savefig(os.path.join(RESULTS_DIR, "gated_shifted_confusion_matrix.png"), dpi=150)
plt.show()

gap_g = compute_robustness_gap(acc_gs, acc_gsh)
print(f"gated standard: {acc_gs:.4f}, shifted: {acc_gsh:.4f}, gap: {gap_g:.4f}")

config = {"feature_type": "enriched", "missingness_strategy": "zero_with_gating", "epochs": EPOCHS, "device": str(device)}
metrics = build_metrics(preds_gsh, labels_gsh, CLASS_NAMES, config, best_val_g, t_gated)
metrics["robustness_gap"] = gap_g
metrics["accuracy_standard"] = round(float(acc_gs), 4)
save_metrics(metrics, RESULTS_DIR, prefix="gated_shifted")
"""))

cells.append(md("""## 9. Robustness Gap Comparison Bar Chart
"""))

cells.append(code("""import pandas as pd

rows = [
    ("baseline (structural)", acc_standard, acc_shifted, gap),
    ("enriched (zero-fill)", acc_es, acc_esh, gap_e),
    ("enriched (prune)", acc_eps, acc_epsh, gap_ep),
    ("gated (novel)", acc_gs, acc_gsh, gap_g),
]
df = pd.DataFrame(rows, columns=["model", "standard", "shifted", "gap"])
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df["standard"], width, label="standard", color="steelblue")
ax.bar(x + width/2, df["shifted"], width, label="shifted (Common)", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(df["model"], rotation=15, ha="right")
ax.set_ylabel("Accuracy")
ax.set_title("Standard vs Shifted Test Accuracy by Model")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "robustness_gap_bar.png"), dpi=150)
plt.show()

df.to_csv(os.path.join(RESULTS_DIR, "robustness_summary.csv"), index=False)
print("saved", os.path.join(RESULTS_DIR, "robustness_summary.csv"))
"""))

cells.append(md("""## Summary

Four models trained on standard MalNet-Tiny, evaluated on both standard and the Common shifted test set. The robustness gap column tells us which model holds up best under covariate shift. These are the numbers that go into the paper.
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

out_path = os.path.join(REPO, "notebooks", "phase4_shifted.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote notebook with {len(cells)} cells to {out_path}")
