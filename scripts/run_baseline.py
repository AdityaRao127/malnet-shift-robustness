# runs the structural baseline gcn pipeline end-to-end
# saves baseline_metrics.json, baseline_results.csv, confusion_matrix.png, training_curve.png

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data_loader import load_malnet_splits, make_loaders
from features import make_pre_transform
from model import BaselineGCN
from train import build_metrics, run_eval, save_metrics, train_model


# ---- config ----
EPOCHS = 50
LR = 0.001
BATCH_SIZE = 64
HIDDEN_DIM = 64
MAX_DEGREE = 128
FEATURE_TYPE = "degree"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_class_names(data_dir):
    # parses class order from raw split files, matches pyg label assignment
    raw_type_dir = os.path.join(data_dir, "raw", "split_info_tiny", "type")
    y_map = {}
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        with open(os.path.join(raw_type_dir, split_file)) as f:
            for line in f.read().strip().split("\n"):
                mtype = line.split("/")[0]
                y_map.setdefault(mtype, len(y_map))
    return [name for name, _ in sorted(y_map.items(), key=lambda x: x[1])]


def save_confusion_matrix(test_labels, test_preds, class_names, output_dir):
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Baseline GCN Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_training_curve(train_losses, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, color="steelblue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, color="steelblue")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print(f"loading dataset with {FEATURE_TYPE} features...")
    t0 = time.time()
    pre_transform = make_pre_transform(feature_type=FEATURE_TYPE, max_degree=MAX_DEGREE)
    train_ds, val_ds, test_ds = load_malnet_splits(root=DATA_DIR, pre_transform=pre_transform)
    print(f"loaded in {time.time() - t0:.1f}s")
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    print(f"features: {train_ds.num_features}, classes: {train_ds.num_classes}")

    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

    class_names = get_class_names(DATA_DIR)
    print("classes:", class_names)

    model = BaselineGCN(
        in_channels=train_ds.num_features,
        hidden_channels=HIDDEN_DIM,
        num_classes=train_ds.num_classes,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params:,}")

    print(f"\ntraining for {EPOCHS} epochs...")
    t_train = time.time()
    model, train_losses, val_accs, best_val_acc = train_model(
        model, train_loader, val_loader, EPOCHS, LR, device, verbose_every=10
    )
    train_time = time.time() - t_train
    print(f"training done in {train_time:.1f}s, best val acc {best_val_acc:.4f}")

    print("\ntest set check:")
    test_preds, test_labels = run_eval(model, test_loader, device)

    training_config = {
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "max_degree": MAX_DEGREE,
        "feature_type": FEATURE_TYPE,
        "optimizer": "Adam",
        "device": str(device),
    }
    metrics = build_metrics(test_preds, test_labels, class_names, training_config, best_val_acc, train_time)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path, csv_path = save_metrics(metrics, RESULTS_DIR, prefix="baseline")
    print("saved", json_path)
    print("saved", csv_path)

    cm_path = save_confusion_matrix(test_labels, test_preds, class_names, RESULTS_DIR)
    print("saved", cm_path)

    curve_path = save_training_curve(train_losses, val_accs, RESULTS_DIR)
    print("saved", curve_path)

    print("")
    print("=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"accuracy: {metrics['accuracy']}")
    print(f"precision (macro): {metrics['precision_macro']}")
    print(f"recall (macro): {metrics['recall_macro']}")
    print(f"f1 (macro): {metrics['f1_macro']}")
    print(f"runtime: {train_time:.1f}s on {device}")
    print("=" * 50)


if __name__ == "__main__":
    main()
