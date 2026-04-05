# shared training and evaluation helpers
# used by scripts/run_baseline.py and the colab notebook

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report


def train_one_epoch(model, loader, optimizer, device):
    # one epoch over the loader, returns avg loss
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def run_eval(model, loader, device):
    # returns predicted and true label arrays
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, epochs, lr, device, verbose_every=10):
    # train loop with best val checkpoint kept in memory
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(loss)

        preds, labels = run_eval(model, val_loader, device)
        val_acc = accuracy_score(labels, preds)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose_every and epoch % verbose_every == 0:
            print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, train_losses, val_accs, best_val_acc


def build_metrics(test_preds, test_labels, class_names, training_config, best_val_acc, train_time):
    # canonical metrics dict, same shape across notebook and script
    report_dict = classification_report(
        test_labels, test_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    test_acc = accuracy_score(test_labels, test_preds)

    return {
        "accuracy": round(test_acc, 4),
        "precision_macro": round(report_dict["macro avg"]["precision"], 4),
        "recall_macro": round(report_dict["macro avg"]["recall"], 4),
        "f1_macro": round(report_dict["macro avg"]["f1-score"], 4),
        "precision_weighted": round(report_dict["weighted avg"]["precision"], 4),
        "recall_weighted": round(report_dict["weighted avg"]["recall"], 4),
        "f1_weighted": round(report_dict["weighted avg"]["f1-score"], 4),
        "best_val_acc": round(best_val_acc, 4),
        "per_class": {
            name: {
                "precision": round(report_dict[name]["precision"], 4),
                "recall": round(report_dict[name]["recall"], 4),
                "f1": round(report_dict[name]["f1-score"], 4),
                "support": int(report_dict[name]["support"]),
            }
            for name in class_names
        },
        "training": {**training_config, "train_time_seconds": round(train_time, 1)},
    }


def save_metrics(metrics, output_dir, prefix="baseline"):
    # writes {prefix}_metrics.json and {prefix}_results.csv
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_rows = [{"class": name, **vals} for name, vals in metrics["per_class"].items()]
    csv_rows.append({
        "class": "macro avg",
        "precision": metrics["precision_macro"],
        "recall": metrics["recall_macro"],
        "f1": metrics["f1_macro"],
        "support": sum(c["support"] for c in metrics["per_class"].values()),
    })
    csv_path = os.path.join(output_dir, f"{prefix}_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return json_path, csv_path
