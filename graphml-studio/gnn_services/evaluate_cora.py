"""
Quick benchmark evaluation for GraphML Studio GNN models on the Cora dataset.

Outputs:
    evaluation/cora/node_classification_metrics.json
    evaluation/cora/link_prediction_metrics.json
    evaluation/cora/node_classification_confusion_matrix.png
    evaluation/cora/node_classification_roc.png
    evaluation/cora/link_prediction_confusion_matrix.png
    evaluation/cora/link_prediction_roc.png
    evaluation/cora/evaluation_report.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit

from model import GraphSAGE_LP, GraphSAGE_NC


BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR.parent / "evaluation" / "cora"
DATA_DIR = BASE_DIR.parent / "data" / "cora"
DEVICE = torch.device("cpu")
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_cora():
    dataset = Planetoid(root=str(DATA_DIR), name="Cora", transform=NormalizeFeatures())
    return dataset, dataset[0].to(DEVICE)


def train_node_classifier(data, num_classes, hidden=128, epochs=200, lr=0.01, dropout=0.35):
    set_seed()
    model = GraphSAGE_NC(data.num_node_features, hidden, num_classes, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_state = None
    best_val = -1.0
    patience = 25
    patience_count = 0
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[data.val_mask]
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def evaluate_node_classifier(model, data, class_names):
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[data.test_mask]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        truth = data.y[data.test_mask].cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(truth, preds)
    truth_bin = label_binarize(truth, classes=np.arange(len(class_names)))
    roc_auc_macro = roc_auc_score(truth_bin, probs, multi_class="ovr", average="macro")
    roc_auc_micro = roc_auc_score(truth_bin, probs, multi_class="ovr", average="micro")

    metrics = {
        "dataset": "Cora",
        "task": "node_classification",
        "test_nodes": int(len(truth)),
        "accuracy": round(float(accuracy_score(truth, preds)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "roc_auc_macro_ovr": round(float(roc_auc_macro), 4),
        "roc_auc_micro_ovr": round(float(roc_auc_micro), 4),
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title("Cora Node Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "node_classification_confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(truth_bin[:, idx], probs[:, idx])
        ax.plot(fpr, tpr, lw=1.5, label=f"{class_name}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Cora Node Classification ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "node_classification_roc.png", dpi=200)
    plt.close(fig)

    return metrics


def train_link_predictor(data, hidden=128, out_channels=96, epochs=120, lr=0.0015, dropout=0.25):
    set_seed()
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data.cpu())
    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    model = GraphSAGE_LP(data.num_node_features, hidden, out_channels, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_state = None
    best_val = -1.0
    patience = 20
    patience_count = 0
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_index)
            val_logits = model.decode(z, val_data.edge_label_index)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_true = val_data.edge_label.cpu().numpy()
            val_auc = roc_auc_score(val_true, val_prob)

        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, train_data, test_data, best_epoch


def evaluate_link_predictor(model, train_data, test_data):
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        logits = model.decode(z, test_data.edge_label_index)
        probs = torch.sigmoid(logits).cpu().numpy()
        truth = test_data.edge_label.cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(truth, preds)
    auc_roc = roc_auc_score(truth, probs)
    avg_precision = average_precision_score(truth, probs)

    metrics = {
        "dataset": "Cora",
        "task": "link_prediction",
        "test_edges": int(len(truth)),
        "accuracy": round(float(accuracy_score(truth, preds)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc_roc), 4),
        "average_precision": round(float(avg_precision), 4),
        "confusion_matrix": cm.tolist(),
    }

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No link", "Link"]).plot(
        ax=ax, cmap="Greens", colorbar=False
    )
    ax.set_title("Cora Link Prediction Confusion Matrix")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "link_prediction_confusion_matrix.png", dpi=200)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(truth, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Cora Link Prediction ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "link_prediction_roc.png", dpi=200)
    plt.close(fig)

    return metrics


def write_report(node_metrics, link_metrics, nc_epoch, lp_epoch):
    report = f"""# Cora Evaluation Report

Generated: {datetime.now().isoformat(timespec='seconds')}

## Setup

- Dataset: `Cora` (Planetoid benchmark)
- Node model: `GraphSAGE_NC`
- Link model: `GraphSAGE_LP`
- Device: `CPU`
- Seed: `{SEED}`

## Node Classification

- Best epoch: `{nc_epoch}`
- Accuracy: `{node_metrics['accuracy']:.4f}`
- Precision (macro): `{node_metrics['precision_macro']:.4f}`
- Recall (macro): `{node_metrics['recall_macro']:.4f}`
- F1-score (macro): `{node_metrics['f1_macro']:.4f}`
- ROC-AUC (macro, OvR): `{node_metrics['roc_auc_macro_ovr']:.4f}`
- ROC-AUC (micro, OvR): `{node_metrics['roc_auc_micro_ovr']:.4f}`

Artifacts:
- `node_classification_confusion_matrix.png`
- `node_classification_roc.png`

## Link Prediction

- Best epoch: `{lp_epoch}`
- Accuracy: `{link_metrics['accuracy']:.4f}`
- Precision: `{link_metrics['precision']:.4f}`
- Recall: `{link_metrics['recall']:.4f}`
- F1-score: `{link_metrics['f1_score']:.4f}`
- ROC-AUC: `{link_metrics['roc_auc']:.4f}`
- Average Precision: `{link_metrics['average_precision']:.4f}`

Artifacts:
- `link_prediction_confusion_matrix.png`
- `link_prediction_roc.png`

## Brief Interpretation

- The node classification scores show how well GraphSAGE separates paper topics on the held-out Cora test nodes.
- The link prediction scores show how reliably the GraphSAGE encoder can recover held-out citation edges.
- Use this benchmark as an external evaluation companion to the in-app demo workflow. It gives you standardized metrics that the live app does not currently expose.
"""
    (EVAL_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")


def main():
    ensure_dirs()
    set_seed()
    dataset, data = load_cora()
    class_names = [f"Class {i}" for i in range(dataset.num_classes)]

    node_model, nc_epoch = train_node_classifier(data, dataset.num_classes)
    node_metrics = evaluate_node_classifier(node_model, data, class_names)
    (EVAL_DIR / "node_classification_metrics.json").write_text(
        json.dumps(node_metrics, indent=2), encoding="utf-8"
    )

    link_model, train_data, test_data, lp_epoch = train_link_predictor(data)
    link_metrics = evaluate_link_predictor(link_model, train_data, test_data)
    (EVAL_DIR / "link_prediction_metrics.json").write_text(
        json.dumps(link_metrics, indent=2), encoding="utf-8"
    )

    write_report(node_metrics, link_metrics, nc_epoch, lp_epoch)
    print(f"Evaluation complete. Outputs saved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
