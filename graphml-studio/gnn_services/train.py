"""
Training pipeline for user-uploaded graphs.
Used when a user uploads their own CSV — not used for demo mode
(demo mode loads pre-trained weights directly).
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

from model import GraphSAGE_NC, GraphSAGE_LP

device = torch.device('cpu')


def make_stratified_node_split(data, train_frac=0.10, val_frac=0.15, seed=42):
    """
    Build stratified train/val/test masks so each community contributes
    at least one train example when possible.
    """
    n_nodes = int(data.num_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    labels = data.y.cpu().numpy()
    rng = np.random.default_rng(seed)

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)

        if n_cls == 1:
            train_n = 1
            val_n = 0
        elif n_cls == 2:
            train_n = 1
            val_n = 0
        else:
            train_n = max(1, int(round(n_cls * train_frac)))
            val_n = max(1, int(round(n_cls * val_frac)))
            if train_n + val_n >= n_cls:
                val_n = max(1, n_cls - train_n - 1)
            if train_n + val_n >= n_cls:
                train_n = max(1, n_cls - val_n - 1)

        train_idx = cls_idx[:train_n]
        val_idx = cls_idx[train_n:train_n + val_n]
        test_idx = cls_idx[train_n + val_n:]

        train_mask[train_idx] = True
        if len(val_idx):
            val_mask[val_idx] = True
        if len(test_idx):
            test_mask[test_idx] = True

    if not val_mask.any() and train_mask.sum().item() < n_nodes:
        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        if len(remaining):
            val_mask[remaining[0]] = True
            test_mask[remaining[0]] = False

    if not test_mask.any():
        candidates = train_mask.nonzero(as_tuple=False).view(-1)
        if len(candidates) > 1:
            moved = candidates[-1]
            train_mask[moved] = False
            test_mask[moved] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def train_node_classification(data, num_classes, epochs=100, hidden=128,
                               lr=0.005, dropout=0.3, patience=15,
                               train_frac=0.10, val_frac=0.15):
    """
    Train GraphSAGE for node classification on user-uploaded graph.
    Returns trained model + metrics dict.
    """
    # Stratified split gives the classifier a more stable view of every class.
    if data.num_nodes < 20:
        train_frac = max(train_frac, 0.20)
        val_frac = 0.20
    data = make_stratified_node_split(data, train_frac=train_frac, val_frac=val_frac)
    if not data.val_mask.any():
        data.val_mask = data.train_mask.clone()
    if not data.test_mask.any():
        remaining = (~data.train_mask & ~data.val_mask).nonzero(as_tuple=False).view(-1)
        if len(remaining):
            data.test_mask[remaining[0]] = True
        else:
            data.test_mask = data.val_mask.clone()

    # Class weights for imbalance
    train_labels  = data.y[data.train_mask].cpu()
    class_counts  = torch.bincount(train_labels, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)

    model     = GraphSAGE_NC(data.num_node_features, hidden, num_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8)

    best_val_f1  = 0
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask],
                               data.y[data.train_mask],
                               weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out   = model(data.x, data.edge_index)
            preds = out[data.val_mask].argmax(dim=1).cpu().numpy()
            truth = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(truth, preds, average='macro', zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test metrics
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
        truth = data.y[data.test_mask].cpu().numpy()

    metrics = {
        'accuracy': round(float(accuracy_score(truth, preds)), 4),
        'macro_f1': round(float(f1_score(truth, preds, average='macro',
                                          zero_division=0)), 4),
        'epochs_trained': epoch,
        'train_fraction': round(float(train_frac), 3),
        'val_fraction': round(float(val_frac), 3),
    }
    return model, metrics


def train_link_prediction(data, epochs=80, hidden=128,
                           lr=0.001, dropout=0.3, patience=15):
    """
    Train GraphSAGE for link prediction on user-uploaded graph.
    Returns trained model + metrics dict.
    """
    from sklearn.metrics import average_precision_score

    # Split edges (5% train, 15% val, 80% test by default)
    train_frac = 0.05
    val_frac = 0.15
    test_frac = 1.0 - train_frac - val_frac
    lp_transform = RandomLinkSplit(
        num_val=val_frac, num_test=test_frac,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = lp_transform(data)
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    model     = GraphSAGE_LP(data.num_node_features, hidden, 64, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_auc = 0
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(train_data.x, train_data.edge_index,
                     train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z    = model.encode(train_data.x, train_data.edge_index)
            out  = model.decode(z, val_data.edge_label_index)
            prob = torch.sigmoid(out).cpu().numpy()
            true = val_data.edge_label.cpu().numpy()
            val_auc = roc_auc_score(true, prob)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test metrics
    model.eval()
    with torch.no_grad():
        z    = model.encode(train_data.x, train_data.edge_index)
        out  = model.decode(z, test_data.edge_label_index)
        prob = torch.sigmoid(out).cpu().numpy()
        true = test_data.edge_label.cpu().numpy()

    metrics = {
        'auc_roc':       round(float(roc_auc_score(true, prob)), 4),
        'avg_precision': round(float(average_precision_score(true, prob)), 4),
        'epochs_trained': epoch,
    }
    return model, metrics
