"""
Copyright (C) 2026 Bitan Majumder

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ──────────────────────────────────────────────
# MODEL DEFINITIONS
# ──────────────────────────────────────────────

class GraphSAGE_NC(torch.nn.Module):
    """
    3-layer GraphSAGE for node classification.
    Mean aggregation, BatchNorm, Dropout.
    Must match the architecture used during Kaggle training exactly.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.skip1 = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip2 = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.skip3 = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels)
        self.drop  = dropout

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index) + self.skip1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.drop, training=self.training)
        h2 = self.conv2(h1, edge_index) + self.skip2(h1)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.drop, training=self.training)
        out = self.conv3(h2, edge_index) + self.skip3(h2)
        return out

    def embed(self, x, edge_index):
        """Return 128-dim penultimate layer embeddings."""
        h1 = F.relu(self.bn1(self.conv1(x, edge_index) + self.skip1(x)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index) + self.skip2(h1)))
        return h2


class GraphSAGE_LP(torch.nn.Module):
    """
    GraphSAGE encoder + dot-product decoder for link prediction.
    Must match the architecture used during Kaggle training exactly.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.skip1 = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip2 = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels)
        self.drop  = dropout

    def encode(self, x, edge_index):
        h1 = self.conv1(x, edge_index) + self.skip1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.drop, training=self.training)
        h2 = self.conv2(h1, edge_index) + self.skip2(h1)
        return h2

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


# ──────────────────────────────────────────────
# DATA BUILDER
# ──────────────────────────────────────────────

def build_pyg_data(nodes_df, edges_df,
                   label_col='louvain_label',
                   id_col='term'):
    """
    Build a PyTorch Geometric Data object from nodes + edges DataFrames.

    Returns:
        data         — PyG Data object with x, edge_index, edge_attr, y
        term_to_idx  — dict mapping node id → integer index
        num_classes  — number of unique community labels
        scaler       — fitted StandardScaler (for inference on new nodes)
        feature_cols — list of feature column names used in X
    """
    # ── Filter to labeled nodes only ──
    if label_col in nodes_df.columns:
        labeled = nodes_df[nodes_df[label_col].notna()].copy()
        labeled[label_col] = labeled[label_col].astype(int)
    else:
        labeled = nodes_df.copy()
        labeled[label_col] = 0

    # Ensure deterministic ordering for caching/reproducibility
    labeled[id_col] = labeled[id_col].astype(str)
    labeled = labeled.sort_values(by=id_col).reset_index(drop=True)
    term_to_idx = {str(t): i for i, t in enumerate(labeled[id_col].tolist())}
    N = len(labeled)

    # ── Node features ──
    # Use whichever feature columns are available
    possible_features = [
        'tf', 'df', 'tfidf', 'pmi_max',
        'degree', 'strength', 'mean_weight', 'total_coocc',
        'max_weight', 'min_weight', 'weight_std',
        'avg_neighbor_degree', 'triangle_count', 'clustering_coeff',
        'two_hop_reach', 'core_number', 'pagerank',
    ]
    feature_cols = [c for c in possible_features if c in labeled.columns]

    if not feature_cols:
        # Fallback: use degree computed from edges
        deg_map = {}
        for _, row in edges_df.iterrows():
            s, t = str(row.get('source', row.get('from', ''))), \
                   str(row.get('target', row.get('to', '')))
            deg_map[s] = deg_map.get(s, 0) + 1
            deg_map[t] = deg_map.get(t, 0) + 1
        labeled['degree'] = labeled[id_col].astype(str).map(deg_map).fillna(0)
        feature_cols = ['degree']

    X_raw = labeled[feature_cols].values.astype(np.float32)

    # Log-scale heavy-tailed columns
    for i, col in enumerate(feature_cols):
        if col in ('tf', 'df', 'tfidf', 'degree', 'strength', 'total_coocc',
                   'max_weight', 'min_weight', 'weight_std',
                   'avg_neighbor_degree', 'triangle_count', 'two_hop_reach',
                   'core_number'):
            X_raw[:, i] = np.log1p(X_raw[:, i])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    x        = torch.tensor(X_scaled, dtype=torch.float)

    # ── Labels ──
    le    = LabelEncoder()
    y_enc = le.fit_transform(labeled[label_col].values)
    y     = torch.tensor(y_enc, dtype=torch.long)
    num_classes = len(le.classes_)

    # ── Edges ──
    src_col = 'source' if 'source' in edges_df.columns else 'from'
    tgt_col = 'target' if 'target' in edges_df.columns else 'to'

    src_list, dst_list, w_list = [], [], []
    for _, row in edges_df.iterrows():
        s = str(row[src_col])
        t = str(row[tgt_col])
        if s in term_to_idx and t in term_to_idx:
            si, ti = term_to_idx[s], term_to_idx[t]
            src_list.extend([si, ti])
            dst_list.extend([ti, si])
            w = float(row.get('weight', 1.0))
            w_list.extend([w, w])

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    return data, term_to_idx, num_classes, scaler, feature_cols
