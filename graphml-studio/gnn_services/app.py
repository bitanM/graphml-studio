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

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import os
import json
import traceback
import hashlib
from datetime import datetime

from model import GraphSAGE_NC, GraphSAGE_LP, build_pyg_data
from train import train_node_classification, train_link_prediction
from sklearn.manifold import TSNE

app = Flask(__name__)
CORS(app)

# ── Paths ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device('cpu')  # CPU-only for local deployment
MIN_INDUCTIVE_CONNECTIONS = 4

# ── In-memory state ──
state = {
    'user_data':      None,
    'user_model_nc':  None,
    'user_model_lp':  None,
    'user_embeddings': None,
    'user_nodes_df':  None,
    'user_adjacency': None,
    'user_term_to_idx': None,
    'user_scaler': None,
    'user_feature_cols': None,
    'user_graph_hash': None,
    'user_cached': False,
    'user_nc_metrics': None,
    'user_training_config': None,
}

def graph_hash(nodes_list, edges_list):
    """
    Deterministic hash of graph structure + basic labels.
    Uses undirected edge normalization and stable ordering.
    """
    node_keys = []
    for n in nodes_list or []:
        nid = str(n.get('id', ''))
        if not nid:
            continue
        comm = n.get('community', '')
        node_keys.append(f"{nid}|{comm}")
    node_keys.sort()

    edge_keys = []
    for e in edges_list or []:
        s = str(e.get('source', e.get('from', '')))
        t = str(e.get('target', e.get('to', '')))
        if not s or not t:
            continue
        if s > t:
            s, t = t, s
        w = e.get('weight', 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        edge_keys.append(f"{s}|{t}|{w:.6f}")
    edge_keys.sort()

    h = hashlib.sha256()
    for k in node_keys:
        h.update(k.encode('utf-8'))
        h.update(b'\n')
    h.update(b'--edges--\n')
    for k in edge_keys:
        h.update(k.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()[:12]

def cache_paths(graph_hash_id):
    return {
        'nc': os.path.join(MODELS_DIR, f'cache_{graph_hash_id}_nc.pt'),
        'lp': os.path.join(MODELS_DIR, f'cache_{graph_hash_id}_lp.pt'),
        'meta': os.path.join(MODELS_DIR, f'cache_{graph_hash_id}.json'),
    }

def load_cache_meta(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def make_tsne(n_components, perplexity, random_state, n_iter):
    """
    Use max_iter when available (sklearn >=1.5) to avoid FutureWarning,
    fallback to n_iter for older versions.
    """
    try:
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=n_iter
        )
    except TypeError:
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=n_iter
        )

def parse_connections(raw_connections):
    """
    Normalize connections input into (ids, weights).
    Supports list[str] or list[{'id':..., 'weight':...}].
    """
    ids = []
    weights = []
    for item in raw_connections or []:
        if isinstance(item, dict):
            nid = str(item.get('id') or item.get('node') or item.get('name') or '').strip()
            w = item.get('weight', 1.0)
        else:
            nid = str(item).strip()
            w = 1.0
        if not nid:
            continue
        try:
            w = float(w)
        except Exception:
            w = 1.0
        ids.append(nid)
        weights.append(w)
    return ids, weights

def collapse_known_connections(conn_ids, conn_weights, valid_ids=None):
    """
    Deduplicate connection IDs while preserving order and summing repeated weights.
    """
    merged = {}
    order = []
    for nid, weight in zip(conn_ids or [], conn_weights or []):
        key = str(nid).strip()
        if not key:
            continue
        if valid_ids is not None and key not in valid_ids:
            continue
        if key not in merged:
            merged[key] = 0.0
            order.append(key)
        merged[key] += float(weight)
    return order, [merged[nid] for nid in order]

def build_graph_structures(node_ids, edges_df):
    """
    Build adjacency + weighted adjacency maps from an undirected edge list.
    """
    adjacency = {str(nid): set() for nid in node_ids}
    weighted_adjacency = {str(nid): {} for nid in node_ids}

    if edges_df is None or edges_df.empty:
        return adjacency, weighted_adjacency

    src_col = 'source' if 'source' in edges_df.columns else 'from'
    tgt_col = 'target' if 'target' in edges_df.columns else 'to'

    for _, row in edges_df.iterrows():
        src = str(row.get(src_col, '')).strip()
        tgt = str(row.get(tgt_col, '')).strip()
        if not src or not tgt or src == tgt:
            continue
        try:
            weight = float(row.get('weight', 1.0))
        except Exception:
            weight = 1.0

        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)
        weighted_adjacency.setdefault(src, {})
        weighted_adjacency.setdefault(tgt, {})
        weighted_adjacency[src][tgt] = weighted_adjacency[src].get(tgt, 0.0) + weight
        weighted_adjacency[tgt][src] = weighted_adjacency[tgt].get(src, 0.0) + weight

    return adjacency, weighted_adjacency

def compute_core_numbers(adjacency):
    """
    Lightweight k-core decomposition for undirected graphs.
    """
    work = {node: set(neighbors) for node, neighbors in adjacency.items()}
    remaining = set(work.keys())
    core = {node: 0 for node in work}
    k = 0

    while remaining:
        changed = True
        while changed:
            changed = False
            removable = [node for node in list(remaining) if len(work[node] & remaining) <= k]
            if removable:
                changed = True
                for node in removable:
                    remaining.remove(node)
                    core[node] = k
        k += 1

    return core

def compute_weighted_pagerank(weighted_adjacency, damping=0.85, steps=40):
    """
    Weighted PageRank without adding a NetworkX dependency.
    """
    nodes = list(weighted_adjacency.keys())
    if not nodes:
        return {}

    n = len(nodes)
    rank = {node: 1.0 / n for node in nodes}

    for _ in range(steps):
        new_rank = {node: (1.0 - damping) / n for node in nodes}
        dangling_mass = 0.0

        for node, nbrs in weighted_adjacency.items():
            out_weight = float(sum(nbrs.values()))
            if out_weight <= 0:
                dangling_mass += rank[node]
                continue
            share = damping * rank[node] / out_weight
            for nbr, weight in nbrs.items():
                new_rank[nbr] += share * float(weight)

        if dangling_mass:
            spread = damping * dangling_mass / n
            for node in nodes:
                new_rank[node] += spread

        total = sum(new_rank.values()) or 1.0
        rank = {node: val / total for node, val in new_rank.items()}

    return rank

def enrich_user_nodes_df(nodes_df, edges_df, id_col='id'):
    """
    Add structural features for user-uploaded graphs so the classifier learns
    from weighted connectivity patterns instead of degree alone.
    """
    nodes_df = nodes_df.copy()
    if id_col not in nodes_df.columns:
        if 'term' in nodes_df.columns:
            nodes_df = nodes_df.rename(columns={'term': id_col})
        else:
            raise ValueError(f'Missing required node id column: {id_col}')

    nodes_df[id_col] = nodes_df[id_col].astype(str)

    if 'community' not in nodes_df.columns and 'communities' in nodes_df.columns:
        nodes_df['community'] = nodes_df['communities'].apply(
            lambda v: int(v.get('louvain', 0)) if isinstance(v, dict) else 0
        )

    derived_cols = [
        'degree', 'strength', 'mean_weight',
        'max_weight', 'min_weight', 'weight_std', 'total_coocc',
        'avg_neighbor_degree', 'triangle_count', 'clustering_coeff',
        'two_hop_reach', 'core_number', 'pagerank',
    ]

    if edges_df is None or edges_df.empty:
        for col in derived_cols:
            if col in nodes_df.columns:
                nodes_df[col] = pd.to_numeric(nodes_df[col], errors='coerce').fillna(0.0)
            else:
                nodes_df[col] = 0.0
        return nodes_df

    src_col = 'source' if 'source' in edges_df.columns else 'from'
    tgt_col = 'target' if 'target' in edges_df.columns else 'to'
    edge_df = edges_df.copy()
    edge_df[src_col] = edge_df[src_col].astype(str)
    edge_df[tgt_col] = edge_df[tgt_col].astype(str)
    if 'weight' in edge_df.columns:
        edge_df['weight'] = pd.to_numeric(edge_df['weight'], errors='coerce').fillna(1.0)
    else:
        edge_df['weight'] = 1.0

    adjacency, weighted_adjacency = build_graph_structures(nodes_df[id_col].tolist(), edge_df)
    core_numbers = compute_core_numbers(adjacency)
    pagerank_scores = compute_weighted_pagerank(weighted_adjacency)

    edge_long = pd.concat([
        edge_df[[src_col, 'weight']].rename(columns={src_col: id_col}),
        edge_df[[tgt_col, 'weight']].rename(columns={tgt_col: id_col}),
    ], ignore_index=True)

    agg = edge_long.groupby(id_col).agg(
        degree=('weight', 'size'),
        strength=('weight', 'sum'),
        mean_weight=('weight', 'mean'),
        max_weight=('weight', 'max'),
        min_weight=('weight', 'min'),
        weight_std=('weight', 'std'),
    )
    agg['weight_std'] = agg['weight_std'].fillna(0.0)
    agg['total_coocc'] = agg['strength']

    structural_rows = []
    for node_id in nodes_df[id_col].tolist():
        nid = str(node_id)
        neighbors = adjacency.get(nid, set())
        degree = len(neighbors)
        neighbor_degrees = [len(adjacency.get(nb, set())) for nb in neighbors]
        avg_neighbor_degree = float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0

        shared_links = 0
        for nb in neighbors:
            shared_links += len(neighbors & adjacency.get(nb, set()))
        triangle_count = float(shared_links / 2.0)

        if degree > 1:
            clustering_coeff = float((2.0 * triangle_count) / (degree * (degree - 1)))
        else:
            clustering_coeff = 0.0

        two_hop = set()
        for nb in neighbors:
            two_hop.update(adjacency.get(nb, set()))
        two_hop.discard(nid)
        two_hop.difference_update(neighbors)

        structural_rows.append({
            id_col: nid,
            'avg_neighbor_degree': avg_neighbor_degree,
            'triangle_count': triangle_count,
            'clustering_coeff': clustering_coeff,
            'two_hop_reach': float(len(two_hop)),
            'core_number': float(core_numbers.get(nid, 0)),
            'pagerank': float(pagerank_scores.get(nid, 0.0)),
        })

    structural_df = pd.DataFrame(structural_rows).set_index(id_col)

    nodes_df = nodes_df.set_index(id_col, drop=False)
    for col in derived_cols:
        existing = pd.to_numeric(nodes_df[col], errors='coerce') if col in nodes_df.columns else pd.Series(index=nodes_df.index, dtype=float)
        incoming = agg[col] if col in agg.columns else pd.Series(index=nodes_df.index, dtype=float)
        if col in structural_df.columns:
            incoming = structural_df[col].combine_first(incoming)
        nodes_df[col] = existing.fillna(incoming).fillna(0.0)

    return nodes_df.reset_index(drop=True)

def build_new_node_feature(feature_cols, scaler, conn_ids, conn_weights, nodes_df=None, adjacency=None):
    """
    Build a single-node feature vector matching training feature_cols,
    using connection-derived statistics (degree, strength, mean_weight, etc.).
    """
    degree = len(conn_ids)
    strength = float(sum(conn_weights)) if conn_weights else 0.0
    mean_weight = (strength / degree) if degree > 0 else 0.0
    total_coocc = strength
    max_weight = float(max(conn_weights)) if conn_weights else 0.0
    min_weight = float(min(conn_weights)) if conn_weights else 0.0
    weight_std = float(np.std(conn_weights)) if conn_weights else 0.0
    valid_connections = [str(nid) for nid in conn_ids]
    node_lookup = None
    if nodes_df is not None and not nodes_df.empty:
        key_col = 'id' if 'id' in nodes_df.columns else nodes_df.columns[0]
        node_lookup = nodes_df.set_index(key_col, drop=False)
        valid_connections = [nid for nid in valid_connections if nid in node_lookup.index]

    if adjacency is None:
        adjacency = {}

    avg_neighbor_degree = 0.0
    triangle_count = 0.0
    clustering_coeff = 0.0
    two_hop_reach = 0.0
    core_number = float(min(degree, 1)) if degree else 0.0
    pagerank = 0.0

    if valid_connections and node_lookup is not None:
        conn_frame = node_lookup.loc[valid_connections]
        if isinstance(conn_frame, pd.Series):
            conn_frame = conn_frame.to_frame().T

        if 'degree' in conn_frame.columns:
            avg_neighbor_degree = float(pd.to_numeric(conn_frame['degree'], errors='coerce').fillna(0.0).mean())
        if 'core_number' in conn_frame.columns:
            core_vals = pd.to_numeric(conn_frame['core_number'], errors='coerce').fillna(0.0)
            core_number = float(min(degree, max(1.0, round(float(core_vals.mean()))))) if degree else 0.0
        if 'pagerank' in conn_frame.columns:
            pagerank = float(pd.to_numeric(conn_frame['pagerank'], errors='coerce').fillna(0.0).mean())

    if valid_connections and adjacency:
        conn_set = set(valid_connections)
        linked_pairs = 0
        for nid in conn_set:
            linked_pairs += len(conn_set & adjacency.get(nid, set()))
        triangle_count = float(linked_pairs / 2.0)
        if degree > 1:
            clustering_coeff = float((2.0 * triangle_count) / (degree * (degree - 1)))

        two_hop = set()
        for nid in conn_set:
            two_hop.update(adjacency.get(nid, set()))
        two_hop.difference_update(conn_set)
        two_hop_reach = float(len(two_hop))

    base = {
        'degree': degree,
        'strength': strength,
        'mean_weight': mean_weight,
        'total_coocc': total_coocc,
        'max_weight': max_weight,
        'min_weight': min_weight,
        'weight_std': weight_std,
        'avg_neighbor_degree': avg_neighbor_degree,
        'triangle_count': triangle_count,
        'clustering_coeff': clustering_coeff,
        'two_hop_reach': two_hop_reach,
        'core_number': core_number,
        'pagerank': pagerank,
        'tf': 0.0,
        'df': 0.0,
        'tfidf': 0.0,
        'pmi_max': 0.0,
    }

    vec = []
    for col in feature_cols:
        vec.append(float(base.get(col, 0.0)))
    vec = np.array(vec, dtype=np.float32).reshape(1, -1)

    for i, col in enumerate(feature_cols):
        if col in ('tf', 'df', 'tfidf', 'degree', 'strength', 'total_coocc',
                   'max_weight', 'min_weight', 'weight_std',
                   'avg_neighbor_degree', 'triangle_count', 'two_hop_reach',
                   'core_number'):
            vec[:, i] = np.log1p(vec[:, i])

    vec_scaled = scaler.transform(vec).astype(np.float32)
    return torch.tensor(vec_scaled, dtype=torch.float)

def build_inductive_graph(data, term_to_idx, new_feat, conn_ids):
    """
    Extend graph with a new node connected to known neighbors.
    Returns (x_ext, edge_index_ext, new_idx, neighbor_idxs).
    """
    new_idx = data.num_nodes
    neighbor_idxs = []
    src = []
    dst = []
    for nid in conn_ids:
        if nid not in term_to_idx:
            continue
        ni = term_to_idx[nid]
        neighbor_idxs.append(ni)
        src.extend([new_idx, ni])
        dst.extend([ni, new_idx])

    if not neighbor_idxs:
        return None, None, None, []

    x_ext = torch.cat([data.x, new_feat.to(data.x.device)], dim=0)
    add_edges = torch.tensor([src, dst], dtype=torch.long, device=data.edge_index.device)
    edge_index_ext = torch.cat([data.edge_index, add_edges], dim=1)
    return x_ext, edge_index_ext, new_idx, neighbor_idxs

def get_neighbors(edge_index, node_idx):
    src = edge_index[0]
    dst = edge_index[1]
    mask = (src == node_idx)
    if mask.sum().item() == 0:
        return set()
    return set(dst[mask].tolist())

# ──────────────────────────────────────────────
# ROOT
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'ok',
        'service': 'graphml-studio-gnn',
        'endpoints': [
            '/health',
            '/gnn/train',
            '/gnn/user/predict-node',
            '/gnn/user/predict-edge',
            '/gnn/user/embeddings',
        ],
    })

# HEALTH CHECK
# ──────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'torch_version': torch.__version__,
        'device': str(device),
    })


# ──────────────────────────────────────────────
# USER UPLOAD — Train GNN on custom graph
# ──────────────────────────────────────────────
@app.route('/gnn/train', methods=['POST'])
def user_train():
    try:
        payload  = request.json or {}
        nodes_list = payload.get('nodes', [])
        edges_list = payload.get('edges', [])
        mode = str(payload.get('mode', 'full')).lower()
        force = bool(payload.get('force', False))

        if not nodes_list or not edges_list:
            return jsonify({'error': 'nodes and edges required.'}), 400

        # Training hyperparams (fast vs full)
        nc_epochs = int(payload.get('nc_epochs', 32 if mode == 'fast' else 120))
        lp_epochs = int(payload.get('lp_epochs', 30 if mode == 'fast' else 100))
        hidden    = int(payload.get('hidden', 96 if mode == 'fast' else 128))
        lp_out    = int(payload.get('lp_out', 64 if mode == 'fast' else 96))
        nc_train_frac = float(payload.get('nc_train_frac', 0.08 if mode == 'fast' else 0.15))
        nc_val_frac = float(payload.get('nc_val_frac', 0.15))
        train_tag = str(round(nc_train_frac, 3)).replace('.', 'p')
        val_tag = str(round(nc_val_frac, 3)).replace('.', 'p')

        # Cache key depends on graph + mode + hyperparams
        gh = graph_hash(nodes_list, edges_list)
        cache_id = f"{gh}_{mode}_{hidden}_{nc_epochs}_{lp_epochs}_{train_tag}_{val_tag}"
        paths = cache_paths(cache_id)
        meta = load_cache_meta(paths['meta'])

        nodes_df = pd.DataFrame(nodes_list)
        edges_df = pd.DataFrame(edges_list)

        # Rename columns to standard format if needed
        if 'from' in edges_df.columns:
            edges_df = edges_df.rename(columns={'from': 'source', 'to': 'target'})

        nodes_df = enrich_user_nodes_df(nodes_df, edges_df, id_col='id')
        adjacency, _ = build_graph_structures(nodes_df['id'].tolist(), edges_df)

        # Need at least degree as a feature — compute if missing
        if 'degree' not in nodes_df.columns:
            deg_map = {}
            for _, row in edges_df.iterrows():
                deg_map[row['source']] = deg_map.get(row['source'], 0) + 1
                deg_map[row['target']] = deg_map.get(row['target'], 0) + 1
            nodes_df['degree'] = nodes_df['id'].map(deg_map).fillna(0)

        print(f'Training on user graph: {len(nodes_df)} nodes, {len(edges_df)} edges')

        data, term_to_idx, num_classes, scaler, feature_cols = build_pyg_data(
            nodes_df, edges_df, label_col='community', id_col='id'
        )

        if num_classes < 2:
            return jsonify({'error': 'Need at least 2 communities for classification.'}), 400

        data = data.to(device)

        # ── Cache hit: load models and skip training ──
        if (not force and meta and os.path.exists(paths['nc']) and os.path.exists(paths['lp'])):
            model_nc = GraphSAGE_NC(
                in_channels=data.num_node_features,
                hidden_channels=meta.get('hidden', hidden),
                out_channels=num_classes,
            ).to(device)
            model_nc.load_state_dict(torch.load(paths['nc'], map_location=device))
            model_nc.eval()

            model_lp = GraphSAGE_LP(
                in_channels=data.num_node_features,
                hidden_channels=meta.get('hidden', hidden),
                out_channels=meta.get('lp_out', lp_out),
            ).to(device)
            model_lp.load_state_dict(torch.load(paths['lp'], map_location=device))
            model_lp.eval()

            state['user_data']     = data
            state['user_model_nc'] = model_nc
            state['user_model_lp'] = model_lp
            state['user_adjacency'] = adjacency
            state['user_term_to_idx'] = term_to_idx
            state['user_nodes_df'] = nodes_df
            state['user_scaler'] = scaler
            state['user_feature_cols'] = feature_cols
            state['user_graph_hash'] = cache_id
            state['user_cached'] = True
            state['user_nc_metrics'] = meta.get('nc_metrics', {})
            state['user_training_config'] = {
                'mode': meta.get('mode', mode),
                'hidden': meta.get('hidden', hidden),
                'nc_epochs': meta.get('nc_epochs', nc_epochs),
                'lp_epochs': meta.get('lp_epochs', lp_epochs),
                'nc_train_frac': meta.get('nc_train_frac', nc_train_frac),
                'nc_val_frac': meta.get('nc_val_frac', nc_val_frac),
                'lp_out': meta.get('lp_out', lp_out),
                'nc_inference_scope': meta.get('nc_inference_scope', 'full_graph'),
            }

            with torch.no_grad():
                embeddings = model_nc.embed(data.x, data.edge_index).cpu().numpy()
            tsne   = make_tsne(n_components=2, perplexity=min(40, len(nodes_df)//4),
                               random_state=42, n_iter=500)
            emb_2d = tsne.fit_transform(embeddings)
            state['user_embeddings'] = emb_2d

            return jsonify({
                'status':        'cached',
                'cached':        True,
                'graph_hash':    cache_id,
                'nodes':         len(nodes_df),
                'edges':         len(edges_df),
                'communities':   num_classes,
                'nc_metrics':    meta.get('nc_metrics', {}),
                'lp_metrics':    meta.get('lp_metrics', {}),
                'mode':          meta.get('mode', mode),
                'training':      state['user_training_config'],
            })

        # Train NC
        model_nc, nc_metrics = train_node_classification(
            data, num_classes,
            epochs=nc_epochs,
            hidden=hidden,
            train_frac=nc_train_frac,
            val_frac=nc_val_frac,
        )
        model_nc.eval()
        state['user_data']     = data
        state['user_model_nc'] = model_nc
        state['user_term_to_idx'] = term_to_idx
        state['user_nodes_df'] = nodes_df
        state['user_adjacency'] = adjacency
        state['user_scaler'] = scaler
        state['user_feature_cols'] = feature_cols
        state['user_graph_hash'] = cache_id
        state['user_cached'] = False
        state['user_nc_metrics'] = nc_metrics
        state['user_training_config'] = {
            'mode': mode,
            'hidden': hidden,
            'nc_epochs': nc_epochs,
            'lp_epochs': lp_epochs,
            'nc_train_frac': nc_train_frac,
            'nc_val_frac': nc_val_frac,
            'lp_out': lp_out,
            'nc_inference_scope': 'full_graph',
        }

        # Compute embeddings + t-SNE
        with torch.no_grad():
            embeddings = model_nc.embed(data.x, data.edge_index).cpu().numpy()
        tsne   = make_tsne(n_components=2,
                           perplexity=min(40, len(nodes_df)//4),
                           random_state=42,
                           n_iter=500)
        emb_2d = tsne.fit_transform(embeddings)
        state['user_embeddings'] = emb_2d

        # Train LP
        model_lp, lp_metrics = train_link_prediction(
            data, epochs=lp_epochs, hidden=hidden, out_channels=lp_out
        )
        model_lp.eval()
        state['user_model_lp'] = model_lp

        # Save cache
        try:
            torch.save(model_nc.state_dict(), paths['nc'])
            torch.save(model_lp.state_dict(), paths['lp'])
            meta_out = {
                'graph_hash':  cache_id,
                'mode':        mode,
                'hidden':      hidden,
                'lp_out':      lp_out,
                'nc_epochs':   nc_epochs,
                'lp_epochs':   lp_epochs,
                'nc_train_frac': nc_train_frac,
                'nc_val_frac': nc_val_frac,
                'nc_inference_scope': 'full_graph',
                'nodes':       len(nodes_df),
                'edges':       len(edges_df),
                'communities': num_classes,
                'nc_metrics':  nc_metrics,
                'lp_metrics':  lp_metrics,
                'trained_at':  datetime.utcnow().isoformat() + 'Z',
            }
            with open(paths['meta'], 'w') as f:
                json.dump(meta_out, f)
        except Exception:
            pass

        return jsonify({
            'status':        'trained',
            'cached':        False,
            'graph_hash':    cache_id,
            'nodes':         len(nodes_df),
            'edges':         len(edges_df),
            'communities':   num_classes,
            'nc_metrics':    nc_metrics,
            'lp_metrics':    lp_metrics,
            'mode':          mode,
            'training':      state['user_training_config'],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# USER — Get embeddings
# ──────────────────────────────────────────────
@app.route('/gnn/user/embeddings', methods=['GET'])
def user_embeddings():
    if state['user_embeddings'] is None:
        return jsonify({'error': 'No user model trained yet.'}), 400
    try:
        emb_2d   = state['user_embeddings']
        nodes_df = state['user_nodes_df']
        id_col   = 'id' if 'id' in nodes_df.columns else nodes_df.columns[0]

        points = []
        for i, row in nodes_df.iterrows():
            if i >= len(emb_2d):
                break
            points.append({
                'id':        str(row[id_col]),
                'x':         float(emb_2d[i, 0]),
                'y':         float(emb_2d[i, 1]),
                'community': int(row.get('community', 0)),
            })

        return jsonify({'points': points})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# USER — Node prediction
# ──────────────────────────────────────────────
@app.route('/gnn/user/predict-node', methods=['POST'])
def user_predict_node():
    if state['user_model_nc'] is None:
        return jsonify({'error': 'No user model trained yet.'}), 400
    try:
        payload = request.json or {}
        node_id = str(payload.get('nodeId') or payload.get('newNodeId') or '').strip()
        term_to_idx  = state['user_term_to_idx']
        model_nc     = state['user_model_nc']
        data         = state['user_data']
        scaler       = state['user_scaler']
        feature_cols = state['user_feature_cols']
        nodes_df      = state['user_nodes_df']
        adjacency     = state.get('user_adjacency')

        if not node_id:
            return jsonify({'error': 'nodeId required.'}), 400

        # Case 1: existing node
        if node_id in term_to_idx:
            idx = term_to_idx[node_id]
            model_nc.eval()
            with torch.no_grad():
                logits = model_nc(data.x, data.edge_index)
                probs  = torch.softmax(logits[idx], dim=0).cpu().numpy()

        # Case 2: new node (inductive) — requires known connections
        else:
            conn_raw = payload.get('connections') or payload.get('newConnections') or []
            conn_ids, conn_weights = parse_connections(conn_raw)
            conn_ids, conn_weights = collapse_known_connections(conn_ids, conn_weights, set(term_to_idx.keys()))
            if len(conn_ids) < MIN_INDUCTIVE_CONNECTIONS:
                return jsonify({
                    'error': (
                        f'Node "{node_id}" is unseen. Provide at least '
                        f'{MIN_INDUCTIVE_CONNECTIONS} known connections from the graph.'
                    )
                }), 400
            if scaler is None or feature_cols is None:
                return jsonify({'error': 'Model metadata missing. Retrain the GNN.'}), 400

            new_feat = build_new_node_feature(
                feature_cols, scaler, conn_ids, conn_weights, nodes_df, adjacency
            )
            x_ext, edge_index_ext, idx, _ = build_inductive_graph(
                data, term_to_idx, new_feat, conn_ids
            )
            if idx is None:
                return jsonify({'error': 'No valid connections found in graph.'}), 400

            model_nc.eval()
            with torch.no_grad():
                logits = model_nc(x_ext, edge_index_ext)
                probs  = torch.softmax(logits[idx], dim=0).cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100
        top5 = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)[:5]
        top5_result = [
            {
                'community': c,
                'theme': f'Community {c}',
                'probability': round(p * 100, 2),
            }
            for c, p in top5
        ]

        return jsonify({
            'nodeId':               node_id,
            'predicted_community':  pred_class,
            'confidence':           round(confidence, 2),
            'top5':                 top5_result,
            'nc_metrics':           state.get('user_nc_metrics'),
            'training':             state.get('user_training_config'),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ————————————————————————————————————————————————————————
# USER — Link prediction (GraphSAGE)
# ————————————————————————————————————————————————————————
@app.route('/gnn/user/predict-edge', methods=['POST'])
def user_predict_edge():
    if state['user_model_lp'] is None:
        return jsonify({'error': 'No user model trained yet.'}), 400
    try:
        payload = request.json or {}
        node_id = str(payload.get('nodeId') or payload.get('newNodeId') or '').strip()
        top_k   = int(payload.get('topK', 8))

        if not node_id:
            return jsonify({'error': 'nodeId required.'}), 400

        term_to_idx  = state['user_term_to_idx']
        model_lp     = state['user_model_lp']
        data         = state['user_data']
        scaler       = state['user_scaler']
        feature_cols = state['user_feature_cols']
        nodes_df      = state['user_nodes_df']
        adjacency     = state.get('user_adjacency')

        # Build graph (existing or inductive)
        if node_id in term_to_idx:
            x_use = data.x
            edge_index_use = data.edge_index
            node_idx = term_to_idx[node_id]
            exclude_neighbors = get_neighbors(edge_index_use, node_idx)
            conn_raw = payload.get('connections') or payload.get('newConnections') or []
            conn_ids, _ = parse_connections(conn_raw)
            for nid in conn_ids:
                if nid in term_to_idx:
                    exclude_neighbors.add(term_to_idx[nid])
        else:
            conn_raw = payload.get('connections') or payload.get('newConnections') or []
            conn_ids, conn_weights = parse_connections(conn_raw)
            conn_ids, conn_weights = collapse_known_connections(conn_ids, conn_weights, set(term_to_idx.keys()))
            if len(conn_ids) < MIN_INDUCTIVE_CONNECTIONS:
                return jsonify({
                    'error': (
                        f'Node "{node_id}" is unseen. Provide at least '
                        f'{MIN_INDUCTIVE_CONNECTIONS} known connections from the graph.'
                    )
                }), 400
            if scaler is None or feature_cols is None:
                return jsonify({'error': 'Model metadata missing. Retrain the GNN.'}), 400

            new_feat = build_new_node_feature(
                feature_cols, scaler, conn_ids, conn_weights, nodes_df, adjacency
            )
            x_use, edge_index_use, node_idx, neighbor_idxs = build_inductive_graph(
                data, term_to_idx, new_feat, conn_ids
            )
            if node_idx is None:
                return jsonify({'error': 'No valid connections found in graph.'}), 400
            exclude_neighbors = set(neighbor_idxs)

        # Map idx -> node id
        id_list = [None] * len(term_to_idx)
        for nid, idx in term_to_idx.items():
            if idx < len(id_list):
                id_list[idx] = nid

        model_lp.eval()
        with torch.no_grad():
            z = model_lp.encode(x_use, edge_index_use)

        base_n = data.num_nodes
        if node_idx >= base_n:
            scores = (z[node_idx] * z[:base_n]).sum(dim=1)
            candidate_idxs = range(base_n)
        else:
            scores = (z[node_idx] * z[:base_n]).sum(dim=1)
            candidate_idxs = range(base_n)

        probs = torch.sigmoid(scores).cpu().numpy()
        predictions = []
        for i in candidate_idxs:
            if i == node_idx:
                continue
            if i in exclude_neighbors:
                continue
            nid = id_list[i] if i < len(id_list) else str(i)
            predictions.append({
                'id': str(nid),
                'score': float(scores[i]),
                'prob': round(float(probs[i]) * 100, 2),
            })

        predictions.sort(key=lambda x: x['score'], reverse=True)
        k = max(1, min(20, top_k))
        return jsonify({ 'predictions': predictions[:k] })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'Starting GraphML Studio GNN Service on port {port}...')
    app.run(host='0.0.0.0', port=port, debug=False)
