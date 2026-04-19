# Thesis Codebase - Comprehensive Exploration Report

**Date**: April 2026  
**Project**: Farmer's Protest Social Media Analysis + GraphML Studio  

---

## 📋 Table of Contents
1. [Project Architecture Overview](#project-architecture-overview)
2. [Major Components & Directories](#major-components--directories)
3. [Farmer's Protest Analysis](#farmers-protest-analysis)
4. [GraphML Studio Application](#graphml-studio-application)
5. [Data Pipelines](#data-pipelines)
6. [Machine Learning Models](#machine-learning-models)
7. [Dependencies & Configuration](#dependencies--configuration)
8. [API Endpoints](#api-endpoints)

---

## 🏗️ Project Architecture Overview

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                  GraphML Studio (Web UI)                │
│  HTML + CSS + Vis.js (3D Network Visualization)         │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
    Node.js Express      Python Flask
    Server (3000)        Server (5001)
    
    - CSV Upload       - GNN Training
    - Graph Analysis   - Node Classification
    - Community Detect - Link Prediction
    - Centrality       - t-SNE Embeddings
    - Leiden/Louvain   
```

### Deployment
**Render.yaml** defines two cloud services:
- **graphml-studio-web**: Node.js server (port 3000)
- **graphml-studio-gnn**: Python Flask (port 5001)

---

## 📁 Major Components & Directories

### Root Level
```
Thesis/
├── Farmer's Protest/          # Social media analysis project
├── graphml-studio/            # Web application + GNN service
├── render.yaml                # Deployment configuration
├── GraphML-Studio.pdf         # Documentation
├── GraphML-Studio.pptx        # Presentation
└── [Project documentation]
```

### Farmer's Protest/ (4 subdirectories)

#### 1. **Farmer's Protest Data/**
- **data.ipynb**: Data loading & basic exploration
  - Loads tweets.csv, users.csv
  - Text cleaning (URL removal, stopword filtering, lemmatization)
  - TF-IDF analysis to identify top 200 words
  - Co-occurrence network construction

#### 2. **word co-occurrence/**
- **kaggle_tfidf_pmi_pipeline.ipynb** & **.py**
  - Two-pass analysis of tweet corpus
  - Pass 1: TF-IDF vocabulary selection (MIN_TERM_FREQ=50, MIN_DOC_FREQ=5)
  - Pass 2: Sliding window co-occurrence (WINDOW=5)
  - PMI (Pointwise Mutual Information) weighting
  - Disparity filter to prune edges (TARGET_DENSITY < 0.05)
  
- **InspectNodes.py**: Examines node properties
- **InspectEdgesPMIPruned.py**: Analyzes pruned edges
- **Output files**:
  - `edges_pmi.csv`, `edges_pmi_pruned.csv`
  - `nodes.csv` (vocabulary with TF, DF, TF-IDF, PMI_MAX)
  - `network_pmi_pruned.gexf`, `stats.json`

#### 3. **retweet network/**
- **kaggle_retweet_graph.ipynb**
  - Extracts directed retweet interactions from metadata
  - Parses `retweetedTweet` field to find original users
  - Builds user → user directed graph
  - Edges weighted by retweet frequency
  
- **Outputs**:
  - `retweet_edges.csv` (source, target, weight)
  - `retweet_nodes.csv` (in_weight, out_weight, in_degree, out_degree)
  - `retweet_graph.gexf`
  - `stats.json` (node/edge counts)

#### 4. **user similarity/**
- **kaggle_user_similarity.ipynb**
  - Similarity components:
    - Log-normalized follower count
    - Log-normalized following count
    - Account age (days since creation)
    - Verified status (binary)
    - Bio TF-IDF cosine similarity (ALPHA_BIO=0.6)
  - KNN over bio similarity → top-K neighbors (K=12)
  - Outputs:
    - `user_similarity_edges.csv` (weight, bio_sim, num_sim)
    - `user_similarity_nodes.csv` (user attributes)
    - `user_similarity_graph.gexf`

---

## 🔬 Farmer's Protest Analysis

### Datasets
1. **tweets.csv**: Tweet corpus
   - Fields: renderedContent, retweetedTweet, userId, etc.
   - Used for: text analysis, retweet graph, word co-occurrence

2. **users.csv**: User profiles
   - Fields: userId, username, displayname, verified, followersCount, friendsCount, rawDescription, created
   - Used for: user similarity analysis

### Analysis Focus Areas
1. **Vocabulary Networks**: Word co-occurrence patterns in protest discourse
2. **Retweet Dynamics**: Information diffusion & user influence
3. **User Profiles**: Community structure based on user attributes & bio similarity
4. **Community Discovery**: 15 distinct themes/clusters identified:
   - Counter-Narrative / Fringe
   - International Amplification
   - Systemic Critique (Adani/Ambani)
   - Celebrity Discourse (MIA/Rihanna)
   - Farm Law & MSP Policy
   - Core Protest Vocabulary
   - Agricultural Heritage / Ethos
   - Media Criticism & Khalistan Framing
   - Cross-Movement Solidarity (CAA/NRC)
   - Ground-Level Geography (Delhi/Punjab)
   - Tech & Climate Crossover
   - Political Opposition (BJP/RSS)
   - Viral Mechanics (Hashtag/Retweet)
   - Punjabi / Sikh Identity
   - Democratic Grievances & Slogans

---

## 🌐 GraphML Studio Application

### Frontend (`public/index.html`)
- **Tech Stack**: HTML5, CSS3, Vis.js
- **Features**:
  - 3D network visualization (Vis.js network layout)
  - Dark theme UI with gradient accents
  - Real-time graph rendering
  - Interactive node/edge exploration
  
- **UI Components**:
  - Topbar: Logo, GNN status badge, metadata
  - Menubar: Controls for analysis/export
  - Network viewport: 3D graph display
  - Side panels: Node info, community details

### Node.js Backend (`server.js` - ~700 lines)

#### Core Technologies
- **Express.js**: REST API server
- **Graphology**: Graph data structure & algorithms
- **Multer**: CSV file upload handling
- **graphology-communities-louvain**: Community detection
- **networkanalysis-ts**: Leiden algorithm wrapper

#### Key Functions

**`buildSampleGraph()`**
- Downsamples large graphs to UI limits
- Keeps top nodes by degree
- Filters edges by frequency

**`computeLeidenCommunities()`**
- Converts nodes/edges to Leiden algorithm format
- Runs community detection (resolution=0.2)
- Tests 10+ random starts for stability

**`detectSeparator()`**
- Auto-detects CSV delimiter (,;|\t)

#### Main REST Endpoints

**POST /api/analyze**
- Uploads CSV file (edges: source, target, weight)
- Computes:
  - Community detection (Louvain + Leiden)
  - Centrality measures:
    - Degree centrality
    - Betweenness centrality (skipped if >1500 nodes)
    - Eigenvector centrality
  - Graph statistics
- Returns: nodes[], edges[], meta{}
- Limits: UI 5K nodes/15K edges; centrality 1.5K/10K

**GNN Proxy Endpoints** (forward to Flask)
- `/api/gnn/status` - Check GNN service availability
- `/api/gnn/wake` - Warm up GNN service
- `/api/gnn/demo/*` - Demo mode endpoints
- `/api/gnn/train` - Train on uploaded graph
- `/api/gnn/user/*` - User upload endpoints

---

## 🤖 Python Flask GNN Service (`gnn_services/app.py` - ~1200 lines)

### Startup
- Port: 5001 (CPU-only deployment)
- CORS enabled
- Pre-loads demo data on startup

### State Management
```python
state = {
    'demo_loaded': bool,
    'demo_data': PyG Data,
    'demo_model_nc': GraphSAGE model,
    'demo_model_lp': GraphSAGE model,
    'demo_embeddings': np.array (N, 2) t-SNE,
    'user_data': PyG Data,
    'user_graph_hash': str,  # For caching
    'user_cached': bool,
}
```

### Core Endpoints

#### Demo Mode (Pre-trained Models)
**POST /gnn/demo/load**
- Loads pre-trained models from `demo_data/`
- Models: `best_sage_nc_v2.pt`, `best_sage_lp.pt`
- Returns: graph stats, community info

**GET /gnn/demo/embeddings**
- Returns t-SNE reduced embeddings (2D coordinates)
- Used for network visualization in UI

**GET /gnn/demo/communities**
- Returns community theme information
- Includes community IDs, labels, and descriptions

**POST /gnn/demo/search**
- Search for nodes by keyword/term
- Returns matching nodes with scores

**POST /gnn/demo/predict-node**
- Predicts community for new term
- Uses node classification model

#### User Upload Mode
**POST /gnn/train**
- Trains GNN on user-uploaded CSV
- Builds PyG Data from nodes/edges
- Trains GraphSAGE NC + LP models
- Caches models with graph hash for re-use

**GET /gnn/user/embeddings**
- Returns t-SNE embeddings for user graph

**POST /gnn/user/predict-node**
- Predicts node properties on user graph

### Helper Functions

**`graph_hash(nodes, edges)`**
- SHA256 hash of graph structure
- Deterministic for caching

**`build_pyg_data(nodes_df, edges_df)`**
- Converts Pandas DataFrames → PyTorch Geometric Data object
- Normalizes features with StandardScaler
- Returns: data, term_to_idx, num_classes, scaler

**`parse_connections()`**
- Flexible connection parsing (IDs or dicts with weights)

**`build_inductive_graph()`**
- Adds new node to existing graph
- Inductive inference setup

---

## 🧠 Machine Learning Models

### GraphSAGE_NC (Node Classification)
```python
class GraphSAGE_NC(torch.nn.Module):
    - Conv Layer 1: in_channels → hidden_channels (128)
    - BatchNorm1d
    - Conv Layer 2: hidden_channels → hidden_channels
    - BatchNorm1d
    - Conv Layer 3: hidden_channels → out_channels (num_classes)
    - Aggregation: Mean
    - Dropout: 0.3
```

**Training** (train.py)
- Train/val/test split: 5% / 15% / 80%
- Optimizer: Adam (lr=0.005, weight_decay=5e-4)
- Loss: Cross-entropy with class weights
- Early stopping: patience=15
- Metrics: Accuracy, Macro F1

### GraphSAGE_LP (Link Prediction)
```python
class GraphSAGE_LP(torch.nn.Module):
    - Encoder: 2-layer SAGEConv
    - Decoder: Dot product
    - Predicts edge probability
```

**Training**
- RandomLinkSplit for edge/non-edge sampling
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=15
- Metric: AUC-ROC

### Embedding Generation
- t-SNE reduction: n_components=2, perplexity=30
- Used for 2D visualization in UI

### Pre-trained Weights
- **Demo Mode**: Pre-trained on Farmer's Protest word co-occurrence network
- Located: `gnn_services/demo_data/best_sage_nc_v2.pt`, `best_sage_lp.pt`
- Prepared by: `prepare_demo_data.py` script

---

## 📦 Dependencies & Configuration

### Frontend Dependencies (`package.json`)
```json
{
  "@tensorflow/tfjs": "^4.22.0",
  "csv-parser": "^3.2.0",
  "express": "^5.2.1",
  "graphology": "^0.26.0",
  "graphology-communities-louvain": "^2.0.2",
  "graphology-metrics": "^2.4.0",
  "multer": "^2.1.1",
  "networkanalysis-ts": "^1.0.0"
}

DevDependencies:
  - @babel/core, babel-jest
  - jest, supertest (testing)
```

### Python Backend (`requirements.txt`)
```
pandas, numpy
torch==2.1.0
torch-geometric==2.5.3
transformers
scikit-learn
tqdm
matplotlib, openpyxl
python-Levenshtein
deep-translator
ollama (LLM integration)
flask, flask-cors
```

### GNN Service (`gnn_services/requirements.txt`)
```
flask==3.0.3
flask-cors==4.0.1
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
torch==2.1.0
torch-geometric==2.5.3
```

### Environment Variables
- **GNN_URL**: Python service URL (e.g., https://graphml-studio-gnn.onrender.com)
- **GNN_HOST**: Localhost (default: localhost)
- **GNN_PORT**: 5001
- **GNN_TIMEOUT_MS**: Request timeout (default: 15000ms)
- **PYTHON_VERSION**: 3.11.11

### Render.yaml Deployment
```yaml
services:
  - graphml-studio-gnn (Python):
      runtime: python 3.11.11
      rootDir: graphml-studio
      buildCommand: |
        pip install torch==2.1.0 (CPU)
        pip install torch_geometric packages
        pip install -r gnn_services/requirements.txt
      startCommand: python gnn_services/app.py

  - graphml-studio-web (Node.js):
      runtime: node
      rootDir: graphml-studio
      buildCommand: npm install
      startCommand: node server.js
      env: GNN_URL=https://graphml-studio-gnn.onrender.com
```

---

## 🔗 API Endpoints

### Graph Analysis (Node.js)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyze` | POST | Upload CSV, analyze graph, detect communities |
| `/api/gnn/status` | GET | Check if Python GNN service is online |
| `/api/gnn/wake` | POST/GET | Warm up GNN service (for Render free tier) |

### Demo Mode (Proxied to Flask)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/gnn/demo/load` | POST | Load pre-trained Farmer's Protest models |
| `/api/gnn/demo/embeddings` | GET | Get t-SNE 2D embeddings |
| `/api/gnn/demo/communities` | GET | Get community theme information |
| `/api/gnn/demo/search` | POST | Search nodes by keyword |
| `/api/gnn/demo/predict-node` | POST | Classify new node |

### User Upload Mode (Proxied to Flask)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/gnn/train` | POST | Train GNN on uploaded graph |
| `/api/gnn/user/embeddings` | GET | Get embeddings for user graph |
| `/api/gnn/user/predict-node` | POST | Predict on user graph |

---

## 📊 Data Inputs & Outputs

### CSV Input Format
**Edges CSV** (for `/api/analyze`):
```
source, target, weight
user1, user2, 3.5
farmer, protest, 10.2
...
```

### JSON Output Format (Analyze Response)
```json
{
  "nodes": [
    {
      "id": "farmers",
      "degree": 45,
      "community": 0,
      "communities": { "louvain": 0, "leiden": 0 },
      "centrality": {
        "degree": 0.15,
        "betweenness": 0.08,
        "eigenvector": 0.12
      }
    }
  ],
  "edges": [
    { "from": "farmers", "to": "protest", "weight": 15.3 }
  ],
  "meta": {
    "sampled": false,
    "total_nodes": 1234,
    "total_edges": 5678,
    "centralitySkipped": false
  }
}
```

---

## 🧪 Testing

### JavaScript Tests (`tests/server.test.js`)
- **Framework**: Jest + Supertest
- **Coverage**: CSV separator detection, routing
- Run: `npx jest tests/server.test.js --verbose`

### Python Tests (`tests/test_gnn_services.py`)
- **Framework**: pytest
- **Fixtures**: Flask test client, sample data
- Run: `pytest tests/test_gnn_services.py -v`

---

## 📈 Overall Data Flow

```
Farmer's Protest Tweets (tweets.csv, users.csv)
    ↓
┌───────────────────────────────────────────────┐
│  Three Parallel Pipelines                     │
└───────────────────────────────────────────────┘
    ↓           ↓                ↓
  Word Co-   Retweet User       Similarity
  occurrence Network            Network
    ↓           ↓                ↓
  PMI-weighted Directed edge User KNN
  undirected   graph            graph
    ↓           ↓                ↓
┌───────────────────────────────────────────────┐
│  GraphML Studio Web UI                        │
├───────────────────────────────────────────────┤
│  Node.js Server (Graph Analysis)              │
│  - CSV Upload                                 │
│  - Community Detection (Leiden/Louvain)       │
│  - Centrality Computation                     │
├───────────────────────────────────────────────┤
│  Flask GNN Service (ML Models)                │
│  - Pre-trained Models (Demo Mode)             │
│  - Training Pipeline (User Uploads)           │
│  - Node Classification & Link Prediction      │
└───────────────────────────────────────────────┘
    ↓
  Browser: 3D Network Visualization (Vis.js)
```

---

## 🎯 Key Features

### GraphML Studio Capabilities
1. **Upload**: Any CSV with edges (source, target, weight)
2. **Analysis**: 
   - Community detection (Louvain + Leiden)
   - Centrality measures
   - Interactive filtering
3. **Visualization**: 3D network with node coloring by community
4. **GNN Models**:
   - Demo: Pre-trained on Farmer's Protest
   - User: Train on custom uploads
5. **Embeddings**: t-SNE 2D visualization
6. **Export**: Graph data (GEXF format)

### Farmer's Protest Analysis Insights
- **15 distinct discourse communities** identified
- **Multi-perspective**: vocabulary, user interactions, user profiles
- **PMI-weighted networks**: statistical association measure
- **Temporal dynamics**: through retweet patterns
- **User influence**: betweenness/eigenvector centrality

---

## 🚀 Deployment & Scaling

### Current Setup (Render)
- Frontend + Graph analysis on **graphml-studio-web** (Node.js)
- ML service on **graphml-studio-gnn** (Python)
- Free tier: Services spin down after 15 min inactivity
- "Wake" endpoint to activate service

### Limitations Handled
- **Graph size**: Samples to 5K nodes / 15K edges for UI
- **Centrality computation**: Skipped for graphs >1.5K nodes
- **GNN training**: CPU-only (set device to 'cpu')
- **Memory**: Demo preloads models; user uploads cached by graph hash

---

## 📝 Summary

This thesis project combines **social media analysis** of the Farmer's Protest with a **general-purpose graph visualization and GNN system**. The Farmer's Protest analysis extracts three complementary network views (word co-occurrence, retweets, user similarity), while GraphML Studio provides the interactive interface and ML capabilities to analyze any graph dataset through community detection, centrality metrics, and GraphSAGE models.

The architecture cleanly separates concerns: Node.js handles graph algorithms and visualization, Flask handles ML operations, and the frontend provides the user experience. This allows independent scaling and deployment of components.
