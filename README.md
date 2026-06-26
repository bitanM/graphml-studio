# GraphML Studio

**GraphML Studio** is an open-source, no-code, browser-based platform designed to democratize **Graph Machine Learning (GML)** for researchers across disciplines. It integrates interactive graph visualization with powerful **Graph Neural Network (GNN)** workflows, allowing for both descriptive and predictive network analysis without programming expertise.

## 🚀 Key Features

*   **Interactive Visualization:** Render networks using force-directed layouts via `vis-network` with real-time zooming, panning, and node selection.
*   **Descriptive Analytics:** Built-in support for **Louvain and Leiden** community detection and centrality metrics (Degree, Betweenness, Eigenvector).
*   **Predictive GNN Workflows:** Train **GraphSAGE** models for **Node Classification** and **Link Prediction** directly in the browser.
*   **Inductive Inference:** Classify "unseen" nodes by providing a new node and its connections to the existing graph.
*   **Model Caching:** Employs **SHA-256 graph hashing** to cache trained models on disk, preventing redundant training sessions.
*   **Fault Isolation:** A two-tier microservices architecture ensures core graph analysis remains functional even if the GNN service is unavailable.

## 🏗️ Architecture

GraphML Studio is deployed as two independent cloud microservices:

1.  **Tier 1 (Node.js/Express, Port 3000):** Serves the web frontend, handles CSV parsing, graph construction, community detection, and proxies requests to the GNN service.
2.  **Tier 2 (Python/Flask, Port 5001):** Manages all machine learning operations including GraphSAGE training, t-SNE embedding generation, and inference.

## 📋 Prerequisites

*   **Node.js:** version ≥18
*   **Python:** version ≥3.10
*   **Environment:** Tested on Render.com; deployable on any platform supporting Node.js and Python.

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/bitanM/graphml-studio.git
cd graphml-studio
```

### 2. Tier 1: Web Server (Node.js)
```bash
# Navigate to the frontend/server directory
cd graphml-studio
npm install
node server.js
```

### 3. Tier 2: GNN Microservice (Python)
```bash
# Navigate to the GNN service directory
cd gnn_service 
pip install -r requirements.txt
python app.py
```

## 📖 Usage Workflow

1.  **Upload:** Select a CSV edge list. The system auto-detects delimiters and header rows.
2.  **Explore:** Interact with the force-directed layout and switch between community detection algorithms.
3.  **Train:** Click **"Train GNN"** to trigger the GraphSAGE pipeline. Training on 5,000 nodes typically takes **10–15 minutes** on a CPU.
4.  **Inference:** Query the trained model for existing nodes or use the **Inductive Inference** panel to predict classes for new nodes.

## 🧪 Technical Specifications

| Layer | Technology | Version |
| :--- | :--- | :--- |
| **GNN Framework** | PyTorch Geometric | 2.5.3 |
| **Deep Learning** | PyTorch (CPU-only) | 2.1.0 |
| **Web Server** | Node.js / Express | 5.2.1 |
| **GNN Backend** | Python / Flask | 3.0.3 |

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the `LICENSE.txt` file for the full license text.

## ✍️ Citation

If you use GraphML Studio in your research, please cite our paper:

> Majumder, B., & Sivakumar, T. (2026). GraphML Studio: An Application for Interactive Network Structure Analysis with Graph Neural Networks. *SoftwareX*.

***
