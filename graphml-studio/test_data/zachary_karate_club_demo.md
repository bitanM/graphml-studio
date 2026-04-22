# Zachary Karate Club Demo Script

Dataset file: `graphml-studio/test_data/zachary_karate_club.csv`

Source: NetworkX `karate_club_graph()` reference implementation based on Zachary (1977). The CSV keeps the standard 0-based node IDs from NetworkX.

## Upload Check

1. Upload `zachary_karate_club.csv`.
2. Expected graph summary:
   - Nodes: `34`
   - Edges: `77`
   - Density: `0.1373`
3. Search checks:
   - Search `0` and the graph should focus node `0`
   - Search `33` and the graph should focus node `33`

## Community Detection

### Louvain

1. Select `Community`
2. Keep algorithm on `Louvain`
3. Click `Detect Communities`
4. Expected result:
   - Total communities: `4`
   - Community sizes:
     - `C0 = 11`
     - `C1 = 5`
     - `C2 = 14`
     - `C3 = 4`
5. Visual proof:
   - Click `Color Communities`
   - Node `0` should be in `C0`
   - Node `33` should be in `C2`
   - Node `31` should be in `C3`

### Leiden

1. Switch algorithm to `Leiden`
2. Click `Detect Communities`
3. Expected result:
   - Total communities: `2`
   - Community sizes:
     - `C0 = 19`
     - `C1 = 15`
4. Visual proof:
   - Node `0` should be in `C1`
   - Node `33` should be in `C0`

## Centrality

### Degree Centrality

1. Select `Centrality`
2. Choose `Degree`
3. Click `Show Centrality`
4. Expected top 5:
   - `0 = 0.484848`
   - `33 = 0.484848`
   - `32 = 0.363636`
   - `2 = 0.303030`
   - `1 = 0.272727`

### Betweenness Centrality

1. Choose `Betweenness`
2. Click `Show Centrality`
3. Expected top 5:
   - `0 = 0.467708`
   - `33 = 0.390372`
   - `19 = 0.229640`
   - `31 = 0.134312`
   - `32 = 0.096780`

### Eigenvector Centrality

1. Choose `Eigenvector`
2. Click `Show Centrality`
3. Expected top 5:
   - `2 = 0.387696`
   - `32 = 0.331913`
   - `0 = 0.330270`
   - `33 = 0.328830`
   - `1 = 0.317600`

## GNN Training

1. Click `GNN Online`
2. Wait until the badge changes to `GNN online`
3. Click `Train GNN`
4. Expected proof that training worked:
   - The button changes to `Trained` or `Cached`
   - `Load t-SNE` becomes enabled
   - The node prediction panel shows model metrics instead of placeholder text

## t-SNE

1. Click `Load t-SNE`
2. Switch to the `t-SNE` tab
3. Expected result:
   - The scatter plot loads successfully
   - The legend shows the learned communities
   - The number of plotted points should match the graph node count: `34`

## Node Classification Demo

Use an existing node first so the audience sees a stable prediction.

1. Open `Node Classify`
2. Enter node `0`
3. Click `Predict Community`
4. Expected proof:
   - The app returns a predicted community, confidence, and top-5 probabilities
   - The predicted community should match node `0`'s displayed community in the current graph view

Repeat with:
- Node `33`
- Node `31`

## Link Prediction Demo

Use a new node so the audience sees inductive prediction.

1. Open `Link Predict`
2. Enter node name: `candidate_member`
3. Enter known connections: `0,1,2`
4. Set top-k to `5`
5. Click `Predict Links`
6. Expected proof:
   - The app returns a ranked list of predicted links
   - The predictions must not include `0`, `1`, or `2` because they were already given as known links
   - The list should show probabilities in descending order

## One-Minute Demo Order

1. Upload the CSV and confirm `34` nodes / `77` edges
2. Run Louvain and color communities
3. Run Degree centrality and show node `0`
4. Search for `33`
5. Turn `GNN Online`
6. Train GNN
7. Run Node Classify for `0`
8. Run Link Predict for `candidate_member` with `0,1,2`
9. Load t-SNE
