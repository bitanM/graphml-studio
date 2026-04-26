# Cora Evaluation Report

Generated: 2026-04-26T11:59:42

## Setup

- Dataset: `Cora` (Planetoid benchmark)
- Node model: `GraphSAGE_NC`
- Link model: `GraphSAGE_LP`
- Device: `CPU`
- Seed: `42`

## Node Classification

- Best epoch: `170`
- Accuracy: `0.7670`
- Precision (macro): `0.7477`
- Recall (macro): `0.7886`
- F1-score (macro): `0.7589`
- ROC-AUC (macro, OvR): `0.9494`
- ROC-AUC (micro, OvR): `0.9407`

Artifacts:
- `node_classification_confusion_matrix.png`
- `node_classification_roc.png`

## Link Prediction

- Best epoch: `74`
- Accuracy: `0.6720`
- Precision: `0.6246`
- Recall: `0.8626`
- F1-score: `0.7245`
- ROC-AUC: `0.7828`
- Average Precision: `0.7853`

Artifacts:
- `link_prediction_confusion_matrix.png`
- `link_prediction_roc.png`

## Brief Interpretation

- The node classification scores show how well GraphSAGE separates paper topics on the held-out Cora test nodes.
- The link prediction scores show how reliably the GraphSAGE encoder can recover held-out citation edges.
- Use this benchmark as an external evaluation companion to the in-app demo workflow. It gives you standardized metrics that the live app does not currently expose.
