# Graph Based Fraud Detection

This project builds a fraud detection workflow around the IEEE-CIS Fraud Detection dataset. It combines classic tabular anomaly detection, semi-supervised experiments, and knowledge-graph features, then presents the result in a Streamlit dashboard with transaction lookup, graph exploration, community inspection, and live fraud-report risk propagation.

## Current Project Structure

```text
.
+-- Dataset/
|   +-- train_transaction.csv
|   +-- train_identity.csv
|   +-- test_transaction.csv
|   +-- test_identity.csv
|   +-- sample_submission.csv
+-- presentation/
|   +-- fraud.py              # Main Streamlit demo dashboard
|   +-- kg_risk.py            # Shared risk scoring and propagation formulas
|   +-- __pycache__/          # Python cache, can be ignored
+-- fraud.py                  # Earlier/simple Streamlit lookup prototype
+-- KG_Fraud_Detection.ipynb  # KG + tabular model experiments
+-- UnsupervisedKG.ipynb      # Earlier unsupervised KG workflow
+-- UnsupervisedKG 2.ipynb    # Updated KG export workflow for the dashboard
+-- unsupervised.ipynb        # Unsupervised tabular/entity clustering workflow
+-- semi unsupervised.ipynb   # Semi-supervised workflow
+-- requirements.txt
+-- README.md
+-- .gitignore
```

## Installation

Python 3.10 or 3.11 is recommended. The notebooks were previously run with a newer local kernel, but several scientific Python packages are usually smoother on 3.10/3.11.

### Conda

```powershell
cd E:\Graph_Based_Fraud_Detection
conda create -n graph-fraud python=3.11 -y
conda activate graph-fraud
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name graph-fraud --display-name "Python (graph-fraud)"
```

### Python venv on Windows

```powershell
cd E:\Graph_Based_Fraud_Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name graph-fraud --display-name "Python (graph-fraud)"
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
cd /path/to/Graph_Based_Fraud_Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name graph-fraud --display-name "Python (graph-fraud)"
```

## Dataset

The project now contains a local `Dataset/` directory with the expected IEEE-CIS CSV files:

```text
Dataset/train_transaction.csv
Dataset/train_identity.csv
Dataset/test_transaction.csv
Dataset/test_identity.csv
Dataset/sample_submission.csv
```

Most of the graph notebooks read files using paths like:

```python
pd.read_csv("Dataset/train_transaction.csv")
pd.read_csv("Dataset/train_identity.csv")
```

Some older notebook cells may still use root-level paths such as `train_transaction.csv`. If a notebook raises `FileNotFoundError`, update those constants or cells to use `Dataset/...`.

## How the Data Is Processed

The main workflow is:

1. Load transaction and identity CSV files.
2. Merge them by `TransactionID` with the transaction table as the base.
3. Sort by `TransactionDT`.
4. Split the merged training data chronologically into an early training set and a later validation set.
5. Build a knowledge graph where transactions are connected to shared entities such as cards, devices, browsers, operating systems, and email domains.
6. Compute graph features such as degree, weighted degree, PageRank, Louvain community/cluster ID, community size, and community-level anomaly information.
7. Train/evaluate models such as Isolation Forest, LightGBM, Random Forest feature selection, PCA, and HDBSCAN depending on the notebook.
8. Export graph and scored validation artifacts for the dashboard.

## Method Overview

This project is organized around two stages: baseline experiments and a proposed knowledge-graph-based improvement.

### Baseline Methods

The baseline notebooks use mostly tabular transaction/entity features without building a full transaction knowledge graph.

Main methods:

- `Unsupervised baseline`: PCA for dimensionality reduction, HDBSCAN for clustering, and Isolation Forest for anomaly detection.
- `Semi-supervised baseline`: Random Forest uses fraud labels for feature selection, then HDBSCAN clusters transactions and Isolation Forest scores anomalous transactions.

Baseline goal:

```text
Can tabular anomaly detection and clustering discover suspicious transactions or suspicious groups?
```

Relevant notebooks:

```text
unsupervised.ipynb
semi unsupervised.ipynb
```

### Proposed Method: Knowledge Graph Fraud Detection

The proposed method improves on the baseline by modeling relationships between transactions and shared entities.

Instead of treating every transaction as independent, it builds a graph:

```text
TransactionID -> card / device / email domain / browser / operating system / other identity fields
```

Main methods:

- `NetworkX`: builds the transaction-entity knowledge graph.
- Hyper-hub penalized edge weighting: reduces the influence of very common shared entities.
- `PageRank`: measures graph centrality and identifies structurally important nodes.
- Louvain community detection: finds transaction/entity communities that may represent coordinated behavior.
- Graph feature extraction: creates features such as `graph_degree`, `weighted_degree`, `pagerank`, `cluster_id`, `community_size`, and community-level risk signals.
- `IsolationForest`: scores transactions using graph-derived features.
- `LightGBM`: compares supervised tabular-only modeling against tabular + KG feature modeling.
- Risk scoring and propagation: assigns `risk_score` / `risk_class` and updates connected nodes when a suspicious activity report is submitted.

Proposed method goal:

```text
Can graph structure reveal fraud patterns that tabular-only baselines miss?
```

Relevant notebooks and scripts:

```text
KG_Fraud_Detection.ipynb
UnsupervisedKG.ipynb
UnsupervisedKG 2.ipynb
presentation/kg_risk.py
presentation/fraud.py
```

## Notebook Guide

Start Jupyter from the activated environment:

```powershell
jupyter notebook
```

Then select the `Python (graph-fraud)` kernel.

### Running the Baselines

Run these first if you want to reproduce the exploratory baseline stage:

1. Open `unsupervised.ipynb`.
2. Confirm the CSV paths point to the local dataset, usually `Dataset/train_transaction.csv` and `Dataset/train_identity.csv`.
3. Run the cells in order.
4. Review the generated anomaly scores, PCA/HDBSCAN clusters, and cluster-level fraud-rate sanity checks.
5. Open `semi unsupervised.ipynb` and run it the same way.

The baseline notebooks are useful for comparing against the KG approach, but they do not generate the final dashboard artifacts.

### Running the Proposed KG Method

Run these notebooks for the graph-based approach:

- `UnsupervisedKG 2.ipynb`: updated workflow that computes graph-based anomaly evidence, initial `risk_score` / `risk_class`, and exports dashboard artifacts.
- `KG_Fraud_Detection.ipynb`: combined KG + tabular experiments, including Isolation Forest and LightGBM comparisons.
- `UnsupervisedKG.ipynb`: earlier unsupervised KG version.

Recommended order:

1. Run `KG_Fraud_Detection.ipynb` to compare tabular-only models with KG-enhanced models.
2. Run `UnsupervisedKG.ipynb` if you want to inspect the earlier unsupervised graph-only workflow.
3. Run `UnsupervisedKG 2.ipynb` to create the dashboard-ready artifacts:

```text
knowledge_graph.pkl
scored_validation_transactions.csv
```

After `UnsupervisedKG 2.ipynb` finishes, run the presentation dashboard with:

```powershell
streamlit run presentation\fraud.py
```

## Dashboard Artifacts

The main presentation dashboard expects these generated files in the project root when launched from the project root:

```text
knowledge_graph.pkl
scored_validation_transactions.csv
```

`UnsupervisedKG 2.ipynb` contains the export logic for these files. The exported scored table includes fields such as:

```text
TransactionID
TransactionAmt
isFraud
graph_degree
weighted_degree
pagerank
cluster_id
community_size
community_avg_pagerank
anomaly_score
anomaly_evidence
risk_score
risk_class
report_evidence
num_reports
```

If these artifacts are missing, the presentation dashboard falls back to a synthetic demo graph so the UI can still be shown.

## Running the Main Dashboard

Run the updated presentation dashboard from the project root:

```powershell
cd E:\Graph_Based_Fraud_Detection
streamlit run presentation\fraud.py
```

Main features:

- Overview of transaction counts, entity counts, and high-risk/suspicious classes.
- Transaction lookup by `TransactionID`.
- Explainability panel showing anomaly evidence, linked reports, and graph cluster information.
- Interactive graph explorer using PyVis.
- Community explorer for suspicious graph communities.
- Suspicious activity reporting form.
- Live update log for new nodes, report links, risk updates, class changes, and propagated risk changes.

## Running the Earlier Prototype

The root-level `fraud.py` is an earlier Streamlit prototype. It only loads `scored_validation_transactions.csv` and provides a simpler transaction lookup/reporting UI.

```powershell
streamlit run fraud.py
```

If `scored_validation_transactions.csv` is missing, it uses a small built-in demo dataframe.

## Main Dependencies

- `pandas`, `numpy` for data processing
- `scikit-learn` for Isolation Forest, Random Forest, PCA, scaling, and metrics
- `networkx` for knowledge graph construction and PageRank
- `python-louvain` for Louvain community detection
- `lightgbm` for supervised tabular/KG classification experiments
- `hdbscan` for clustering experiments
- `matplotlib` for notebook visualizations
- `streamlit` for dashboards
- `pyvis` for interactive graph visualization in the presentation dashboard
- `jupyter`, `ipykernel` for notebook execution

## Notes

- The full dataset is large, so graph construction and PageRank can take time and memory.
- For quicker iteration, reduce sample sizes in the notebooks before running full experiments.
- Generated artifacts such as `knowledge_graph.pkl` and `scored_validation_transactions.csv` can be large; commit them only if you intentionally want to share generated outputs.
