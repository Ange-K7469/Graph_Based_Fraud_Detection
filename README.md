# Graph Based Fraud Detection

This project explores fraud detection on transaction data using tabular machine learning and knowledge-graph-derived features. It includes Jupyter notebooks for experimentation and a Streamlit app for looking up scored transactions and viewing graph-based explanations.

## Project Structure

```text
.
├── fraud.py                         # Streamlit app for transaction lookup and reporting
├── KG_Fraud_Detection.ipynb         # Main KG + tabular fraud detection workflow
├── UnsupervisedKG.ipynb             # Unsupervised knowledge graph workflow
├── unsupervised.ipynb               # Unsupervised tabular/entity workflow
├── semi unsupervised.ipynb          # Semi-supervised workflow
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Ignores raw dataset CSV files
```

## Installation

Use Python 3.10 or 3.11 for the smoothest package compatibility.

### Conda

```powershell
cd E:\Graph_Based_Fraud_Detection
conda create -n graph-fraud python=3.11 -y
conda activate graph-fraud
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name graph-fraud --display-name "Python (graph-fraud)"
```

### Windows PowerShell

```powershell
cd E:\Graph_Based_Fraud_Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name graph-fraud --display-name "Python (graph-fraud)"
```

If PowerShell blocks virtual environment activation, run:

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

## Dataset Setup

The notebooks expect the IEEE-CIS Fraud Detection CSV files. Because these files are large, they are not stored in this repository.

Place the dataset files in one of the paths used by the notebooks:

```text
Dataset/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
└── sample_submission.csv
```

Some notebooks use root-level paths such as `train_transaction.csv` and `train_identity.csv`. If a notebook raises `FileNotFoundError`, either move/copy the CSV files into the project root or update the notebook constants to point to `Dataset/...`.

The raw dataset files are ignored by `.gitignore`:

```text
sample_submission.csv
test_identity.csv
test_transaction.csv
train_identity.csv
train_transaction.csv
```

## Running the Notebooks

Start Jupyter from the activated environment:

```powershell
jupyter notebook
```

Then open one of the notebooks and select the `Python (graph-fraud)` kernel.

Recommended order:

1. `KG_Fraud_Detection.ipynb` - builds graph features, trains Isolation Forest and LightGBM models, and evaluates ROC-AUC.
2. `UnsupervisedKG.ipynb` - focuses on unsupervised graph topology features.
3. `unsupervised.ipynb` - tabular/entity unsupervised baseline.
4. `semi unsupervised.ipynb` - semi-supervised experiments.

## Running the Streamlit App

The Streamlit app is in `fraud.py`.

```powershell
streamlit run fraud.py
```

The app tries to load:

```text
scored_validation_transactions.csv
```

If that file does not exist, the app falls back to a small built-in demo dataframe.

For full project results, generate `scored_validation_transactions.csv` from the validation dataframe created in `KG_Fraud_Detection.ipynb`. After the cell that creates `df_val_full['supervised_kg_tabular_risk']`, add and run:

```python
export_cols = [
    "TransactionID",
    "TransactionAmt",
    "card1",
    "isFraud",
    "supervised_kg_tabular_risk",
    "weighted_degree",
    "pagerank",
    "community_fraud_rate",
]

df_val_full[export_cols].to_csv("scored_validation_transactions.csv", index=False)
```

Then restart Streamlit:

```powershell
streamlit run fraud.py
```

## Main Dependencies

- `pandas`, `numpy` for data processing
- `scikit-learn` for Isolation Forest, Random Forest, PCA, scaling, and metrics
- `networkx` for graph construction and PageRank
- `python-louvain` for Louvain community detection
- `lightgbm` for supervised fraud classification
- `hdbscan` for density-based clustering experiments
- `matplotlib` for notebook visualizations
- `streamlit` for the transaction lookup interface
- `jupyter`, `ipykernel` for running notebooks

## Notes

- The graph-based notebooks can take time and memory because they build transaction/entity graphs over many rows.
- If memory is limited, reduce sample sizes in the notebooks before running full experiments.
- Keep large raw data files and generated scored CSVs out of commits unless they are intentionally needed for sharing.