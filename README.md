# Fraud Detection Dashboard — Deployment

## Project Structure

```
.
├── app/                          # Streamlit dashboard
│   └── dashboard.py
├── src/                          # Core modules & utilities
│   ├── benford_core.py           # Benford's Law math functions
│   ├── benford_by_category.py    # Category analysis
│   └── check_user_patterns.py    # User pattern diagnostics
├── scripts/                      # Data processing & analysis
│   ├── phase1_eda.py             # Initial exploration
│   ├── extract_user_feature.py   # Per-user feature extraction
│   ├── phase3_train_model.py     # Model training
│   └── expected_mad_simulation.py # MAD expectation simulation
├── data/                         # Input data files
├── figures/                      # Generated plots
└── requirements.txt
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## Running Analysis Scripts

From the project root:

```bash
# Phase 1: Initial EDA
python scripts/phase1_eda.py

# Phase 2c: Extract user features
python scripts/extract_user_feature.py

# Phase 2b: Expected MAD simulation
python scripts/expected_mad_simulation.py

# Phase 3: Train models
python scripts/phase3_train_model.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public repo).
2. Sign in at https://share.streamlit.io with your GitHub account.
3. Click "New app" → select your repo, branch (`main`), and file (`app/dashboard.py`).
4. Click "Deploy" — your app will be live in ~1–2 minutes.

The dashboard requires the data files in the `data/` folder:
- `user_features.csv` — per-user Benford features and labels
- `model_results.csv` — model metrics
- `expected_mad_table.csv` — lookup table for expected MAD values

All data files are included in this repo.
