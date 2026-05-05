# Benford's Law for Credit Card Fraud Detection

Pipeline for testing whether per-user, multi-dimensional Benford's Law conformance
can detect credit card fraud on the IEEE-CIS Fraud Detection dataset.

Website: https://benford-pca-fraud-detection.streamlit.app/

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

## Project Instruction
Download `train_transaction.csv` from the
[IEEE-CIS Fraud Detection Kaggle competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
and place it in `data/`.

## Pipeline (run in order)

### Step 1 — `phase1_eda.py`

**Purpose:** Confirm the dataset is feasible for per-user Benford analysis
before building the rest of the pipeline.

**Checks:**
- Total transactions and fraud rate
- Whether transaction amounts span ≥ 3 orders of magnitude (Benford requirement)
- How many users have ≥ 30 transactions (per-user analysis minimum)
- How many of those frequent users have any fraudulent transactions

**Outputs:**
- Console summary with feasibility verdict
- `figures/phase1_eda.png` — distribution of amounts and per-user transaction counts

### Step 2 — `expected_mad_simulation.py`

**Purpose:** Empirically estimate E[MAD] under the Benford null for various
sample sizes, used for the small-sample correction in Step 3.

**Method:** Monte Carlo simulation with 5,000 trials per sample size across
N ∈ {10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000}.

**Outputs:**
- `data/expected_mad_table.csv` — lookup table of E[MAD] vs N
- `figures/expected_mad_curve.png` — log-log plot showing E[MAD] ≈ k/√N with k = 0.235

### Step 3 — `extract_user_feature.py`

**Purpose:** For each user with ≥ 30 transactions, compute the three-dimensional
Benford fingerprint plus baseline transaction features.

**Method:**
1. Infer user identity from `card1 + addr1 + P_emaildomain`
2. For each frequent user, compute Excess MAD on:
   - Transaction amounts (`benford_amt`)
   - Inter-transaction time deltas (`benford_time`)
   - Amount-to-mean ratios (`benford_ratio`)
3. Use the lookup table from Step 2 to apply the small-sample correction:
   `Excess MAD = MAD - E[MAD(N)]`

**Outputs:**
- `data/user_features.csv` — one row per user with baseline features + Benford
  features + fraud label
- Console summary table comparing Benford features by class

### Step 4a — `benford_by_category.py` (diagnostic)

**Purpose:** Test whether the unexpected per-user inversion holds across
IEEE-CIS product categories or is an artifact of one category dominating
the dataset.

**Method:** For each `ProductCD` value (W, C, R, H, S), compute aggregate
MAD on amounts separately for legitimate and fraudulent transactions.

**Outputs:**
- `figures/benford_by_category.png` — side-by-side digit distributions per category
- Console table comparing legit vs fraud MAD across all five categories

**Finding:** Legit deviates from Benford more than fraud in every category,
and legit MAD varies 4× across categories (0.018 to 0.071) — evidence that
Hill's mixture theorem fails at the category level.

### Step 4b — `check_user_patterns.py` (diagnostic)

**Purpose:** Investigate the underlying transaction patterns that produce
the inverted Benford signal, by inspecting individual users.

**Method:** Sample three legitimate and three fraudulent users at varying
transaction volumes. For each, print:
- Transaction count, amount range, log10 span
- Top-5 most-repeated amounts (subscription detection)
- Leading-digit distribution and per-user MAD

Plus an aggregate metric: fraction of transactions with repeated amounts
across all frequent users in each class.

**Outputs:**
- Console diagnostic per user
- Console aggregate repeat-rate comparison

**Finding:** Legitimate frequent users repeat exact amounts in 64.9% of
transactions (subscription contamination, e.g. `$59.00` appearing 27% of
the time). Fraudulent users have wider amount distributions consistent
with automated card-testing scripts.

### Step 5 — `phase3_train_model.py`

**Purpose:** Test whether the per-user Benford features improve fraud detection
end-to-end, comparing three Random Forest models.

**Models:**
1. **Baseline** — n_txn, mean_amt, std_amt, median_amt, max_amt, median_time_delta
2. **ODU-style** — baseline + single aggregate raw MAD on amounts
3. **Ours (3D Benford)** — baseline + benford_amt, benford_time, benford_ratio

**Evaluation:** Stratified 75/25 user-level train/test split, Random Forest with
200 trees, max depth 10, balanced class weights, seed 42.

**Outputs:**
- `data/model_results.csv` — AUC-ROC, precision, recall, F1 per model
- `figures/roc_comparison.png` — ROC curves overlaid
- `figures/feature_importance.png` — feature importance for the 3D Benford model
- `figures/confusion_matrices.png` — row-normalized confusion matrices

## Imported (not run directly)

### `benford_core.py`

Math primitives used by all other scripts:
- `benford_probs()` → length-9 array of P(d) = log10(1 + 1/d)
- `leading_digit(x)` → first significant digit using log10 mantissa
- `digit_distribution(values)` → (counts, proportions) of digits 1..9
- `mad(values)` → Mean Absolute Deviation from Benford
- `chi_squared(values)` → Pearson chi-squared statistic
- `z_statistics(values)` → per-digit Z-statistic with continuity correction

## Key Findings

1. Per-user Benford features **do not improve** fraud detection on IEEE-CIS
   (baseline AUC 0.749 vs 0.739 with Benford features).
2. The expected signal **inverts**: legitimate users deviate from Benford more
   than fraudulent users in every product category.
3. Two mechanisms explain the inversion: subscription contamination on the
   legitimate side, and card-testing bot patterns on the fraud side.
