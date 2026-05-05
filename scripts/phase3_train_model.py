"""
Phase 3: Model training and comparison
========================================
Train three Random Forest models to test whether our per-user multi-dim
Benford features (including the inverted signal we discovered) actually
improve fraud detection beyond baseline features.

Models compared:
  1. Baseline    - traditional per-user features only
  2. ODU-style   - baseline + single aggregate Benford feature
  3. Ours        - baseline + benford_amt, benford_time, benford_ratio

Outputs:
  - figuress/roc_comparison.png    - ROC curves for all three models
  - figuress/feature_importance.png - which Benford dim matters most
  - figuress/confusion_matrices.png - 2x3 grid, per model
  - data/model_results.csv        - metrics table
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benford_core import mad


DATA_DIR = Path(__file__).parent.parent / "data"
figures_DIR = Path(__file__).parent.parent / "figures"

FEATURES_FILE = DATA_DIR / "user_features.csv"
RANDOM_STATE = 42



# Feature sets

BASELINE_FEATURES = [
    "n_txn", "mean_amt", "std_amt", "median_amt", "max_amt",
    "median_time_delta",
]

ODU_FEATURES = BASELINE_FEATURES + ["benford_aggregate"]

OURS_FEATURES = BASELINE_FEATURES + ["benford_amt", "benford_time", "benford_ratio"]


def compute_odu_benford(feat_df, raw_df):
    """
    Compute an ODU-style single aggregate Benford feature: per user, the
    raw MAD on transaction amounts (no small-sample correction, single
    dimension). This is what the ODU paper would have computed.
    """
    print("Computing ODU-style aggregate Benford feature...")
    user_ids = feat_df["user_id"].values
    result = []
    for uid in user_ids:
        amts = raw_df[raw_df["user_id"] == uid]["TransactionAmt"].values
        raw_mad = mad(amts) if len(amts) >= 9 else np.nan
        result.append(raw_mad)
    return np.array(result)


def load_features():
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found.")
        print("Run extract_user_features.py first.")
        sys.exit(1)
    return pd.read_csv(FEATURES_FILE)


def load_raw_transactions_for_odu():
    """We need the raw data to compute ODU-style aggregate MAD per user."""
    print("Loading raw transactions for ODU baseline...")
    path = DATA_DIR / "train_transaction.csv"
    if not path.exists():
        path = DATA_DIR / "sync_train_transaction.csv"
    cols_needed = [
        "TransactionAmt", "card1", "addr1", "P_emaildomain",
    ]
    df = pd.read_csv(path, usecols=cols_needed)
    df["user_id"] = (
        df["card1"].fillna(-1).astype(int).astype(str) + "_"
        + df["addr1"].fillna(-1).astype(int).astype(str) + "_"
        + df["P_emaildomain"].fillna("none").astype(str)
    )
    return df


def train_and_eval(X_train, X_test, y_train, y_test, name):
    """Train one Random Forest and return metrics + predictions."""
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",     # handle class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "auc_roc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return clf, metrics, y_pred, y_prob


def main():
    feat_df = load_features()
    print(f"Loaded features: {feat_df.shape[0]:,} users, "
          f"{feat_df['isFraud'].sum():,} fraud-positive")

    # Compute the ODU-style aggregate Benford feature
    raw_df = load_raw_transactions_for_odu()
    feat_df["benford_aggregate"] = compute_odu_benford(feat_df, raw_df)

    # Drop rows with NaN in any feature column
    all_features = list(set(BASELINE_FEATURES + ODU_FEATURES + OURS_FEATURES))
    feat_df_clean = feat_df.dropna(subset=all_features + ["isFraud"]).copy()
    print(f"After NaN removal: {len(feat_df_clean):,} users")
    print(f"  fraud: {feat_df_clean['isFraud'].sum()}")
    print(f"  legit: {(1-feat_df_clean['isFraud']).sum()}")

    
    # Train/test split at the USER level (no leakage)
    
    y = feat_df_clean["isFraud"].astype(int).values

    # Use the full feature matrix once for splitting; subset per model
    X_all = feat_df_clean[all_features].values

    idx_train, idx_test = train_test_split(
        np.arange(len(feat_df_clean)),
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    y_train, y_test = y[idx_train], y[idx_test]
    print(f"\nTrain: {len(idx_train):,} users ({y_train.sum()} fraud)")
    print(f"Test:  {len(idx_test):,} users ({y_test.sum()} fraud)")


    # Train three models
    results = []
    probs_by_model = {}
    preds_by_model = {}
    importance_by_model = {}

    for name, feat_list in [
        ("Baseline", BASELINE_FEATURES),
        ("ODU-style", ODU_FEATURES),
        ("Ours (3D Benford)", OURS_FEATURES),
    ]:
        print(f"\n=== Training: {name} ===")
        print(f"  features: {feat_list}")
        X = feat_df_clean[feat_list].values
        X_train, X_test = X[idx_train], X[idx_test]

        clf, metrics, y_pred, y_prob = train_and_eval(
            X_train, X_test, y_train, y_test, name
        )
        results.append(metrics)
        probs_by_model[name] = y_prob
        preds_by_model[name] = y_pred
        importance_by_model[name] = pd.Series(
            clf.feature_importances_, index=feat_list
        ).sort_values(ascending=False)

        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / "model_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics: {out_path.resolve()}")
    
    print("RESULTS SUMMARY")

    print(results_df.to_string(index=False))


    # figures 1: ROC curves overlaid
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Baseline": "gray", "ODU-style": "steelblue",
              "Ours (3D Benford)": "coral"}
    for r in results:
        name = r["model"]
        fpr, tpr, _ = roc_curve(y_test, probs_by_model[name])
        ax.plot(fpr, tpr, color=colors[name], linewidth=2,
                label=f"{name} (AUC={r['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: Baseline vs. ODU-style vs. Ours")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.subplots_adjust(right=0.92, wspace=0.3)
    fig_path = figuresS_DIR / "roc_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figures: {fig_path.resolve()}")

    
    # figures 2: Feature importance for our model
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ours_imp = importance_by_model["Ours (3D Benford)"]
    colors_bar = ["coral" if "benford" in f else "steelblue"
                  for f in ours_imp.index]
    ax.barh(range(len(ours_imp)), ours_imp.values, color=colors_bar)
    ax.set_yticks(range(len(ours_imp)))
    ax.set_yticklabels(ours_imp.index)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title("Feature importance (Our model)\n"
                 "Coral = Benford features, Blue = baseline")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = figuresS_DIR / "feature_importance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved figures: {fig_path.resolve()}")

    
    # figures 3: Confusion matrices side-by-side
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, ["Baseline", "ODU-style", "Ours (3D Benford)"]):
        cm = confusion_matrix(y_test, preds_by_model[name])
        # Row-normalize: each row sums to 1 (percentages of actual class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xticks([0, 1]);
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # Annotate each cell with both count and percentage
        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j] * 100
                count = cm[i, j]
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{pct:.1f}%\n(n={count})",
                        ha="center", va="center",
                        color=text_color, fontsize=12, fontweight="bold")

    # Add a colorbar to the rightmost subplot
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="Proportion of actual class")
    plt.tight_layout()
    fig_path = figuresS_DIR / "confusion_matrices.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

    # Interpretation
    baseline_auc = results_df[results_df["model"] == "Baseline"]["auc_roc"].iloc[0]
    odu_auc = results_df[results_df["model"] == "ODU-style"]["auc_roc"].iloc[0]
    ours_auc = results_df[results_df["model"] == "Ours (3D Benford)"]["auc_roc"].iloc[0]

    



if __name__ == "__main__":
    main()