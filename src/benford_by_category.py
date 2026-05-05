"""
Diagnostic 2: Benford conformance by product category ["A","B","C","D","E","F", ...]
=======================================================
Goal: Test hypothesis H2 - IEEE-CIS may not be a natural mixture of
distributions. If conformance varies dramatically by ProductCD (the
product category), then the data doesn't satisfy Hill's mixture theorem
within each category, explaining why per-user Benford is noisy.

If different categories have very different digit distributions, it means:
  - Legit users shopping in one category (e.g., ProductCD='W') see
    distorted Benford even though they're legitimate
  - This explains the puzzling "legit > fraud" Excess MAD we saw

Outputs:
  figuress/benford_by_category.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benford_core import BENFORD_P, digit_distribution, mad

DATA_DIR = Path(__file__).parent.parent / "data"
figures_DIR = Path(__file__).parent.parent / "figures"
TXN_FILE = DATA_DIR / "train_transaction.csv"


def main():
    cols_needed = [
        "isFraud", "TransactionAmt", "ProductCD",
        "card1", "addr1", "P_emaildomain",
    ]
    df = pd.read_csv(TXN_FILE, usecols=cols_needed)

    print(f"\nTotal transactions: {len(df):,}")
    print(f"\nProductCD distribution:")
    print(df["ProductCD"].value_counts())

    # Compute digit distribution per category, per class
    categories = df["ProductCD"].dropna().unique()
    categories = sorted(categories)

    results = []
    for cat in categories:
        cat_df = df[df["ProductCD"] == cat]
        legit_amts = cat_df[cat_df["isFraud"] == 0]["TransactionAmt"].values
        fraud_amts = cat_df[cat_df["isFraud"] == 1]["TransactionAmt"].values

        _, legit_props = digit_distribution(legit_amts)
        _, fraud_props = digit_distribution(fraud_amts)

        legit_mad = mad(legit_amts) if len(legit_amts) > 9 else np.nan
        fraud_mad = mad(fraud_amts) if len(fraud_amts) > 9 else np.nan

        results.append({
            "category": cat,
            "n_legit": len(legit_amts),
            "n_fraud": len(fraud_amts),
            "legit_mad": legit_mad,
            "fraud_mad": fraud_mad,
            "legit_props": legit_props,
            "fraud_props": fraud_props,
        })

    # Print summary
    print(f"\n{'=' * 70}")
    print("BENFORD MAD BY PRODUCT CATEGORY")
    print(f"{'=' * 70}")
    print(f"{'Category':<10} {'n_legit':>10} {'n_fraud':>10} "
          f"{'legit_MAD':>10} {'fraud_MAD':>10} {'diff':>8}")

    for r in results:
        diff = r['fraud_mad'] - r['legit_mad'] if not np.isnan(r['legit_mad']) else np.nan
        print(
            f"{r['category']:<10} "
            f"{r['n_legit']:>10,} {r['n_fraud']:>10,} "
            f"{r['legit_mad']:>10.4f} {r['fraud_mad']:>10.4f} "
            f"{diff:>+8.4f}"
        )

    # Visualize per-category Benford conformance
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4), sharey=True)
    if n_cats == 1:
        axes = [axes]

    digits = np.arange(1, 10)
    width = 0.35

    for ax, r in zip(axes, results):
        ax.bar(digits - width / 2, r["legit_props"], width,
               label=f"Legit (MAD={r['legit_mad']:.3f})",
               color="steelblue", alpha=0.85)
        ax.bar(digits + width / 2, r["fraud_props"], width,
               label=f"Fraud (MAD={r['fraud_mad']:.3f})",
               color="coral", alpha=0.85)
        ax.plot(digits, BENFORD_P, "k--o", label="Benford expected",
                linewidth=1.5, markersize=4)
        ax.set_xlabel("Leading digit")
        ax.set_title(
            f"ProductCD = '{r['category']}'\n"
            f"n_legit={r['n_legit']:,}, n_fraud={r['n_fraud']:,}"
        )
        ax.set_xticks(digits)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Proportion")
    plt.tight_layout()

    out_path = figuresS_DIR / "benford_by_category.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figures: {out_path.resolve()}")


if __name__ == "__main__":
    main()