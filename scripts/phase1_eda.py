"""
Phase 1: IEEE-CIS Fraud Detection Dataset - Initial Exploration
================================================================
Purpose: Answer the feasibility questions before committing to the
         per-user Benford approach.

Key questions:
  1. How many transactions, and what's the fraud rate?
  2. Can we infer a "user" from card+address columns?
  3. How many transactions does each inferred user have?
  4. Is there enough per-user data for Benford analysis?
  5. Do transaction amounts span enough orders of magnitude?

Usage:
    python phase1_eda.py [--sample-rows N]

  --sample-rows: if set, only load the first N rows. Useful for a
                 first-pass sanity check on a laptop. Leave unset to
                 load the full dataset.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
figures_DIR = Path(__file__).parent.parent / "figures"
figures_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Only load the first N rows (useful for quick sanity checks)",
    )
    return p.parse_args()


def load_transactions(sample_rows=None):
    """Load the train_transaction.csv file."""
    path = DATA_DIR / "train_transaction.csv"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    # Only load columns we actually need - saves memory on the 700MB file
    cols_needed = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "P_emaildomain",
    ]

    if sample_rows:
        print(f"  (sample mode: first {sample_rows:,} rows)")
        df = pd.read_csv(path, nrows=sample_rows, usecols=cols_needed)
    else:
        df = pd.read_csv(path, usecols=cols_needed)
    return df


def main():
    args = parse_args()
    df = load_transactions(sample_rows=args.sample_rows)

    # 1. Basic stats
    n_total = len(df)
    n_fraud = df["isFraud"].sum()
    fraud_rate = n_fraud / n_total
    print("1. DATASET BASICS")
    print(f"Total transactions: {n_total:,}")
    print(f"Fraud transactions: {n_fraud:,} ({fraud_rate:.2%})")
    print(f"Legit transactions: {n_total - n_fraud:,}")

    # 2. Transaction amounts - Benford feasibility check
    amts = df["TransactionAmt"].dropna()
    log_range = np.log10(amts.max()) - np.log10(amts.min())

    print("2. TRANSACTION AMOUNTS (Benford feasibility)")

    print(f"min:    ${amts.min():.2f}")
    print(f"median: ${amts.median():.2f}")
    print(f"mean:   ${amts.mean():.2f}")
    print(f"max:    ${amts.max():.2f}")
    print(f"log10 range: {log_range:.1f} orders of magnitude")
    if log_range >= 3:
        print("  -> GOOD: spans multiple orders of magnitude (Benford-feasible)")
    else:
        print("  -> WARNING: narrow range may not produce clean Benford distribution")


    # 3. Infer user identity
    # IEEE-CIS has no explicit user_id. The Kaggle community uses combinations
    # of card/address/email columns. We start simple: card1 + addr1 + P_email_domain.
    df["user_id"] = (
        df["card1"].fillna(-1).astype(int).astype(str)
        + "_"
        + df["addr1"].fillna(-1).astype(int).astype(str)
        + "_"
        + df["P_emaildomain"].fillna("none").astype(str)
    )

    counts = df.groupby("user_id").size()


    print("3. INFERRED USERS")
    print(f"Inferred users: {len(counts):,}")
    print(f"Transactions per user:")
    print(f"  min:      {counts.min()}")
    print(f"  median:   {counts.median():.0f}")
    print(f"  mean:     {counts.mean():.1f}")
    print(f"  max:      {counts.max()}")
    print(f"  p90:      {counts.quantile(0.9):.0f}")
    print(f"  p99:      {counts.quantile(0.99):.0f}")


    # 4. User filtering thresholds
    print("4. PER-USER DATA AVAILABILITY")

    for threshold in [10, 20, 30, 50, 100, 200]:
        n_users = (counts >= threshold).sum()
        pct = n_users / len(counts)
        print(
            f"  users with >= {threshold:3d} transactions: "
            f"{n_users:7,} ({pct:.1%})"
        )

    # --------------------------------------------------------
    # 5. Fraud distribution at the user level
    # --------------------------------------------------------
    user_fraud = df.groupby("user_id")["isFraud"].agg(["sum", "count"])
    user_fraud.columns = ["n_fraud", "n_total"]

    n_users_with_fraud = (user_fraud["n_fraud"] > 0).sum()

    print(f"\n{'=' * 60}")
    print("5. FRAUD AT THE USER LEVEL")

    print(
        f"Users with >=1 fraudulent transaction: {n_users_with_fraud:,} "
        f"({n_users_with_fraud/len(user_fraud):.2%})"
    )

    # Users with >=30 transactions AND any fraud
    big_users = user_fraud[user_fraud["n_total"] >= 30]
    big_users_with_fraud = (big_users["n_fraud"] > 0).sum()
    print(f"\nUsers with >= 30 transactions: {len(big_users):,}")
    print(
        f"  of these, how many have any fraud: {big_users_with_fraud:,} "
        f"({big_users_with_fraud/max(len(big_users),1):.2%})"
    )

    # 6. Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(np.log10(amts), bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("log10(Transaction Amount)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Transaction amount distribution (log10)")

    axes[1].hist(
        np.log10(counts[counts >= 1]), bins=60, color="coral", edgecolor="white"
    )
    axes[1].axvline(
        np.log10(30), color="red", linestyle="--", label="N=30 threshold"
    )
    axes[1].set_xlabel("log10(transactions per user)")
    axes[1].set_ylabel("Number of users")
    axes[1].set_title("Per-user transaction count (log10)")
    axes[1].legend()

    plt.tight_layout()
    out_path = figuresS_DIR / "phase1_eda.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figures: {out_path.resolve()}")

    # 7. Summary / verdict
    print("7. FEASIBILITY VERDICT")

    n_feasible_users = (counts >= 30).sum()
    ok_range = log_range >= 3
    ok_users = n_feasible_users >= 1000
    ok_fraud = big_users_with_fraud >= 50

    print(f"[{'OK' if ok_range else 'NO'}] amount range >= 3 orders of magnitude")
    print(f"[{'OK' if ok_users else 'NO'}] >= 1000 users with >= 30 transactions")
    print(f"[{'OK' if ok_fraud else 'NO'}] >= 50 users with fraud (for model training)")

    if ok_range and ok_users and ok_fraud:
        print("\n>>> Phase 1 PASSED. Proceed to Phase 2 (Benford feature extraction).")
    else:
        print("\n>>> Phase 1 has concerns. Review output above before proceeding.")


if __name__ == "__main__":
    main()