"""
Diagnostic 1: Inspect individual user transaction patterns
============================================================
Goal: Understand WHY legit frequent users have higher Excess MAD than
expected. Hypotheses to check:

  H1 - Subscription contamination: legit users have recurring amounts
       like $9.99, $14.99, $29.99 that cluster digit distributions
  H2 - Card-testing attacks: fraud "users" have many small similar
       amounts that happen to span Benford-favorable range
  H3 - Merchant concentration: legit users transact with few merchants,
       creating narrow amount distributions

We sample several users from each class and print their amount patterns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benford_core import leading_digit, mad

DATA_DIR = Path(__file__).parent.parent / "data"
TXN_FILE = DATA_DIR / "train_transaction.csv"


def inspect_user(df, user_id, label):
    """Print a detailed profile of one user."""
    g = df[df["user_id"] == user_id].sort_values("TransactionDT")
    amts = g["TransactionAmt"].values

    print(f"{label} | user_id={user_id[:50]}...")
    print(f"  n_transactions: {len(g)}")
    print(f"  fraud in this user: {g['isFraud'].sum()} / {len(g)}")
    print(f"  amount range: ${amts.min():.2f} - ${amts.max():.2f}")
    print(f"  amount mean: ${amts.mean():.2f}  median: ${np.median(amts):.2f}")
    print(f"  log10 range: {np.log10(amts.max()/amts.min()):.2f} orders of magnitude")

    # Count repeated amounts (subscription-like behavior)
    amount_counts = pd.Series(amts).value_counts()
    top_repeats = amount_counts.head(5)
    repeat_count = (amount_counts > 1).sum()

    print(f"\n  unique amounts: {len(amount_counts)}")
    print(f"  amounts that repeat (>=2 times): {repeat_count}")
    print(f"  top 5 most-repeated amounts:")
    for amt, cnt in top_repeats.items():
        pct = 100 * cnt / len(g)
        print(f"    ${amt:>8.2f} appears {cnt:>4}x ({pct:.1f}%)")

    # Leading-digit distribution
    digits = leading_digit(amts)
    digit_counts = np.bincount(digits, minlength=10)[1:10]
    print(f"\n  leading digit counts:")
    print(f"    digit:  1   2   3   4   5   6   7   8   9")
    print(f"    count: " + "  ".join(f"{c:3d}" for c in digit_counts))
    print(f"    MAD from Benford: {mad(amts):.4f}")


def main():
    print("Loading transaction data (this takes ~30 seconds)...")
    cols_needed = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        "card1", "addr1", "P_emaildomain", "ProductCD",
    ]
    df = pd.read_csv(TXN_FILE, usecols=cols_needed)

    # Same user inference as the main pipeline
    df["user_id"] = (
        df["card1"].fillna(-1).astype(int).astype(str) + "_"
        + df["addr1"].fillna(-1).astype(int).astype(str) + "_"
        + df["P_emaildomain"].fillna("none").astype(str)
    )

    # Focus on users with >= 30 transactions
    counts = df.groupby("user_id").size()
    frequent_users = counts[counts >= 30].index
    df_freq = df[df["user_id"].isin(frequent_users)].copy()

    # Classify each user as fraud-positive if they have any fraud
    user_fraud = df_freq.groupby("user_id")["isFraud"].max()

    legit_users = user_fraud[user_fraud == 0].index.tolist()
    fraud_users = user_fraud[user_fraud == 1].index.tolist()

    print(f"\nFrequent users: {len(frequent_users):,}")
    print(f"  legit: {len(legit_users):,}")
    print(f"  fraud: {len(fraud_users):,}")

    # Sort by transaction count to inspect diverse users
    user_n_txn = df_freq.groupby("user_id").size()

    # Pick a few users from each class, with varying transaction counts
    rng = np.random.default_rng(42)

    # Legit users: sample 3 with different volumes
    legit_sorted = sorted(legit_users, key=lambda u: user_n_txn[u])
    legit_sample = [
        legit_sorted[len(legit_sorted) // 4],   # lower quartile
        legit_sorted[len(legit_sorted) // 2],   # median
        legit_sorted[-10],                       # high volume
    ]

    # Fraud users: sample 3 with different volumes
    fraud_sorted = sorted(fraud_users, key=lambda u: user_n_txn[u])
    fraud_sample = [
        fraud_sorted[len(fraud_sorted) // 4],
        fraud_sorted[len(fraud_sorted) // 2],
        fraud_sorted[-10],
    ]

    print("LEGITIMATE USERS")
    for u in legit_sample:
        inspect_user(df_freq, u, label="LEGIT")

    print("FRAUDULENT USERS")
    for u in fraud_sample:
        inspect_user(df_freq, u, label="FRAUD")

    # Aggregate diagnostic: subscription score across classes
    print("# AGGREGATE DIAGNOSTIC: How much do amounts repeat?")
    def repeat_score(amts):
        """Fraction of transactions whose amount appears >=2 times."""
        if len(amts) == 0:
            return np.nan
        counts = pd.Series(amts).value_counts()
        repeated_amts = counts[counts >= 2].index
        return np.mean(pd.Series(amts).isin(repeated_amts))

    repeat_scores = []
    for u in frequent_users:
        amts = df_freq[df_freq["user_id"] == u]["TransactionAmt"].values
        is_fraud = user_fraud[u]
        repeat_scores.append({
            "user_id": u,
            "n_txn": len(amts),
            "repeat_score": repeat_score(amts),
            "isFraud": is_fraud,
        })

    rs_df = pd.DataFrame(repeat_scores)
    print("\nFraction of transactions with repeated amounts (subscription-like):")
    print(rs_df.groupby("isFraud")["repeat_score"].agg(["mean", "median", "std"]))

    print("\nHypothesis H1 (subscription contamination for legit):")
    legit_repeat = rs_df[rs_df["isFraud"] == 0]["repeat_score"].mean()
    fraud_repeat = rs_df[rs_df["isFraud"] == 1]["repeat_score"].mean()
    if legit_repeat > fraud_repeat + 0.05:
        print(f"  CONFIRMED: legit users repeat amounts more ({legit_repeat:.2%}) than")
        print(f"  fraud users ({fraud_repeat:.2%}). Subscriptions likely contaminating.")
    elif fraud_repeat > legit_repeat + 0.05:
        print(f"  INVERSE: fraud users repeat amounts more ({fraud_repeat:.2%}) than")
        print(f"  legit users ({legit_repeat:.2%}). Possible card-testing patterns.")
    else:
        print(f"  INCONCLUSIVE: similar repeat rates")
        print(f"  (legit: {legit_repeat:.2%}, fraud: {fraud_repeat:.2%})")


if __name__ == "__main__":
    main()