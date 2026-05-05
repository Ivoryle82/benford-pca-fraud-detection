"""
Phase 2: Per-user Benford feature extraction
==============================================
For each user in the IEEE-CIS data (or our synthetic stand-in), compute:

  - benford_amt    : Excess MAD on transaction amounts
  - benford_time   : Excess MAD on inter-transaction time deltas
  - benford_ratio  : Excess MAD on (amount / user_mean_amount)

Plus baseline per-user features (n_txn, mean_amt, std_amt, etc.).

Outputs a per-user feature table ready for ML training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benford_core import mad


DATA_DIR = Path(__file__).parent.parent / "data"

# Column constants (match phase1_eda.py)
TXN_FILE = DATA_DIR / "train_transaction.csv"  # change to train_transaction.csv for real data
MIN_TXN_PER_USER = 30  # users below this are excluded


def load_and_prep():
    """Load transactions and infer user_id."""
    cols_needed = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        "card1", "addr1", "P_emaildomain",
    ]
    print(f"Loading {TXN_FILE.name}...")
    df = pd.read_csv(TXN_FILE, usecols=cols_needed)

    df["user_id"] = (
        df["card1"].fillna(-1).astype(int).astype(str)
        + "_"
        + df["addr1"].fillna(-1).astype(int).astype(str)
        + "_"
        + df["P_emaildomain"].fillna("none").astype(str)
    )
    print(f"  {len(df):,} transactions, {df['user_id'].nunique():,} users")
    return df


def load_emad_table():
    """Load the E[MAD] lookup table from the Monte Carlo simulation."""
    path = DATA_DIR / "expected_mad_table.csv"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Run expected_mad_simulation.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def expected_mad_for_n(n, emad_table):
    """
    Look up E[MAD] for sample size n using the simulation table.
    Linear interpolation between simulated points.
    """
    if n < emad_table["N"].min():
        # Below table range: use smallest available
        return emad_table["E_MAD"].iloc[0]
    if n > emad_table["N"].max():
        # Above range: extrapolate using k/sqrt(N) scaling
        n_max = emad_table["N"].max()
        emad_max = emad_table["E_MAD"].iloc[-1]
        return emad_max * np.sqrt(n_max / n)
    return float(np.interp(n, emad_table["N"], emad_table["E_MAD"]))


def excess_mad(values, emad_table):
    """
    Compute MAD - E[MAD] for the given sample.
    Returns NaN if the sample is too small or has no positive values.
    """
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    n = len(values)
    if n < 9:
        return np.nan
    raw_mad = mad(values)
    if np.isnan(raw_mad):
        return np.nan
    e_mad = expected_mad_for_n(n, emad_table)
    return raw_mad - e_mad


def per_user_features(df, emad_table):
    """
    For each user with >= MIN_TXN_PER_USER transactions, compute the
    Benford feature vector and baseline features.
    """
    features = []
    user_groups = df.sort_values(["user_id", "TransactionDT"]).groupby("user_id")

    print(f"Computing features for users with >= {MIN_TXN_PER_USER} transactions...")
    n_processed = 0
    n_kept = 0

    for user_id, g in user_groups:
        n_processed += 1
        if len(g) < MIN_TXN_PER_USER:
            continue
        n_kept += 1

        amts = g["TransactionAmt"].values

        # Time deltas: consecutive differences in TransactionDT (in seconds)
        time_deltas = np.diff(g["TransactionDT"].values)
        # Filter zero/negative deltas (same-second transactions)
        time_deltas = time_deltas[time_deltas > 0]

        # Ratios: each amount / user's mean amount
        mean_amt = amts.mean()
        ratios = amts / mean_amt if mean_amt > 0 else np.array([])

        # Benford features
        b_amt = excess_mad(amts, emad_table)
        b_time = excess_mad(time_deltas, emad_table)
        b_ratio = excess_mad(ratios, emad_table)

        features.append({
            "user_id": user_id,
            "n_txn": len(g),
            "mean_amt": mean_amt,
            "std_amt": amts.std(),
            "median_amt": np.median(amts),
            "max_amt": amts.max(),
            "median_time_delta": np.median(time_deltas) if len(time_deltas) > 0 else np.nan,
            "benford_amt": b_amt,
            "benford_time": b_time,
            "benford_ratio": b_ratio,
            "isFraud": int(g["isFraud"].max()),  # any fraud -> labeled fraud
        })

        if n_processed % 500 == 0:
            print(f"  processed {n_processed:,} users, kept {n_kept:,}")

    feat_df = pd.DataFrame(features)
    print(f"\nFinal: {len(feat_df):,} users in feature table")
    return feat_df


def main():
    df = load_and_prep()
    emad_table = load_emad_table()
    feat_df = per_user_features(df, emad_table)

    # Save
    out_path = DATA_DIR / "user_features.csv"
    feat_df.to_csv(out_path, index=False)

    # summary by class
    print("FEATURE SUMMARY BY CLASS (Benford features only)")
    print("=" * 60)
    summary = feat_df.groupby("isFraud")[
        ["benford_amt", "benford_time", "benford_ratio"]
    ].agg(["mean", "median", "std"]).round(5)
    print(summary)

    n_fraud = feat_df["isFraud"].sum()
    n_legit = len(feat_df) - n_fraud
    print(f"\nLegitimate users: {n_legit:,}")
    print(f"Fraudulent users: {n_fraud:,}")
    print(f"Class balance:    {n_fraud / len(feat_df):.2%} fraud")


if __name__ == "__main__":
    main()