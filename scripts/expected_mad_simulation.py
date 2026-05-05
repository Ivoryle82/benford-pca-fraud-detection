"""
Phase 2b: Monte Carlo simulation of E[MAD] under the Benford null
==================================================================
Empirically estimates E[MAD(N)] = expected MAD when N samples are drawn
from a true Benford distribution. We need this for the small-sample
correction defined in our paper:

    Excess MAD = MAD(observed) - E[MAD(N)]

Theoretical scaling: E[MAD] ~ k / sqrt(N), where k is estimated below.

Output:
  - figuress/expected_mad_curve.png    : the E[MAD] vs. N plot
  - data/expected_mad_table.csv       : lookup table for the per-user pipeline
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benford_core import BENFORD_P, mad


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# Data/figure paths
DATA_DIR = Path(__file__).parent.parent / "data"
figures_DIR = Path(__file__).parent.parent / "figures"

# Sample sizes to evaluate
SAMPLE_SIZES = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]

# Number of Monte Carlo trials per sample size
N_TRIALS = 5000

RNG = np.random.default_rng(seed=42)


def sample_benford(n):
    """Draw n samples from a true Benford distribution (digits 1..9)."""
    return RNG.choice(np.arange(1, 10), size=n, p=BENFORD_P)


def expected_mad(n_samples, n_trials=N_TRIALS):
    """
    Monte Carlo estimate of E[MAD] when drawing N samples from a true
    Benford distribution. Returns mean and std across trials.
    """
    mads = np.empty(n_trials)
    for i in range(n_trials):
        digits = sample_benford(n_samples)
        # Compute MAD directly from digit counts (skip leading_digit step)
        counts = np.bincount(digits, minlength=10)[1:10]
        proportions = counts / counts.sum()
        mads[i] = np.mean(np.abs(proportions - BENFORD_P))
    return mads.mean(), mads.std()


def main():
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Trials per N: {N_TRIALS:,}")
    print()

    rows = []
    for n in SAMPLE_SIZES:
        mu, sigma = expected_mad(n)
        rows.append({"N": n, "E_MAD": mu, "std_MAD": sigma})
        print(f"  N={n:5d}: E[MAD] = {mu:.5f}  (std = {sigma:.5f})")

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "expected_mad_table.csv"
    df.to_csv(out_path, index=False)


    # ------------------------------------------------------
    # Fit k / sqrt(N) model
    # ------------------------------------------------------
    # If E[MAD] ~ k / sqrt(N), then sqrt(N) * E[MAD] = k (constant).
    # Estimate k by least squares.
    sqrt_n = np.sqrt(df["N"].values)
    k_per_n = sqrt_n * df["E_MAD"].values
    k_estimate = k_per_n.mean()
    k_std = k_per_n.std()

    print(f"\nFit: E[MAD] ≈ k / sqrt(N)")
    print(f"  k estimated = {k_estimate:.4f}  (std = {k_std:.4f})")

    # ------------------------------------------------------
    # Visualize
    # ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: E[MAD] vs N (log-log)
    n_dense = np.logspace(np.log10(10), np.log10(1000), 200)
    axes[0].loglog(df["N"], df["E_MAD"], "o", color="steelblue",
                   markersize=8, label="Monte Carlo")
    axes[0].loglog(n_dense, k_estimate / np.sqrt(n_dense),
                   "--", color="coral",
                   label=f"k/√N fit, k={k_estimate:.4f}")
    axes[0].set_xlabel("Sample size N (log)")
    axes[0].set_ylabel("E[MAD] (log)")
    axes[0].set_title("Expected MAD under Benford null vs. N")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # Right: residuals around fit (linear scale)
    axes[1].plot(df["N"], df["E_MAD"], "o-", color="steelblue",
                 label="Monte Carlo E[MAD]")
    axes[1].fill_between(df["N"],
                         df["E_MAD"] - df["std_MAD"],
                         df["E_MAD"] + df["std_MAD"],
                         alpha=0.2, color="steelblue", label="±1 std")
    # Nigrini cutoff lines
    axes[1].axhline(0.006, color="green", linestyle=":", label="close conformity (0.006)")
    axes[1].axhline(0.012, color="orange", linestyle=":", label="acceptable (0.012)")
    axes[1].axhline(0.015, color="red", linestyle=":", label="nonconformity (0.015)")
    axes[1].set_xlabel("Sample size N")
    axes[1].set_ylabel("E[MAD]")
    axes[1].set_title("E[MAD] vs. Nigrini cutoffs")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = figuresS_DIR / "expected_mad_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")



if __name__ == "__main__":
    main()