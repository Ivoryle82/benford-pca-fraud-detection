"""
Phase 2a: Core Benford's Law math
=================================
Pure functions for leading-digit extraction and conformance tests.
Imported by other Phase 2 scripts.

Math reference: see the LaTeX derivation in paper/benford_derivation.tex
"""

import numpy as np


# Benford expected probabilities
def benford_probs():
    """
    Return the Benford probabilities for digits 1..9.
    P(d) = log10(1 + 1/d)
    """
    digits = np.arange(1, 10)
    return np.log10(1 + 1 / digits)


BENFORD_P = benford_probs()  # cached, length-9 array


# Leading digit extraction
def leading_digit(x):
    """
    Extract the first significant digit of x.

    Works on positive numbers only. Strips leading zeros from values < 1
    (e.g., 0.0345 -> 3). Returns an integer in {1,...,9}.

    Vectorized: accepts scalars or numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        # Filter out non-positive values; they have no leading digit
        x = x[x > 0]

    # log10(x) + small epsilon to avoid log(0); 10**(fractional part) gives the
    # mantissa in [1, 10), and floor-int gives the leading digit
    log_x = np.log10(x)
    mantissa = 10 ** (log_x - np.floor(log_x))
    return np.floor(mantissa).astype(int)


def digit_distribution(values):
    """
    Given an array of positive numbers, return the empirical
    leading-digit frequency distribution as a length-9 array.
    Returns counts (not normalized) and proportions.
    """
    values = np.asarray(values, dtype=float)
    values = values[values > 0]  # Benford only defined for positive values
    if len(values) == 0:
        return np.zeros(9, dtype=int), np.zeros(9, dtype=float)

    digits = leading_digit(values)
    counts = np.bincount(digits, minlength=10)[1:10]  # bins 1..9
    proportions = counts / counts.sum()
    return counts, proportions


# Conformance tests
def chi_squared(values):
    """
    Pearson chi-squared statistic for Benford conformance.
    Returns NaN if N < 9 (not enough data for the test).

    Under the null (Benford), this follows chi^2 with 8 degrees of freedom.
    Critical value at 5% significance: 15.51
    """
    counts, _ = digit_distribution(values)
    n = counts.sum()
    if n < 9:
        return np.nan
    expected = n * BENFORD_P
    return float(np.sum((counts - expected) ** 2 / expected))


def mad(values):
    """
    Mean Absolute Deviation from Benford.
    MAD = (1/9) * sum_d |O_d/N - P_d|

    Returns NaN if N < 9. Lower = closer to Benford.

    Nigrini's empirical cutoffs (2012):
      < 0.006 : close conformity
      < 0.012 : acceptable
      < 0.015 : marginal
      >=0.015 : nonconformity
    """
    counts, proportions = digit_distribution(values)
    if counts.sum() < 9:
        return np.nan
    return float(np.mean(np.abs(proportions - BENFORD_P)))


def z_statistics(values):
    """
    Per-digit Z-statistic with continuity correction (Nigrini 2012).
    Returns a length-9 array. |Z| > 1.96 indicates significant deviation
    at the 5% level for that digit.
    """
    counts, _ = digit_distribution(values)
    n = counts.sum()
    if n == 0:
        return np.full(9, np.nan)

    observed_p = counts / n
    expected_p = BENFORD_P
    # Continuity correction: 1/(2N), only applied if it would shrink the gap
    correction = 1 / (2 * n)
    raw_diff = np.abs(observed_p - expected_p)
    diff = np.maximum(raw_diff - correction, 0.0)
    se = np.sqrt(expected_p * (1 - expected_p) / n)
    return diff / se