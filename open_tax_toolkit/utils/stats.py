"""Statistical utilities used across detection modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sp_stats


def iqr_range(values: ArrayLike) -> tuple[float, float, float, float]:
    """Compute interquartile range boundaries (OECD transfer pricing standard).

    Returns (Q1, median, Q3, IQR) where IQR = Q3 - Q1.
    The arm's length range is [Q1, Q3] per OECD Transfer Pricing Guidelines Ch. III.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    q1 = float(np.percentile(arr, 25))
    median = float(np.median(arr))
    q3 = float(np.percentile(arr, 75))
    return q1, median, q3, q3 - q1


def chi_squared_test(
    observed: ArrayLike, expected: ArrayLike, min_expected: float = 5.0
) -> tuple[float, float]:
    """Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : array-like
        Observed frequency counts.
    expected : array-like
        Expected frequency counts.
    min_expected : float
        Minimum expected count per bin (bins below this are merged).

    Returns
    -------
    (chi2_statistic, p_value)
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)

    # Merge bins with expected count below threshold (right-to-left)
    while len(exp) > 1 and exp[-1] < min_expected:
        obs[-2] += obs[-1]
        exp[-2] += exp[-1]
        obs = obs[:-1]
        exp = exp[:-1]

    chi2, p_value = sp_stats.chisquare(obs, f_exp=exp)
    return float(chi2), float(p_value)


def z_score_outliers(values: ArrayLike, threshold: float = 2.5) -> np.ndarray:
    """Identify outliers using modified Z-scores (MAD-based).

    Uses median absolute deviation instead of standard deviation for
    robustness against the very outliers we're trying to detect.

    Returns boolean array where True indicates an outlier.
    """
    arr = np.asarray(values, dtype=float)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    # 0.6745 is the 0.75th quantile of the standard normal distribution
    modified_z = 0.6745 * (arr - median) / max(mad, 1e-10)
    return np.abs(modified_z) > threshold
