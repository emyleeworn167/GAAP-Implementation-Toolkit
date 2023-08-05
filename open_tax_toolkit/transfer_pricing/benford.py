"""Benford's Law analysis for financial figure manipulation detection.

Benford's Law predicts the frequency distribution of leading digits in
naturally occurring numerical datasets. Deviations from the expected
distribution can indicate fabricated or manipulated financial data.

This module implements first-digit and second-digit Benford tests with
chi-squared goodness-of-fit testing, following the methodology established
in Nigrini (2012) "Benford's Law: Applications for Forensic Accounting,
Auditing, and Fraud Detection."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from open_tax_toolkit.utils.stats import chi_squared_test


def _benford_expected(digit_position: int = 1) -> np.ndarray:
    """Return expected Benford probabilities for digit position 1 or 2."""
    if digit_position == 1:
        # P(d) = log10(1 + 1/d) for d = 1..9
        return np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
    elif digit_position == 2:
        # P(d2) = sum over d1 of log10(1 + 1/(10*d1 + d2)) for d2 = 0..9
        probs = []
        for d2 in range(10):
            p = sum(np.log10(1 + 1 / (10 * d1 + d2)) for d1 in range(1, 10))
            probs.append(p)
        return np.array(probs)
    else:
        raise ValueError("Only digit positions 1 and 2 are supported")


def _extract_leading_digits(values: np.ndarray, position: int = 1) -> np.ndarray:
    """Extract the leading digit at the given position from each value."""
    abs_vals = np.abs(values)
    abs_vals = abs_vals[abs_vals > 0]
    # Normalize to have first digit before decimal point
    exponents = np.floor(np.log10(abs_vals))
    normalized = abs_vals / (10.0**exponents)

    if position == 1:
        return np.floor(normalized).astype(int)
    elif position == 2:
        return np.floor(normalized * 10).astype(int) % 10
    else:
        raise ValueError("Only positions 1 and 2 are supported")


@dataclass
class BenfordResult:
    """Result of a Benford's Law analysis."""

    digit_position: int
    observed_freq: np.ndarray
    expected_freq: np.ndarray
    chi2_statistic: float
    p_value: float
    conforms: bool
    n_values: int
    max_deviation_digit: int
    max_deviation: float


@dataclass
class BenfordAnalysis:
    """Benford's Law conformity testing for financial datasets.

    Parameters
    ----------
    significance_level : float
        P-value threshold below which we reject the null hypothesis
        that the data follows Benford's Law.
    min_sample_size : int
        Minimum number of values required for a meaningful test.
    """

    significance_level: float = 0.05
    min_sample_size: int = 50

    def test(
        self,
        values: np.ndarray | pd.Series,
        digit_position: int = 1,
    ) -> BenfordResult:
        """Test whether values conform to Benford's Law.

        Parameters
        ----------
        values : array-like
            Financial figures to test (e.g., revenue, expenses, payments).
        digit_position : int
            Which digit to analyze (1 = first digit, 2 = second digit).

        Returns
        -------
        BenfordResult
        """
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        arr = arr[arr != 0]

        if len(arr) < self.min_sample_size:
            raise ValueError(
                f"Need at least {self.min_sample_size} non-zero values, got {len(arr)}"
            )

        digits = _extract_leading_digits(arr, position=digit_position)
        expected_probs = _benford_expected(digit_position)

        if digit_position == 1:
            digit_range = range(1, 10)
            observed = np.array([np.sum(digits == d) for d in digit_range])
        else:
            digit_range = range(0, 10)
            observed = np.array([np.sum(digits == d) for d in digit_range])

        expected = expected_probs * len(digits)
        chi2, p_value = chi_squared_test(observed, expected)

        # Find the digit with maximum deviation
        deviations = np.abs(observed / len(digits) - expected_probs)
        if digit_position == 1:
            max_dev_digit = int(np.argmax(deviations)) + 1
        else:
            max_dev_digit = int(np.argmax(deviations))

        return BenfordResult(
            digit_position=digit_position,
            observed_freq=observed,
            expected_freq=expected.round(2),
            chi2_statistic=round(chi2, 4),
            p_value=round(p_value, 6),
            conforms=p_value >= self.significance_level,
            n_values=len(digits),
            max_deviation_digit=max_dev_digit,
            max_deviation=round(float(deviations.max()), 6),
        )

    def test_dataframe(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        digit_position: int = 1,
    ) -> pd.DataFrame:
        """Test multiple columns of a DataFrame for Benford conformity.

        Returns a summary DataFrame with one row per tested column.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        rows = []
        for col in columns:
            values = df[col].dropna()
            if len(values) < self.min_sample_size:
                continue
            result = self.test(values, digit_position=digit_position)
            rows.append(
                {
                    "column": col,
                    "n_values": result.n_values,
                    "chi2": result.chi2_statistic,
                    "p_value": result.p_value,
                    "conforms": result.conforms,
                    "max_deviation_digit": result.max_deviation_digit,
                    "max_deviation": result.max_deviation,
                }
            )
        return pd.DataFrame(rows)
