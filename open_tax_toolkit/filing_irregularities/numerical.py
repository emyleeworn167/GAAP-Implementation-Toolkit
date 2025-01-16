"""Numerical analysis for detecting filing irregularities.

Identifies mathematical inconsistencies within individual filings:
- Sum/subtotal validation (do reported components add up?)
- Rounding pattern analysis (excessive rounding suggests estimation)
- Digit frequency anomalies within a single return
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SumCheckResult:
    """Result of a sum-validation check."""

    filer_id: str
    expected_sum_field: str
    component_fields: list[str]
    reported_total: float
    computed_total: float
    difference: float
    passes: bool


@dataclass
class NumericalAnalyzer:
    """Analyze numerical patterns within tax filings.

    Parameters
    ----------
    sum_tolerance : float
        Allowable relative difference for sum checks (default 0.5% for rounding).
    rounding_threshold : float
        Fraction of values ending in 000 before flagging (default 0.5 = 50%).
    """

    sum_tolerance: float = 0.005
    rounding_threshold: float = 0.50

    def validate_sums(
        self,
        df: pd.DataFrame,
        total_col: str,
        component_cols: list[str],
        id_col: str = "filer_id",
    ) -> list[SumCheckResult]:
        """Validate that component columns sum to the total column.

        A fundamental integrity check: does `total_col ≈ sum(component_cols)`?
        Catches both data entry errors and deliberate manipulation.
        """
        results: list[SumCheckResult] = []

        for _, row in df.iterrows():
            reported = row[total_col]
            computed = sum(row[c] for c in component_cols if pd.notna(row[c]))

            if pd.isna(reported):
                continue

            denom = max(abs(reported), abs(computed), 1e-10)
            rel_diff = abs(reported - computed) / denom
            passes = rel_diff <= self.sum_tolerance

            results.append(
                SumCheckResult(
                    filer_id=row[id_col],
                    expected_sum_field=total_col,
                    component_fields=component_cols,
                    reported_total=round(reported, 2),
                    computed_total=round(computed, 2),
                    difference=round(reported - computed, 2),
                    passes=passes,
                )
            )

        return results

    def detect_rounding_patterns(
        self,
        df: pd.DataFrame,
        value_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Detect excessive rounding in financial figures.

        A high fraction of round numbers (ending in 000, 00, etc.) across
        a filing may indicate estimated rather than actual figures.

        Returns a DataFrame with one row per column, showing the fraction
        of values at each rounding level.
        """
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        rows = []
        for col in value_cols:
            values = df[col].dropna().values
            n = len(values)
            if n == 0:
                continue

            # Check rounding at different levels
            round_1000 = np.sum(np.abs(values) % 1000 == 0) / n
            round_100 = np.sum(np.abs(values) % 100 == 0) / n
            round_10 = np.sum(np.abs(values) % 10 == 0) / n

            rows.append(
                {
                    "column": col,
                    "n_values": n,
                    "frac_round_1000": round(round_1000, 4),
                    "frac_round_100": round(round_100, 4),
                    "frac_round_10": round(round_10, 4),
                    "suspicious": round_1000 >= self.rounding_threshold,
                }
            )

        return pd.DataFrame(rows)

    def last_two_digits_test(self, values: np.ndarray | pd.Series) -> dict:
        """Test uniformity of last two digits (Nigrini's last-two-digits test).

        In naturally occurring data, the last two digits of large numbers
        should be approximately uniform over 00-99. Non-uniformity suggests
        manual data fabrication.

        Returns dict with chi2 statistic, p-value, and conformity boolean.
        """
        from open_tax_toolkit.utils.stats import chi_squared_test

        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        arr = np.abs(arr[arr >= 100])  # Need values >= 100 for last-two-digit meaning

        last_two = (arr % 100).astype(int)
        observed = np.array([np.sum(last_two == d) for d in range(100)])
        expected = np.full(100, len(last_two) / 100)

        chi2, p_value = chi_squared_test(observed, expected)

        return {
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 6),
            "uniform": p_value >= 0.05,
            "n_values": len(arr),
        }
