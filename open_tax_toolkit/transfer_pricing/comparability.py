"""Arm's length comparability analysis for transfer pricing.

Implements the interquartile range (IQR) method prescribed by OECD Transfer
Pricing Guidelines (Chapter III) and used by the IRS in Treas. Reg. §1.482.

The tested party's profit level indicator is compared against the arm's length
range derived from uncontrolled comparables. Results outside [Q1, Q3] are
flagged for further examination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from open_tax_toolkit.utils.stats import iqr_range, z_score_outliers


@dataclass
class ComparabilityResult:
    """Result of a comparability analysis."""

    tested_value: float
    q1: float
    median: float
    q3: float
    iqr: float
    is_within_range: bool
    adjustment_to_median: float
    comparable_count: int
    outliers_removed: int


@dataclass
class ComparabilityAnalysis:
    """OECD-aligned comparability analysis using the interquartile range method.

    Parameters
    ----------
    indicator : str
        Profit level indicator column name (e.g., 'operating_margin', 'berry_ratio').
    remove_outliers : bool
        Whether to remove statistical outliers from comparables before computing the range.
    outlier_threshold : float
        Modified Z-score threshold for outlier removal.
    """

    indicator: str = "operating_margin"
    remove_outliers: bool = True
    outlier_threshold: float = 2.5
    _results: list[ComparabilityResult] = field(default_factory=list, repr=False)

    def analyze(
        self,
        comparables: pd.DataFrame,
        tested_party_value: float,
    ) -> ComparabilityResult:
        """Run comparability analysis for a single tested party.

        Parameters
        ----------
        comparables : DataFrame
            Must contain a column matching ``self.indicator``.
        tested_party_value : float
            The tested party's profit level indicator value.

        Returns
        -------
        ComparabilityResult
        """
        values = comparables[self.indicator].dropna().values
        original_count = len(values)

        outliers_removed = 0
        if self.remove_outliers and len(values) > 5:
            mask = z_score_outliers(values, threshold=self.outlier_threshold)
            outliers_removed = int(mask.sum())
            values = values[~mask]

        q1, median, q3, iqr_val = iqr_range(values)
        within = q1 <= tested_party_value <= q3
        adjustment = median - tested_party_value if not within else 0.0

        result = ComparabilityResult(
            tested_value=tested_party_value,
            q1=round(q1, 6),
            median=round(median, 6),
            q3=round(q3, 6),
            iqr=round(iqr_val, 6),
            is_within_range=within,
            adjustment_to_median=round(adjustment, 6),
            comparable_count=len(values),
            outliers_removed=outliers_removed,
        )
        self._results.append(result)
        return result

    def screen_batch(
        self,
        comparables: pd.DataFrame,
        tested_parties: pd.DataFrame,
        entity_col: str = "entity",
    ) -> pd.DataFrame:
        """Screen multiple tested parties against the same comparable set.

        Returns a DataFrame with one row per tested party, including
        the arm's length range and whether each is within range.
        """
        rows = []
        for _, row in tested_parties.iterrows():
            result = self.analyze(comparables, row[self.indicator])
            rows.append(
                {
                    "entity": row[entity_col],
                    "tested_value": result.tested_value,
                    "q1": result.q1,
                    "median": result.median,
                    "q3": result.q3,
                    "within_range": result.is_within_range,
                    "adjustment_to_median": result.adjustment_to_median,
                }
            )
        return pd.DataFrame(rows)
