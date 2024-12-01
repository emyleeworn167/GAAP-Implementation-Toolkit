"""Cross-document consistency checking for tax filings.

Detects discrepancies between paired documents that should report consistent
figures — e.g., a primary return (Form 1120) vs. supporting schedules,
or reported income vs. third-party information returns (1099s, W-2s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DiscrepancyRecord:
    """A single detected discrepancy between two documents."""

    filer_id: str
    field: str
    value_a: float
    value_b: float
    absolute_diff: float
    relative_diff: float
    severity: str  # "low", "medium", "high"


@dataclass
class ConsistencyChecker:
    """Check numerical consistency between paired tax documents.

    Parameters
    ----------
    relative_threshold : float
        Minimum relative difference to flag (e.g., 0.01 = 1%).
    absolute_threshold : float
        Minimum absolute difference to flag (avoids flagging rounding on small values).
    severity_thresholds : tuple[float, float]
        (medium, high) relative difference thresholds for severity classification.
    """

    relative_threshold: float = 0.01
    absolute_threshold: float = 100.0
    severity_thresholds: tuple[float, float] = (0.05, 0.15)

    def _classify_severity(self, relative_diff: float) -> str:
        med, high = self.severity_thresholds
        if relative_diff >= high:
            return "high"
        elif relative_diff >= med:
            return "medium"
        return "low"

    def check_pair(
        self,
        doc_a: pd.DataFrame,
        doc_b: pd.DataFrame,
        id_col: str = "filer_id",
        check_cols: list[str] | None = None,
    ) -> list[DiscrepancyRecord]:
        """Compare two documents field-by-field for each filer.

        Parameters
        ----------
        doc_a, doc_b : DataFrame
            Paired documents with a shared identifier column.
        id_col : str
            Column name for the filer/entity identifier.
        check_cols : list[str] | None
            Columns to compare. If None, compares all shared numeric columns.

        Returns
        -------
        list of DiscrepancyRecord
        """
        merged = doc_a.merge(doc_b, on=id_col, suffixes=("_a", "_b"))

        if check_cols is None:
            # Find numeric columns present in both (by suffix matching)
            a_cols = {c.removesuffix("_a") for c in merged.columns if c.endswith("_a")}
            b_cols = {c.removesuffix("_b") for c in merged.columns if c.endswith("_b")}
            check_cols = sorted(a_cols & b_cols)

        discrepancies: list[DiscrepancyRecord] = []

        for _, row in merged.iterrows():
            for col in check_cols:
                val_a = row.get(f"{col}_a")
                val_b = row.get(f"{col}_b")
                if pd.isna(val_a) or pd.isna(val_b):
                    continue

                abs_diff = abs(val_a - val_b)
                denom = max(abs(val_a), abs(val_b), 1e-10)
                rel_diff = abs_diff / denom

                if rel_diff >= self.relative_threshold and abs_diff >= self.absolute_threshold:
                    discrepancies.append(
                        DiscrepancyRecord(
                            filer_id=row[id_col],
                            field=col,
                            value_a=round(val_a, 2),
                            value_b=round(val_b, 2),
                            absolute_diff=round(abs_diff, 2),
                            relative_diff=round(rel_diff, 4),
                            severity=self._classify_severity(rel_diff),
                        )
                    )

        return discrepancies

    def check_pair_summary(
        self,
        doc_a: pd.DataFrame,
        doc_b: pd.DataFrame,
        id_col: str = "filer_id",
        check_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return discrepancies as a summary DataFrame."""
        records = self.check_pair(doc_a, doc_b, id_col, check_cols)
        if not records:
            return pd.DataFrame(
                columns=[
                    "filer_id", "field", "value_a", "value_b",
                    "absolute_diff", "relative_diff", "severity",
                ]
            )
        return pd.DataFrame([vars(r) for r in records])
