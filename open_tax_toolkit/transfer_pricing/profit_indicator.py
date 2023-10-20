"""Profit level indicator (PLI) screening for transfer pricing anomalies.

Computes standard PLIs used in TNMM / CPM analyses and flags entities
whose indicators deviate significantly from industry norms. PLIs include:

- Operating margin (OM): Operating profit / Revenue
- Berry ratio (BR): Gross profit / Operating expenses
- Net cost plus (NCP): Operating profit / Total costs
- Return on assets (ROA): Operating profit / Operating assets
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from open_tax_toolkit.utils.stats import z_score_outliers


PLI_DEFINITIONS = {
    "operating_margin": ("operating_profit", "revenue"),
    "berry_ratio": ("gross_profit", "operating_expenses"),
    "net_cost_plus": ("operating_profit", "total_costs"),
    "return_on_assets": ("operating_profit", "operating_assets"),
}


@dataclass
class ProfitIndicatorScreen:
    """Screen entities for anomalous profit level indicators.

    Parameters
    ----------
    indicators : list[str] | None
        Which PLIs to compute. Defaults to all available given input columns.
    z_threshold : float
        Modified Z-score threshold for flagging anomalies.
    """

    indicators: list[str] | None = None
    z_threshold: float = 2.5

    def compute_plis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all applicable PLIs and add them as columns.

        Only computes PLIs whose required columns exist in the DataFrame.
        """
        result = df.copy()
        targets = self.indicators or list(PLI_DEFINITIONS.keys())

        for pli_name in targets:
            if pli_name not in PLI_DEFINITIONS:
                continue
            numerator_col, denominator_col = PLI_DEFINITIONS[pli_name]
            if numerator_col in df.columns and denominator_col in df.columns:
                denom = df[denominator_col].replace(0, np.nan)
                result[pli_name] = (df[numerator_col] / denom).round(6)

        return result

    def screen(self, df: pd.DataFrame, entity_col: str = "entity") -> pd.DataFrame:
        """Screen all entities and flag those with anomalous PLIs.

        Returns a DataFrame with entity, each computed PLI, and boolean
        flag columns (*_anomaly) indicating statistical outliers.
        """
        enriched = self.compute_plis(df)
        available_plis = [
            col for col in PLI_DEFINITIONS if col in enriched.columns
        ]

        flags = {}
        for pli in available_plis:
            values = enriched[pli].dropna().values
            if len(values) < 5:
                continue
            outlier_mask = z_score_outliers(values, threshold=self.z_threshold)
            # Map back to full DataFrame (NaN positions are not outliers)
            full_mask = np.zeros(len(enriched), dtype=bool)
            non_null_idx = enriched[pli].dropna().index
            full_mask[non_null_idx] = outlier_mask
            flags[f"{pli}_anomaly"] = full_mask

        flag_df = pd.DataFrame(flags, index=enriched.index)
        flag_df["any_anomaly"] = flag_df.any(axis=1)

        cols = [entity_col] + available_plis
        return pd.concat([enriched[cols], flag_df], axis=1)
