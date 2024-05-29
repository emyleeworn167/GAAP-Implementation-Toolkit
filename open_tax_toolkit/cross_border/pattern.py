"""Statistical pattern detection for cross-border transaction anomalies.

Identifies anomalous transaction patterns using unsupervised methods:
- Amount distribution outliers per entity pair
- Temporal clustering (burst detection)
- Volume-weighted anomaly scoring combining multiple signals
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from open_tax_toolkit.utils.stats import z_score_outliers


@dataclass
class PatternDetector:
    """Detect anomalous patterns in cross-border transaction data.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies for Isolation Forest (0.0 to 0.5).
    z_threshold : float
        Z-score threshold for amount outlier detection.
    random_state : int
        Random seed for reproducibility.
    """

    contamination: float = 0.05
    z_threshold: float = 2.5
    random_state: int = 42

    def amount_outliers(
        self,
        transactions: pd.DataFrame,
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Flag transactions with statistically anomalous amounts.

        Uses MAD-based modified Z-scores for robustness against
        the heavy-tailed distributions typical in financial data.
        """
        result = transactions.copy()
        values = result[amount_col].values
        result["amount_outlier"] = z_score_outliers(values, threshold=self.z_threshold)
        return result

    def entity_pair_anomalies(
        self,
        transactions: pd.DataFrame,
        sender_col: str = "sender",
        receiver_col: str = "receiver",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Detect anomalous entity pairs based on transaction features.

        Aggregates transaction features per entity pair and applies
        Isolation Forest to identify pairs with unusual characteristics:
        - Transaction count
        - Total amount
        - Mean amount
        - Amount standard deviation
        - Max/min ratio (consistency measure)
        """
        pairs = (
            transactions.groupby([sender_col, receiver_col])
            .agg(
                txn_count=(amount_col, "count"),
                total_amount=(amount_col, "sum"),
                mean_amount=(amount_col, "mean"),
                std_amount=(amount_col, "std"),
                max_amount=(amount_col, "max"),
                min_amount=(amount_col, "min"),
            )
            .reset_index()
        )

        pairs["std_amount"] = pairs["std_amount"].fillna(0)
        pairs["max_min_ratio"] = pairs["max_amount"] / pairs["min_amount"].clip(lower=1e-10)

        feature_cols = [
            "txn_count", "total_amount", "mean_amount", "std_amount", "max_min_ratio"
        ]
        features = pairs[feature_cols].values

        # Log-transform skewed features
        features = np.log1p(np.abs(features))

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        pairs["anomaly_label"] = iso.fit_predict(features)
        pairs["anomaly_score"] = -iso.score_samples(features)
        pairs["is_anomalous"] = pairs["anomaly_label"] == -1

        return pairs.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    def composite_risk_score(
        self,
        transactions: pd.DataFrame,
        amount_col: str = "amount",
        risk_col: str = "transaction_risk",
    ) -> pd.DataFrame:
        """Compute a composite anomaly score combining amount and jurisdiction risk.

        Multiplies normalized amount Z-scores by jurisdiction risk to produce
        a single risk-weighted anomaly indicator.
        """
        result = transactions.copy()
        amounts = result[amount_col].values
        median = np.median(amounts)
        mad = np.median(np.abs(amounts - median))
        z_scores = np.abs(0.6745 * (amounts - median) / max(mad, 1e-10))

        # Normalize risk to [0, 1]
        if risk_col in result.columns:
            risk_norm = result[risk_col].values / 100.0
        else:
            risk_norm = np.ones(len(result)) * 0.5

        result["amount_z_score"] = np.round(z_scores, 4)
        result["composite_score"] = np.round(z_scores * risk_norm, 4)
        result = result.sort_values("composite_score", ascending=False)
        return result.reset_index(drop=True)
