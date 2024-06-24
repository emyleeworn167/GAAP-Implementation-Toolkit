"""Jurisdiction risk scoring for cross-border tax analysis.

Assigns risk scores to jurisdictions based on tax transparency, treaty
network coverage, and known characteristics of tax haven jurisdictions.
Scores are used to weight transaction anomalies — a suspicious pattern
involving a high-risk jurisdiction warrants more scrutiny.

Risk factors are derived from:
- OECD Global Forum peer review ratings
- EU list of non-cooperative jurisdictions
- Financial Secrecy Index (Tax Justice Network)
- Effective corporate tax rates
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# Simplified risk scores for MVP (0-100, higher = riskier)
# Based on composite of transparency indices and tax haven lists
DEFAULT_RISK_SCORES: dict[str, float] = {
    # Low risk (major transparent economies)
    "US": 15, "UK": 18, "DE": 12, "FR": 14, "JP": 10, "AU": 12, "CA": 13,
    "SE": 8, "NO": 8, "DK": 9, "FI": 9, "NZ": 10, "KR": 16,
    # Medium risk (financial centers / favorable regimes)
    "IE": 45, "NL": 42, "SG": 40, "HK": 48, "CH": 50, "LU": 52,
    "BE": 30, "AT": 25, "MT": 55, "CY": 58,
    # High risk (secrecy jurisdictions / tax havens)
    "KY": 80, "BM": 78, "BVI": 90, "PA": 75, "MU": 70,
    "JE": 65, "GG": 65, "LI": 68, "MC": 62, "GI": 60,
    "VG": 88, "AI": 82, "TC": 85, "SC": 72, "VU": 78,
}

# Unknown jurisdictions default to medium-high risk
DEFAULT_UNKNOWN_RISK = 55.0


@dataclass
class JurisdictionRiskScorer:
    """Score jurisdictions and transactions by tax compliance risk.

    Parameters
    ----------
    risk_scores : dict[str, float] | None
        Jurisdiction ISO code -> risk score (0-100). Uses built-in defaults if None.
    unknown_risk : float
        Default risk score for jurisdictions not in the lookup table.
    """

    risk_scores: dict[str, float] | None = None
    unknown_risk: float = DEFAULT_UNKNOWN_RISK
    _scores: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self):
        self._scores = self.risk_scores or DEFAULT_RISK_SCORES.copy()

    def score_jurisdiction(self, jurisdiction: str) -> float:
        """Return the risk score for a single jurisdiction."""
        return self._scores.get(jurisdiction.upper(), self.unknown_risk)

    def score_transaction(
        self,
        sender_jurisdiction: str,
        receiver_jurisdiction: str,
    ) -> float:
        """Score a single transaction based on jurisdiction pair risk.

        Uses the maximum of the two jurisdictions' scores, with a bonus
        when both are high-risk (compounding secrecy).
        """
        s1 = self.score_jurisdiction(sender_jurisdiction)
        s2 = self.score_jurisdiction(receiver_jurisdiction)
        base = max(s1, s2)
        # Both high-risk adds 10% compounding factor
        if s1 >= 60 and s2 >= 60:
            base = min(base * 1.1, 100.0)
        return round(base, 2)

    def score_transactions(
        self,
        df: pd.DataFrame,
        sender_juris_col: str = "sender_jurisdiction",
        receiver_juris_col: str = "receiver_jurisdiction",
    ) -> pd.DataFrame:
        """Add jurisdiction risk scores to a transaction DataFrame.

        Adds columns: sender_risk, receiver_risk, transaction_risk.
        """
        result = df.copy()
        result["sender_risk"] = result[sender_juris_col].map(
            lambda j: self.score_jurisdiction(j)
        )
        result["receiver_risk"] = result[receiver_juris_col].map(
            lambda j: self.score_jurisdiction(j)
        )
        result["transaction_risk"] = result.apply(
            lambda row: self.score_transaction(
                row[sender_juris_col], row[receiver_juris_col]
            ),
            axis=1,
        )
        return result

    def jurisdiction_summary(self) -> pd.DataFrame:
        """Return all jurisdiction risk scores as a sorted DataFrame."""
        rows = [
            {"jurisdiction": k, "risk_score": v}
            for k, v in sorted(self._scores.items(), key=lambda x: -x[1])
        ]
        return pd.DataFrame(rows)
