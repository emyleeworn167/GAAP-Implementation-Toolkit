"""Synthetic data generators for testing and demonstration.

Generates realistic-looking but entirely artificial tax data so that
detection algorithms can be tested without access to confidential returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataGenerator:
    """Generate synthetic datasets for each detection module.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    seed: int = 42

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def comparable_companies(
        self,
        n_comparables: int = 30,
        n_anomalous: int = 3,
        base_margin: float = 0.08,
        margin_std: float = 0.03,
        anomaly_shift: float = 0.15,
    ) -> pd.DataFrame:
        """Generate comparable company financial data for transfer pricing analysis.

        Creates a set of independent comparables with normal profit margins,
        plus a few anomalous entities with shifted margins (simulating
        profit-shifting behavior).
        """
        rng = self._rng()
        n_total = n_comparables + n_anomalous

        revenue = rng.lognormal(mean=18, sigma=1.2, size=n_total)  # ~65M median
        normal_margins = rng.normal(base_margin, margin_std, size=n_comparables)
        anomalous_margins = rng.normal(
            base_margin - anomaly_shift, margin_std * 0.5, size=n_anomalous
        )
        margins = np.concatenate([normal_margins, anomalous_margins])
        operating_profit = revenue * margins
        cogs = revenue * rng.uniform(0.55, 0.75, size=n_total)

        labels = ["comparable"] * n_comparables + ["anomalous"] * n_anomalous
        names = [f"Entity_{i:03d}" for i in range(n_total)]

        return pd.DataFrame(
            {
                "entity": names,
                "revenue": revenue.round(2),
                "cogs": cogs.round(2),
                "operating_profit": operating_profit.round(2),
                "operating_margin": margins.round(4),
                "berry_ratio": ((revenue - cogs + operating_profit) / cogs).round(4),
                "label": labels,
            }
        )

    def tax_filings(
        self,
        n_filers: int = 100,
        inconsistency_rate: float = 0.10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate paired tax filing records with deliberate inconsistencies.

        Returns (form_a, form_b) where form_a is the primary return and
        form_b is a supporting schedule that should be consistent with form_a.
        """
        rng = self._rng()
        n_inconsistent = int(n_filers * inconsistency_rate)

        filer_ids = [f"TIN_{i:06d}" for i in range(n_filers)]
        gross_income = rng.lognormal(mean=11.5, sigma=0.8, size=n_filers)
        deductions = gross_income * rng.uniform(0.15, 0.45, size=n_filers)
        taxable_income = gross_income - deductions

        form_a = pd.DataFrame(
            {
                "filer_id": filer_ids,
                "gross_income": gross_income.round(2),
                "total_deductions": deductions.round(2),
                "taxable_income": taxable_income.round(2),
            }
        )

        # form_b should mirror form_a, but with injected inconsistencies
        form_b = form_a.copy()
        inconsistent_idx = rng.choice(n_filers, size=n_inconsistent, replace=False)
        noise = rng.uniform(0.02, 0.20, size=n_inconsistent)
        form_b.loc[inconsistent_idx, "gross_income"] *= 1 + noise
        form_b["is_inconsistent"] = False
        form_b.loc[inconsistent_idx, "is_inconsistent"] = True

        return form_a, form_b

    def cross_border_transactions(
        self,
        n_entities: int = 50,
        n_transactions: int = 500,
        n_circular: int = 5,
    ) -> pd.DataFrame:
        """Generate cross-border transaction records with embedded circular flows.

        Circular flows (A->B->C->A) are a hallmark of round-tripping fraud.
        """
        rng = self._rng()
        jurisdictions = [
            "US", "UK", "DE", "IE", "NL", "SG", "HK", "KY", "BM", "CH",
            "LU", "JP", "AU", "CA", "BVI", "PA", "MU", "JE", "GG", "LI",
        ]

        entities = []
        for i in range(n_entities):
            entities.append(
                {
                    "entity_id": f"ENT_{i:03d}",
                    "jurisdiction": rng.choice(jurisdictions),
                }
            )
        entity_df = pd.DataFrame(entities)
        entity_ids = entity_df["entity_id"].tolist()

        # Normal transactions
        records = []
        for _ in range(n_transactions - n_circular * 3):
            sender, receiver = rng.choice(entity_ids, size=2, replace=False)
            records.append(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "amount": float(rng.lognormal(10, 1.5)),
                    "is_circular": False,
                }
            )

        # Inject circular flows: A -> B -> C -> A with similar amounts
        for _ in range(n_circular):
            cycle = rng.choice(entity_ids, size=3, replace=False).tolist()
            base_amount = float(rng.lognormal(12, 0.5))  # larger amounts
            for j in range(3):
                records.append(
                    {
                        "sender": cycle[j],
                        "receiver": cycle[(j + 1) % 3],
                        "amount": base_amount * float(rng.uniform(0.95, 1.05)),
                        "is_circular": True,
                    }
                )

        txn_df = pd.DataFrame(records)
        txn_df["amount"] = txn_df["amount"].round(2)

        # Merge jurisdiction info
        txn_df = txn_df.merge(
            entity_df.rename(columns={"entity_id": "sender", "jurisdiction": "sender_jurisdiction"}),
            on="sender",
        ).merge(
            entity_df.rename(
                columns={"entity_id": "receiver", "jurisdiction": "receiver_jurisdiction"}
            ),
            on="receiver",
        )
        return txn_df
