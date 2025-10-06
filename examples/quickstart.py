#!/usr/bin/env python3
"""Quickstart example for the Open Tax Infrastructure Toolkit.

Demonstrates all three detection modules using synthetic data.
Run with: python examples/quickstart.py
"""

from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator

# ── Initialize synthetic data generator ────────────────────────────
gen = SyntheticDataGenerator(seed=42)


# ── 1. Transfer Pricing: Comparability Analysis ───────────────────
print("=" * 60)
print("1. Transfer Pricing — Comparability Analysis")
print("=" * 60)

from open_tax_toolkit.transfer_pricing import ComparabilityAnalysis

# Generate comparable company data (30 normal + 3 with shifted margins)
data = gen.comparable_companies(n_comparables=30, n_anomalous=3)
comparables = data[data["label"] == "comparable"]
tested_parties = data[data["label"] == "anomalous"]

# Run OECD-aligned IQR analysis
analyzer = ComparabilityAnalysis(indicator="operating_margin")
results = analyzer.screen_batch(comparables, tested_parties)
print(results.to_string(index=False))
print()


# ── 2. Transfer Pricing: Benford's Law ────────────────────────────
print("=" * 60)
print("2. Transfer Pricing — Benford's Law Analysis")
print("=" * 60)

from open_tax_toolkit.transfer_pricing import BenfordAnalysis

benford = BenfordAnalysis()
result = benford.test(data["revenue"], digit_position=1)
print(f"Chi-squared: {result.chi2_statistic}")
print(f"P-value: {result.p_value}")
print(f"Conforms to Benford's Law: {result.conforms}")
print()


# ── 3. Filing Irregularities: Consistency Check ───────────────────
print("=" * 60)
print("3. Filing Irregularities — Cross-Document Consistency")
print("=" * 60)

from open_tax_toolkit.filing_irregularities import ConsistencyChecker

form_a, form_b = gen.tax_filings(n_filers=100, inconsistency_rate=0.10)
checker = ConsistencyChecker()
discrepancies = checker.check_pair_summary(form_a, form_b)
print(f"Detected {len(discrepancies)} discrepancies:")
if len(discrepancies) > 0:
    print(discrepancies.head(10).to_string(index=False))
print()


# ── 4. Cross-Border: Circular Flow Detection ─────────────────────
print("=" * 60)
print("4. Cross-Border — Circular Flow Detection")
print("=" * 60)

from open_tax_toolkit.cross_border import TransactionNetwork

txns = gen.cross_border_transactions(n_entities=50, n_transactions=500, n_circular=5)
network = TransactionNetwork()
cycles = network.detect_circular_flows(txns, max_cycle_length=4)
print(f"Detected {len(cycles)} circular flows:")
for c in cycles[:5]:
    path = " → ".join(c.entities) + f" → {c.entities[0]}"
    print(f"  {path}  (${c.total_amount:,.0f}, var={c.amount_variance:.4f})")
print()


# ── 5. Cross-Border: Risk Scoring ────────────────────────────────
print("=" * 60)
print("5. Cross-Border — Jurisdiction Risk Scoring")
print("=" * 60)

from open_tax_toolkit.cross_border import JurisdictionRiskScorer, PatternDetector

scorer = JurisdictionRiskScorer()
scored_txns = scorer.score_transactions(txns)
high_risk = scored_txns[scored_txns["transaction_risk"] >= 70]
print(f"High-risk transactions: {len(high_risk)} / {len(scored_txns)}")
print(high_risk[["sender", "receiver", "amount", "transaction_risk"]].head(10).to_string(index=False))
print()


# ── 6. Cross-Border: Composite Anomaly Scoring ───────────────────
print("=" * 60)
print("6. Cross-Border — Composite Anomaly Scoring")
print("=" * 60)

detector = PatternDetector()
composite = detector.composite_risk_score(scored_txns)
print("Top 10 highest-risk transactions:")
print(composite[["sender", "receiver", "amount", "transaction_risk", "composite_score"]].head(10).to_string(index=False))
