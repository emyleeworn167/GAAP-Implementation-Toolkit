"""Command-line interface for the Open Tax Infrastructure Toolkit."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from open_tax_toolkit import __version__
from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a demonstration of all three detection modules."""
    gen = SyntheticDataGenerator(seed=args.seed)

    print("=" * 70)
    print("Open Tax Infrastructure Toolkit — Demo")
    print(f"Version {__version__}")
    print("=" * 70)

    # --- Transfer Pricing ---
    print("\n[1/3] Transfer Pricing — Comparability Analysis")
    print("-" * 50)
    from open_tax_toolkit.transfer_pricing import ComparabilityAnalysis, BenfordAnalysis

    data = gen.comparable_companies(n_comparables=50, n_anomalous=5)
    comparables = data[data["label"] == "comparable"]
    tested = data[data["label"] == "anomalous"]

    ca = ComparabilityAnalysis(indicator="operating_margin")
    results = ca.screen_batch(comparables, tested, entity_col="entity")
    print(f"  Comparables: {len(comparables)} | Tested parties: {len(tested)}")
    print(f"  Arm's length range: [{results.iloc[0]['q1']:.4f}, {results.iloc[0]['q3']:.4f}]")
    flagged = results[~results["within_range"]]
    print(f"  Flagged (outside range): {len(flagged)} entities")
    if len(flagged) > 0:
        for _, row in flagged.iterrows():
            print(f"    → {row['entity']}: margin={row['tested_value']:.4f}, "
                  f"adjustment={row['adjustment_to_median']:.4f}")

    print("\n  Benford's Law (first digit on revenue):")
    ba = BenfordAnalysis()
    benford = ba.test(data["revenue"])
    print(f"    χ²={benford.chi2_statistic:.2f}, p={benford.p_value:.4f}, "
          f"conforms={benford.conforms}")

    # --- Filing Irregularities ---
    print("\n[2/3] Filing Irregularities — Cross-Document Consistency")
    print("-" * 50)
    from open_tax_toolkit.filing_irregularities import ConsistencyChecker

    form_a, form_b = gen.tax_filings(n_filers=100, inconsistency_rate=0.10)
    checker = ConsistencyChecker()
    discrepancies = checker.check_pair_summary(form_a, form_b)
    actual_inconsistent = form_b["is_inconsistent"].sum()
    print(f"  Filers: {len(form_a)} | Injected inconsistencies: {actual_inconsistent}")
    print(f"  Detected discrepancies: {len(discrepancies)}")
    if len(discrepancies) > 0:
        by_severity = discrepancies["severity"].value_counts().to_dict()
        print(f"    Severity: {by_severity}")

    # --- Cross-Border ---
    print("\n[3/3] Cross-Border Transactions — Network & Risk Analysis")
    print("-" * 50)
    from open_tax_toolkit.cross_border import (
        TransactionNetwork,
        JurisdictionRiskScorer,
        PatternDetector,
    )

    txns = gen.cross_border_transactions(n_entities=50, n_transactions=500, n_circular=5)
    net = TransactionNetwork()
    cycles = net.detect_circular_flows(txns, max_cycle_length=4)
    print(f"  Transactions: {len(txns)} | Entities: {txns['sender'].nunique()}")
    print(f"  Circular flows detected: {len(cycles)}")
    for i, c in enumerate(cycles[:3]):
        print(f"    → Cycle {i+1}: {' → '.join(c.entities)} → {c.entities[0]} "
              f"(${c.total_amount:,.0f}, variance={c.amount_variance:.4f})")

    scorer = JurisdictionRiskScorer()
    scored = scorer.score_transactions(txns)
    high_risk = scored[scored["transaction_risk"] >= 70]
    print(f"  High-risk transactions (score ≥ 70): {len(high_risk)}")

    detector = PatternDetector()
    pair_anomalies = detector.entity_pair_anomalies(txns)
    n_anomalous_pairs = pair_anomalies["is_anomalous"].sum()
    print(f"  Anomalous entity pairs (Isolation Forest): {n_anomalous_pairs}")

    print("\n" + "=" * 70)
    print("Demo complete. See examples/quickstart.py for programmatic usage.")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tax-toolkit",
        description="Open Tax Infrastructure Toolkit — tax compliance anomaly detection",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run detection demo with synthetic data")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)
    if args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
