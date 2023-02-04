# Open Tax Infrastructure Toolkit

Computational methods for detecting tax compliance anomalies — reproducible detection algorithms for researchers, enforcement practitioners, and compliance professionals.

**Author:** Jiayan Fan

## Overview

This toolkit implements peer-reviewed detection algorithms across three technically distinct anomaly categories:

1. **Transfer Pricing Manipulation** — Detects non-arm's-length pricing in related-party transactions using OECD-aligned comparability analysis, profit level indicator screening, and Benford's Law conformity testing.

2. **Tax Filing Irregularities** — Identifies inconsistencies across filed returns, financial statements, and supporting documentation through cross-document consistency checks and numerical pattern analysis.

3. **Cross-Border Transaction Fraud** — Analyzes e-commerce and digital payment flows using transaction network analysis, jurisdiction risk scoring, and unsupervised anomaly detection to surface patterns that obscure taxable income.

Each module corresponds to methodologies developed from three years of professional transfer pricing work at Ernst & Young and Deloitte — including direct TP defense for multinational clients, value chain analyses, and Advance Pricing Agreement negotiations — and documented in nine peer-reviewed publications cited 210+ times across 74 U.S. cities and 47 countries.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the built-in demo with synthetic data
tax-toolkit demo

# Or use Python directly
python examples/quickstart.py
```

```python
from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator
from open_tax_toolkit.transfer_pricing import ComparabilityAnalysis

gen = SyntheticDataGenerator(seed=42)
data = gen.comparable_companies(n_comparables=30, n_anomalous=3)

analyzer = ComparabilityAnalysis(indicator="operating_margin")
result = analyzer.analyze(
    comparables=data[data["label"] == "comparable"],
    tested_party_value=-0.05,
)
print(f"Within arm's length range: {result.is_within_range}")
print(f"Range: [{result.q1:.4f}, {result.q3:.4f}]")
```

## Modules

### Transfer Pricing (`open_tax_toolkit.transfer_pricing`)

- **ComparabilityAnalysis** — OECD interquartile range method for arm's length testing
- **ProfitIndicatorScreen** — Multi-PLI screening (operating margin, Berry ratio, net cost plus, ROA)
- **BenfordAnalysis** — First/second digit distribution testing with chi-squared goodness-of-fit

### Filing Irregularities (`open_tax_toolkit.filing_irregularities`)

- **ConsistencyChecker** — Cross-document numerical consistency with severity classification
- **NumericalAnalyzer** — Sum validation, rounding pattern detection, last-two-digits uniformity test

### Cross-Border Transactions (`open_tax_toolkit.cross_border`)

- **TransactionNetwork** — Graph-based circular flow detection and hub entity analysis
- **JurisdictionRiskScorer** — Composite risk scoring based on transparency indices
- **PatternDetector** — Isolation Forest anomaly detection with composite risk weighting

## Testing

```bash
pytest
```

All tests use synthetic data generators — no real tax data is required or included.

## Project Structure

```
open_tax_toolkit/
├── transfer_pricing/      # Module 1: TP manipulation detection
│   ├── comparability.py   # IQR-based arm's length analysis
│   ├── profit_indicator.py # PLI screening
│   └── benford.py         # Benford's Law testing
├── filing_irregularities/ # Module 2: Filing inconsistency detection
│   ├── consistency.py     # Cross-document checks
│   └── numerical.py       # Numerical pattern analysis
├── cross_border/          # Module 3: Cross-border fraud detection
│   ├── network.py         # Transaction graph analysis
│   ├── risk_scoring.py    # Jurisdiction risk scoring
│   └── pattern.py         # Unsupervised anomaly detection
├── utils/
│   ├── data_gen.py        # Synthetic data generators
│   └── stats.py           # Shared statistical utilities
└── cli.py                 # Command-line interface
```

## License

MIT License. See [LICENSE](LICENSE) for details.
