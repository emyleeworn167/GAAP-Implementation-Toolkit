"""Cross-border transaction fraud detection.

Analyzes e-commerce and digital payment flows to identify patterns that
obscure taxable income, including:

- Transaction network analysis for circular flows and layering
- Jurisdiction-level risk scoring based on transparency indices
- Statistical anomaly detection in cross-border payment patterns
"""

from open_tax_toolkit.cross_border.network import TransactionNetwork
from open_tax_toolkit.cross_border.risk_scoring import JurisdictionRiskScorer
from open_tax_toolkit.cross_border.pattern import PatternDetector

__all__ = ["TransactionNetwork", "JurisdictionRiskScorer", "PatternDetector"]
