"""Transfer pricing manipulation detection.

Implements reproducible algorithms for identifying non-arm's-length pricing
in related-party transactions, including:

- Comparability analysis using OECD-aligned interquartile range methods
- Profit level indicator screening across multiple financial metrics
- Benford's Law conformity testing on reported financial figures
"""

from open_tax_toolkit.transfer_pricing.comparability import ComparabilityAnalysis
from open_tax_toolkit.transfer_pricing.profit_indicator import ProfitIndicatorScreen
from open_tax_toolkit.transfer_pricing.benford import BenfordAnalysis

__all__ = ["ComparabilityAnalysis", "ProfitIndicatorScreen", "BenfordAnalysis"]
