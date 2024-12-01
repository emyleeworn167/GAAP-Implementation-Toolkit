"""Tax filing irregularity detection.

Identifies inconsistencies across filed returns, financial statements, and
supporting documentation using:

- Cross-document numerical consistency checks
- Mathematical relationship validation (sums, subtotals, ratios)
- Rounding pattern analysis and digit distribution tests
"""

from open_tax_toolkit.filing_irregularities.consistency import ConsistencyChecker
from open_tax_toolkit.filing_irregularities.numerical import NumericalAnalyzer

__all__ = ["ConsistencyChecker", "NumericalAnalyzer"]
