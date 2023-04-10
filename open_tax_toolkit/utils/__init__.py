"""Shared utilities for the Open Tax Toolkit."""

from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator
from open_tax_toolkit.utils.stats import iqr_range, chi_squared_test, z_score_outliers

__all__ = [
    "SyntheticDataGenerator",
    "iqr_range",
    "chi_squared_test",
    "z_score_outliers",
]
