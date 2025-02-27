"""Tests for filing irregularities detection module."""

import numpy as np
import pandas as pd
import pytest

from open_tax_toolkit.filing_irregularities import ConsistencyChecker, NumericalAnalyzer
from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator


@pytest.fixture
def gen():
    return SyntheticDataGenerator(seed=42)


class TestConsistencyChecker:
    def test_detects_discrepancies(self, gen):
        form_a, form_b = gen.tax_filings(n_filers=100, inconsistency_rate=0.10)
        checker = ConsistencyChecker(relative_threshold=0.01)

        discrepancies = checker.check_pair(form_a, form_b)
        assert len(discrepancies) > 0

        # Should detect at least some of the 10 injected inconsistencies
        detected_filers = {d.filer_id for d in discrepancies}
        actual_inconsistent = set(form_b[form_b["is_inconsistent"]]["filer_id"])
        overlap = detected_filers & actual_inconsistent
        assert len(overlap) > 0

    def test_no_false_positives_on_identical_data(self):
        df = pd.DataFrame({
            "filer_id": ["A", "B", "C"],
            "gross_income": [100000.0, 200000.0, 300000.0],
            "deductions": [10000.0, 20000.0, 30000.0],
        })
        checker = ConsistencyChecker()
        discrepancies = checker.check_pair(df, df.copy())
        assert len(discrepancies) == 0

    def test_severity_classification(self):
        checker = ConsistencyChecker(
            relative_threshold=0.001,
            absolute_threshold=0.0,
            severity_thresholds=(0.05, 0.15),
        )
        df_a = pd.DataFrame({"filer_id": ["X"], "income": [100000.0]})
        df_b = pd.DataFrame({"filer_id": ["X"], "income": [120000.0]})  # 20% diff

        records = checker.check_pair(df_a, df_b)
        assert len(records) == 1
        assert records[0].severity == "high"

    def test_summary_returns_dataframe(self, gen):
        form_a, form_b = gen.tax_filings()
        checker = ConsistencyChecker()
        result = checker.check_pair_summary(form_a, form_b)
        assert isinstance(result, pd.DataFrame)
        assert "severity" in result.columns


class TestNumericalAnalyzer:
    def test_validate_sums_passes_correct_data(self):
        df = pd.DataFrame({
            "filer_id": ["A", "B"],
            "total": [300.0, 600.0],
            "part1": [100.0, 200.0],
            "part2": [200.0, 400.0],
        })
        analyzer = NumericalAnalyzer()
        results = analyzer.validate_sums(df, "total", ["part1", "part2"])
        assert all(r.passes for r in results)

    def test_validate_sums_catches_mismatch(self):
        df = pd.DataFrame({
            "filer_id": ["A"],
            "total": [500.0],  # Should be 300
            "part1": [100.0],
            "part2": [200.0],
        })
        analyzer = NumericalAnalyzer()
        results = analyzer.validate_sums(df, "total", ["part1", "part2"])
        assert not results[0].passes
        assert results[0].difference == 200.0

    def test_rounding_detection(self):
        df = pd.DataFrame({
            "amounts": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0,
                        60000.0, 70000.0, 80000.0, 90000.0, 100000.0],
        })
        analyzer = NumericalAnalyzer(rounding_threshold=0.5)
        result = analyzer.detect_rounding_patterns(df)
        assert result.iloc[0]["suspicious"] == True

    def test_last_two_digits(self):
        rng = np.random.default_rng(42)
        # Lognormal data should have approximately uniform last two digits
        values = rng.lognormal(mean=12, sigma=1, size=2000)
        analyzer = NumericalAnalyzer()
        result = analyzer.last_two_digits_test(values)
        assert "chi2" in result
        assert "p_value" in result
        assert result["n_values"] > 0
