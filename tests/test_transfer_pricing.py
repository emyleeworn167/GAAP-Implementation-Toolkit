"""Tests for transfer pricing detection module."""

import numpy as np
import pandas as pd
import pytest

from open_tax_toolkit.transfer_pricing import (
    BenfordAnalysis,
    ComparabilityAnalysis,
    ProfitIndicatorScreen,
)
from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator


@pytest.fixture
def gen():
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def comparable_data(gen):
    return gen.comparable_companies(n_comparables=30, n_anomalous=3)


class TestComparabilityAnalysis:
    def test_within_range_detection(self, comparable_data):
        comparables = comparable_data[comparable_data["label"] == "comparable"]
        ca = ComparabilityAnalysis(indicator="operating_margin")

        # Median of comparables should be within its own range
        median_val = comparables["operating_margin"].median()
        result = ca.analyze(comparables, median_val)
        assert result.is_within_range == True
        assert result.adjustment_to_median == 0.0

    def test_outside_range_detection(self, comparable_data):
        comparables = comparable_data[comparable_data["label"] == "comparable"]
        ca = ComparabilityAnalysis(indicator="operating_margin")

        # An extreme value should be outside the range
        result = ca.analyze(comparables, -0.50)
        assert result.is_within_range is False
        assert result.adjustment_to_median != 0.0

    def test_screen_batch(self, comparable_data):
        comparables = comparable_data[comparable_data["label"] == "comparable"]
        tested = comparable_data[comparable_data["label"] == "anomalous"]
        ca = ComparabilityAnalysis(indicator="operating_margin")

        results = ca.screen_batch(comparables, tested)
        assert len(results) == len(tested)
        assert "within_range" in results.columns

    def test_outlier_removal(self, comparable_data):
        comparables = comparable_data[comparable_data["label"] == "comparable"]
        ca_with = ComparabilityAnalysis(remove_outliers=True)
        ca_without = ComparabilityAnalysis(remove_outliers=False)

        r1 = ca_with.analyze(comparables, 0.08)
        r2 = ca_without.analyze(comparables, 0.08)
        # Outlier removal should reduce comparable count
        assert r1.comparable_count <= r2.comparable_count


class TestBenfordAnalysis:
    def test_benford_conforming_data(self):
        """Naturally generated lognormal data should conform to Benford's Law."""
        rng = np.random.default_rng(42)
        values = rng.lognormal(mean=10, sigma=2, size=1000)

        ba = BenfordAnalysis(significance_level=0.05)
        result = ba.test(values, digit_position=1)
        assert result.conforms is True
        assert result.n_values == 1000

    def test_benford_uniform_fails(self):
        """Uniformly distributed data should NOT conform to Benford's Law."""
        rng = np.random.default_rng(42)
        values = rng.uniform(100, 999, size=1000)

        ba = BenfordAnalysis(significance_level=0.05)
        result = ba.test(values, digit_position=1)
        assert result.conforms is False

    def test_min_sample_size(self):
        ba = BenfordAnalysis(min_sample_size=50)
        with pytest.raises(ValueError, match="at least 50"):
            ba.test(np.array([1.0, 2.0, 3.0]))

    def test_dataframe_analysis(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"revenue": rng.lognormal(mean=10, sigma=2, size=200)})
        ba = BenfordAnalysis()
        results = ba.test_dataframe(df, columns=["revenue"])
        assert len(results) == 1
        assert "conforms" in results.columns


class TestProfitIndicatorScreen:
    def test_compute_plis(self, comparable_data):
        screen = ProfitIndicatorScreen()
        result = screen.compute_plis(comparable_data)
        assert "operating_margin" in result.columns

    def test_screen_flags_anomalies(self, comparable_data):
        screen = ProfitIndicatorScreen(z_threshold=2.0)
        result = screen.screen(comparable_data, entity_col="entity")
        assert "any_anomaly" in result.columns
        # Should flag at least some of the anomalous entities
        assert result["any_anomaly"].sum() > 0
