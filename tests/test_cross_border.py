"""Tests for cross-border transaction fraud detection module."""

import pytest

from open_tax_toolkit.cross_border import (
    JurisdictionRiskScorer,
    PatternDetector,
    TransactionNetwork,
)
from open_tax_toolkit.utils.data_gen import SyntheticDataGenerator


@pytest.fixture
def gen():
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def txn_data(gen):
    return gen.cross_border_transactions(n_entities=30, n_transactions=200, n_circular=3)


class TestTransactionNetwork:
    def test_build_graph(self, txn_data):
        net = TransactionNetwork()
        g = net.build_graph(txn_data)
        assert g.number_of_nodes() > 0
        assert g.number_of_edges() > 0

    def test_detect_circular_flows(self, txn_data):
        net = TransactionNetwork()
        cycles = net.detect_circular_flows(txn_data, max_cycle_length=4)
        # Should detect at least some of the injected circular flows
        assert len(cycles) > 0
        # Each cycle should have the right structure
        for c in cycles:
            assert c.length >= 2
            assert c.total_amount > 0

    def test_hub_analysis(self, txn_data):
        net = TransactionNetwork()
        hubs = net.hub_analysis(txn_data, top_n=5)
        assert len(hubs) <= 5
        assert "betweenness_centrality" in hubs.columns
        assert "entity" in hubs.columns


class TestJurisdictionRiskScorer:
    def test_known_jurisdiction(self):
        scorer = JurisdictionRiskScorer()
        assert scorer.score_jurisdiction("US") == 15
        assert scorer.score_jurisdiction("BVI") == 90

    def test_unknown_jurisdiction(self):
        scorer = JurisdictionRiskScorer()
        score = scorer.score_jurisdiction("XX")
        assert score == 55.0  # default unknown risk

    def test_transaction_scoring(self):
        scorer = JurisdictionRiskScorer()
        # US to US: low risk
        low = scorer.score_transaction("US", "US")
        # US to BVI: high risk
        high = scorer.score_transaction("US", "BVI")
        assert high > low

    def test_compounding_risk(self):
        scorer = JurisdictionRiskScorer()
        # Both high-risk: BVI to KY should get compounding bonus
        score = scorer.score_transaction("BVI", "KY")
        assert score > 90  # Base max(90,80)=90, with 10% bonus = 99

    def test_score_transactions_dataframe(self, txn_data):
        scorer = JurisdictionRiskScorer()
        result = scorer.score_transactions(txn_data)
        assert "transaction_risk" in result.columns
        assert "sender_risk" in result.columns
        assert "receiver_risk" in result.columns

    def test_jurisdiction_summary(self):
        scorer = JurisdictionRiskScorer()
        summary = scorer.jurisdiction_summary()
        assert len(summary) > 0
        # Should be sorted descending by risk
        assert summary.iloc[0]["risk_score"] >= summary.iloc[-1]["risk_score"]


class TestPatternDetector:
    def test_amount_outliers(self, txn_data):
        detector = PatternDetector()
        result = detector.amount_outliers(txn_data)
        assert "amount_outlier" in result.columns
        # Should flag some but not all
        assert 0 < result["amount_outlier"].sum() < len(result)

    def test_entity_pair_anomalies(self, txn_data):
        detector = PatternDetector(contamination=0.10)
        result = detector.entity_pair_anomalies(txn_data)
        assert "is_anomalous" in result.columns
        assert "anomaly_score" in result.columns
        assert result["is_anomalous"].sum() > 0

    def test_composite_risk_score(self, txn_data):
        scorer = JurisdictionRiskScorer()
        scored = scorer.score_transactions(txn_data)
        detector = PatternDetector()
        result = detector.composite_risk_score(scored)
        assert "composite_score" in result.columns
        # Should be sorted descending
        scores = result["composite_score"].values
        assert scores[0] >= scores[-1]
