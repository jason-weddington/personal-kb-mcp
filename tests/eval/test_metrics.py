"""Unit tests for IR metrics."""

import math

import pytest

from tests.eval.metrics import (
    correct_rejection_rate,
    evaluate_query_set,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert reciprocal_rank(["a"], ["a", "b", "c"]) == 1.0

    def test_second_result_relevant(self):
        assert reciprocal_rank(["b"], ["a", "b", "c"]) == 0.5

    def test_third_result_relevant(self):
        assert reciprocal_rank(["c"], ["a", "b", "c"]) == pytest.approx(1.0 / 3)

    def test_no_relevant_results(self):
        assert reciprocal_rank(["x"], ["a", "b", "c"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert reciprocal_rank(["b", "c"], ["a", "b", "c"]) == 0.5

    def test_empty_results(self):
        assert reciprocal_rank(["a"], []) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank([], ["a", "b"]) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0

    def test_partial_found(self):
        assert recall_at_k(["a", "b", "c"], ["a", "x", "b"], k=3) == pytest.approx(2.0 / 3)

    def test_none_found(self):
        assert recall_at_k(["a"], ["x", "y", "z"], k=3) == 0.0

    def test_k_truncates(self):
        assert recall_at_k(["c"], ["a", "b", "c"], k=2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k([], ["a", "b"], k=5) == 0.0

    def test_empty_results(self):
        assert recall_at_k(["a"], [], k=5) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        # Two relevant items at positions 2 and 1 vs ideal 1 and 2
        result = ndcg_at_k(["a", "b"], ["b", "a", "c"], k=3)
        # Still 1.0 — binary relevance, same DCG regardless of order among relevant
        assert result == pytest.approx(1.0)

    def test_relevant_at_end(self):
        # One relevant item at position 3 (0-indexed: 2)
        result = ndcg_at_k(["c"], ["a", "b", "c"], k=3)
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert result == pytest.approx(expected)

    def test_no_relevant_results(self):
        assert ndcg_at_k(["x"], ["a", "b", "c"], k=3) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k([], ["a", "b"], k=5) == 0.0

    def test_k_truncates(self):
        assert ndcg_at_k(["c"], ["a", "b", "c"], k=2) == 0.0


class TestEvaluateQuerySet:
    def test_aggregates_correctly(self):
        queries = [
            {"id": "q1", "expected": ["a"]},
            {"id": "q2", "expected": ["b"]},
        ]
        results_map = {
            "q1": ["a", "x"],  # MRR=1.0, recall@2=1.0
            "q2": ["x", "b"],  # MRR=0.5, recall@2=1.0
        }
        agg = evaluate_query_set(queries, results_map, k=2)
        assert agg["mean_mrr"] == pytest.approx(0.75)
        assert agg["mean_recall_at_k"] == pytest.approx(1.0)

    def test_missing_results(self):
        queries = [{"id": "q1", "expected": ["a"]}]
        agg = evaluate_query_set(queries, {}, k=5)
        assert agg["mean_mrr"] == 0.0
        assert agg["mean_recall_at_k"] == 0.0
        assert agg["mean_ndcg_at_k"] == 0.0


class TestCorrectRejectionRate:
    """Unit tests for the abstention metric."""

    def _abs_query(self, qid: str, include_stale: bool = False) -> dict:
        """Helper: build a query dict that counts as an abstention query."""
        return {
            "id": qid,
            "category": "abstention",
            "expected": [],
            "include_stale": include_stale,
        }

    def test_all_empty_results_rate_one(self):
        """All abstention queries return empty → rate 1.0."""
        queries = [self._abs_query("q1"), self._abs_query("q2")]
        results_map: dict[str, list] = {"q1": [], "q2": []}
        assert correct_rejection_rate(queries, results_map) == 1.0

    def test_any_non_empty_result_counts_zero(self):
        """A query with non-empty results scores 0 for that query."""
        queries = [self._abs_query("q1")]
        results_map = {"q1": ["some-entry"]}
        assert correct_rejection_rate(queries, results_map) == 0.0

    def test_mixed_results_fractional(self):
        """One correctly rejected, one not → 0.5."""
        queries = [self._abs_query("q1"), self._abs_query("q2")]
        results_map = {"q1": [], "q2": ["leaked-entry"]}
        assert correct_rejection_rate(queries, results_map) == pytest.approx(0.5)

    def test_no_abstention_queries_returns_one(self):
        """Denominator 0 → graceful 1.0."""
        queries = [{"id": "q1", "category": "temporal", "expected": ["a"], "include_stale": False}]
        results_map = {"q1": []}
        assert correct_rejection_rate(queries, results_map) == 1.0

    def test_include_stale_true_excluded_from_denominator(self):
        """Queries with include_stale=True are recall controls, not abstention cases."""
        queries = [
            self._abs_query("q-abs", include_stale=False),  # counted
            self._abs_query("q-ctrl", include_stale=True),  # excluded from denominator
        ]
        results_map = {"q-abs": [], "q-ctrl": ["some-entry"]}
        # Only q-abs is in denominator; it passes → 1.0
        assert correct_rejection_rate(queries, results_map) == 1.0

    def test_missing_from_results_map_treated_as_empty(self):
        """A query absent from results_map defaults to empty results."""
        queries = [self._abs_query("q1")]
        assert correct_rejection_rate(queries, {}) == 1.0

    def test_score_threshold_with_tuple_results(self):
        """score_threshold: top score below threshold counts as rejected."""
        queries = [self._abs_query("q1")]
        results_map = {"q1": [("some-entry", 0.02)]}
        # Top score 0.02 < threshold 0.03 → correctly rejected
        assert correct_rejection_rate(queries, results_map, score_threshold=0.03) == 1.0
        # Top score 0.02 >= threshold 0.01 → not rejected
        assert correct_rejection_rate(queries, results_map, score_threshold=0.01) == 0.0
