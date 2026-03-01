"""Unit tests for IR metrics."""

import math

import pytest

from tests.eval.metrics import evaluate_query_set, ndcg_at_k, recall_at_k, reciprocal_rank


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
