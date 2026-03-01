"""Baseline snapshot: run all golden queries, compute metrics, write JSON.

The output file (baseline.json) is checked into git. After making ranking
changes, re-run this test and `git diff tests/eval/baseline.json` to see
what moved.
"""

import json
from pathlib import Path

import pytest

from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search
from tests.eval.metrics import evaluate_query_set, ndcg_at_k, recall_at_k, reciprocal_rank

_BASELINE_PATH = Path(__file__).parent / "baseline.json"


@pytest.mark.eval
class TestBaseline:
    async def test_generate_baseline(self, eval_kb):
        db, embedder, title_to_id, queries = eval_kb

        # Only score queries that have expected entries
        scorable = [q for q in queries if q.get("expected")]

        per_query: dict[str, dict] = {}
        results_map: dict[str, list[str]] = {}

        for q in scorable:
            search_query = SearchQuery(
                query=q["query"],
                limit=q["top_k"],
                project_ref=q.get("project_ref"),
                tags=q.get("tags"),
                include_stale=q.get("include_stale", False),
            )

            results = await hybrid_search(db, embedder, search_query)
            result_ids = [r.entry.id for r in results]
            results_map[q["id"]] = result_ids

            expected_ids = [title_to_id[t] for t in q["expected"]]
            k = q["top_k"]

            per_query[q["id"]] = {
                "mrr": round(reciprocal_rank(expected_ids, result_ids), 4),
                "recall_at_k": round(recall_at_k(expected_ids, result_ids, k), 4),
                "ndcg_at_k": round(ndcg_at_k(expected_ids, result_ids, k), 4),
                "top_k": k,
                "result_count": len(result_ids),
            }

        aggregate = evaluate_query_set(
            [
                {"id": q["id"], "expected": [title_to_id[t] for t in q["expected"]]}
                for q in scorable
            ],
            results_map,
            k=5,
        )
        aggregate = {key: round(val, 4) for key, val in aggregate.items()}

        baseline = {
            "aggregate": aggregate,
            "per_query": per_query,
        }

        _BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")

        # Print summary to stdout for -s visibility
        print(f"\n{'Query':<25} {'MRR':>6} {'R@k':>6} {'NDCG':>6}")
        print("-" * 47)
        for qid, m in per_query.items():
            print(f"{qid:<25} {m['mrr']:>6.3f} {m['recall_at_k']:>6.3f} {m['ndcg_at_k']:>6.3f}")
        print("-" * 47)
        a = aggregate
        print(
            f"{'MEAN':<25} {a['mean_mrr']:>6.3f}"
            f" {a['mean_recall_at_k']:>6.3f}"
            f" {a['mean_ndcg_at_k']:>6.3f}"
        )
        print(f"\nBaseline written to {_BASELINE_PATH}")
