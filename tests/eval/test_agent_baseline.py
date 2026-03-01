"""Agent baseline: run golden queries through agentic_query with real LLM.

NOT deterministic — requires a live Anthropic API key. Run manually with:
    uv run pytest tests/eval/test_agent_baseline.py -s
"""

import json
from pathlib import Path

import pytest

from personal_kb.graph.agent import agentic_query
from personal_kb.llm.anthropic import AnthropicLLMClient
from tests.eval.metrics import evaluate_query_set, ndcg_at_k, recall_at_k, reciprocal_rank

_BASELINE_PATH = Path(__file__).parent / "agent_baseline.json"


@pytest.mark.eval
class TestAgentBaseline:
    async def test_generate_agent_baseline(self, eval_kb):
        db, embedder, title_to_id, queries = eval_kb

        try:
            llm = AnthropicLLMClient()
            if not await llm.is_available():
                pytest.skip("Anthropic API not available")
        except Exception:
            pytest.skip("Anthropic SDK not available")

        scorable = [q for q in queries if q.get("expected")]

        per_query: dict[str, dict] = {}
        results_map: dict[str, list[str]] = {}

        for q in scorable:
            result = await agentic_query(db, embedder, llm, q["query"], max_tool_calls=4)
            result_ids = [eid for eid, _ in result.entries]
            results_map[q["id"]] = result_ids

            expected_ids = [title_to_id[t] for t in q["expected"]]
            k = q["top_k"]

            per_query[q["id"]] = {
                "mrr": round(reciprocal_rank(expected_ids, result_ids), 4),
                "recall_at_k": round(recall_at_k(expected_ids, result_ids, k), 4),
                "ndcg_at_k": round(ndcg_at_k(expected_ids, result_ids, k), 4),
                "top_k": k,
                "result_count": len(result_ids),
                "turns_used": result.turns_used,
                "reasoning": result.reasoning,
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

        print(f"\n{'Query':<25} {'MRR':>6} {'R@k':>6} {'NDCG':>6} {'Turns':>6}")
        print("-" * 55)
        for qid, m in per_query.items():
            print(
                f"{qid:<25} {m['mrr']:>6.3f} {m['recall_at_k']:>6.3f}"
                f" {m['ndcg_at_k']:>6.3f} {m['turns_used']:>6}"
            )
        print("-" * 55)
        a = aggregate
        print(
            f"{'MEAN':<25} {a['mean_mrr']:>6.3f}"
            f" {a['mean_recall_at_k']:>6.3f}"
            f" {a['mean_ndcg_at_k']:>6.3f}"
        )
        print(f"\nAgent baseline written to {_BASELINE_PATH}")

        await llm.close()
