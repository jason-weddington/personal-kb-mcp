"""IR metrics for manual A/B comparison of search quality.

These are NOT asserted in CI — they're too noisy on a small corpus.
Use them when making ranking changes to compare before/after.
"""

import math


def reciprocal_rank(relevant_ids: list[str], result_ids: list[str]) -> float:
    """Mean Reciprocal Rank — 1/(rank of first relevant result).

    Returns 0.0 if no relevant result appears in result_ids.
    """
    for i, rid in enumerate(result_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevant_ids: list[str], result_ids: list[str], k: int) -> float:
    """Fraction of relevant items found in the top-k results.

    Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(result_ids[:k])
    found = sum(1 for r in relevant_ids if r in top_k)
    return found / len(relevant_ids)


def ndcg_at_k(relevant_ids: list[str], result_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: 1 if in relevant_ids, 0 otherwise.
    Returns 0.0 if relevant_ids is empty or no relevant results in top-k.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = result_ids[:k]

    # DCG: sum of 1/log2(rank+1) for relevant items in top-k
    dcg = 0.0
    for i, rid in enumerate(top_k):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: all relevant items ranked first
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query_set(
    queries: list[dict],
    results_map: dict[str, list[str]],
    k: int = 5,
) -> dict[str, float]:
    """Aggregate metrics across a set of queries.

    Args:
        queries: List of query dicts with 'id' and 'expected' keys.
        results_map: Mapping of query_id → list of result entry_ids.
        k: Cutoff for recall and NDCG.

    Returns:
        Dict with mean_mrr, mean_recall_at_k, mean_ndcg_at_k.
    """
    mrrs: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    for q in queries:
        qid = q["id"]
        expected = q.get("expected", [])
        results = results_map.get(qid, [])

        mrrs.append(reciprocal_rank(expected, results))
        recalls.append(recall_at_k(expected, results, k))
        ndcgs.append(ndcg_at_k(expected, results, k))

    n = len(queries) or 1
    return {
        "mean_mrr": sum(mrrs) / n,
        "mean_recall_at_k": sum(recalls) / n,
        "mean_ndcg_at_k": sum(ndcgs) / n,
    }
