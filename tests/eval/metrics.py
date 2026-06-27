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


def correct_rejection_rate(
    queries: list[dict],
    results_map: dict[str, list],
    *,
    score_threshold: float | None = None,
) -> float:
    """Fraction of abstention queries that correctly returned nothing.

    An abstention query is one with category=='abstention', expected==[], and
    include_stale is False (the include_stale=True sibling q6x-stale-flag-recovers
    is a recall control with a non-empty expected set, not an abstention case).

    A query is correctly rejected iff its result list is empty (strict), OR if
    score_threshold is given and the top result score < threshold.

    Returns 1.0 if the denominator is 0 (no abstention queries).

    Args:
        queries: List of query dicts with 'id', 'category', 'expected',
                 and 'include_stale' keys.
        results_map: Mapping of query_id → list of entry_ids (strings) or
                     (entry_id, score) tuples. Score is used only when
                     score_threshold is provided.
        score_threshold: If given, treat results whose top score < threshold
                         as correctly rejected even if the list is non-empty.

    Returns:
        Fraction of abstention queries correctly rejected (0.0-1.0).
    """
    abstention_queries = [
        q
        for q in queries
        if q.get("category") == "abstention"
        and q.get("expected") == []
        and not q.get("include_stale", False)
    ]

    if not abstention_queries:
        return 1.0

    correctly_rejected = 0
    for q in abstention_queries:
        qid = q["id"]
        results = results_map.get(qid, [])
        if not results:
            correctly_rejected += 1
        elif score_threshold is not None:
            # Support both (entry_id, score) tuples and plain entry_id strings
            first = results[0]
            top_score = first[1] if isinstance(first, (tuple, list)) else 0.0
            if top_score < score_threshold:
                correctly_rejected += 1

    return correctly_rejected / len(abstention_queries)


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
