"""Query classifier: explore vs summarize mode for the SSE query stream.

Uses the LLM to classify a user question as either ``explore`` (graph
navigation) or ``summarize`` (information retrieval / synthesis).  Falls back
to ``explore`` on any error, None result, or unrecognised LLM output.
"""

from kb_core.llm.provider import LLMProvider

_CLASSIFIER_SYSTEM = (
    "You are a query classifier for a knowledge base explorer. Given a user "
    "question, classify it as one of two modes:\n\n"
    "- explore: The user wants to visually navigate the knowledge graph — see "
    "connections, browse clusters, or filter by node type. These are "
    "graph-centric requests, not questions seeking information. Examples: "
    '"show connections to Python", "explore the API cluster", '
    '"what nodes link to SQLite?", "browse debugging entries"\n'
    "- summarize: The user wants information — an answer, overview, status "
    "update, or explanation synthesized from knowledge entries. Any question "
    "about a topic, even casually phrased, is summarize. Examples: "
    '"what work is happening on X?", "why did we choose FastAPI?", '
    '"tell me about the deployment pipeline", "what\'s related to Lightroom?", '
    '"what do we know about error handling?"\n\n'
    "When in doubt, choose summarize — it's more useful for most questions.\n\n"
    "Respond with EXACTLY one word: explore or summarize"
)


async def classify_query(llm: LLMProvider, question: str) -> str:
    """Classify a query as ``'explore'`` or ``'summarize'``.

    Calls the LLM with the classifier system prompt and inspects the response.
    Returns ``'explore'`` on any failure (exception, None result, or
    unrecognised output) so the stream degrades gracefully.

    Args:
        llm: LLM provider used for classification (``kb.query_llm``).
        question: The user's question string.

    Returns:
        ``'explore'`` or ``'summarize'``.
    """
    try:
        result = await llm.generate(question, system=_CLASSIFIER_SYSTEM)
    except Exception:
        return "explore"
    if result is not None:
        word = result.strip().lower()
        if word in ("explore", "summarize"):
            return word
        if "summarize" in word:
            return "summarize"
    return "explore"
