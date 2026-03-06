"""Query classifier — routes questions to explore or summarize."""

import logging

from personal_kb.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """\
You are a query classifier for a knowledge base explorer. Given a user question, \
classify it as one of two modes:

- explore: The user wants to visually navigate the knowledge graph — see \
connections, browse clusters, or filter by node type. These are graph-centric \
requests, not questions seeking information. \
Examples: "show connections to Python", "explore the API cluster", \
"what nodes link to SQLite?", "browse debugging entries"
- summarize: The user wants information — an answer, overview, status update, \
or explanation synthesized from knowledge entries. Any question about a topic, \
even casually phrased, is summarize. \
Examples: "what work is happening on X?", "why did we choose FastAPI?", \
"tell me about the deployment pipeline", "what's related to Lightroom?", \
"what do we know about error handling?"

When in doubt, choose summarize — it's more useful for most questions.

Respond with EXACTLY one word: explore or summarize\
"""


async def classify_query(llm: LLMProvider, question: str) -> str:
    """Classify a question as 'explore' or 'summarize'.

    Returns 'explore' on any failure (cheaper, user can refine).
    """
    try:
        result = await llm.generate(question, system=_CLASSIFIER_SYSTEM)
        if result is not None:
            word = result.strip().lower()
            if word in ("explore", "summarize"):
                return word
            # Check if the response contains one of the words
            if "summarize" in word:
                return "summarize"
        logger.debug("Classifier returned unexpected: %r, defaulting to explore", result)
    except Exception:
        logger.debug("Classifier failed, defaulting to explore", exc_info=True)
    return "explore"
