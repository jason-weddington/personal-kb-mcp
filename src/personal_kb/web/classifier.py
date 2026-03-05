"""Query classifier — routes questions to explore or summarize."""

import logging

from personal_kb.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """\
You are a query classifier for a knowledge base explorer. Given a user question, \
classify it as one of two modes:

- explore: The user wants to browse, discover, or navigate the knowledge graph. \
Examples: "what connects to Python?", "show me debugging entries", \
"what's related to SQLite?", "explore the API cluster"
- summarize: The user wants a direct answer synthesized from knowledge. \
Examples: "why did we choose FastAPI?", "explain the deployment pipeline", \
"what's our convention for error handling?"

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
