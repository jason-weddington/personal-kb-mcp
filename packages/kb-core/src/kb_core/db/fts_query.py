"""Backend-neutral full-text query tokenization and coverage threshold.

Both database backends turn a natural-language query into the same token
list (``fts_query_tokens``) and require the same number of those tokens to
match (``fts_min_coverage``), so SQLite FTS5 and Postgres tsquery search
agree on which entries a query can reach. Standard library only.
"""

from __future__ import annotations

import math
import re

# PostgreSQL's english.stop list (127 words), applied on both backends.
_STOPWORD_TEXT = (
    "i me my myself we our ours ourselves you your yours yourself "
    "yourselves he him his himself she her hers herself it its itself they "
    "them their theirs themselves what which who whom this that these those "
    "am is are was were be been being have has had having do does did doing "
    "a an the and but if or because as until while of at by for with about "
    "against between into through during before after above below to from "
    "up down in out on off over under again further then once here there "
    "when where why how all any both each few more most other some such no "
    "nor not only own same so than too very s t can will just don should "
    "now"
)
FTS_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

# Upper bound on tokens per query (guards SQLite's compound-select limit).
FTS_MAX_QUERY_TOKENS = 32

# How many query tokens an entry may lack and still match (for n > 2).
FTS_MAX_MISSING_TOKENS = 2

_WORD_RUN = re.compile(r"[^\W_]+")


def fts_query_tokens(query: str) -> list[str]:
    """Split a natural-language query into deduplicated full-text tokens.

    Apostrophes (straight and curly) become word boundaries, as in
    PostgreSQL's parser. Double quotes are stripped, tokens are lowercased,
    a token whose alphanumeric runs are all stopwords is dropped, repeats
    are dropped, and at most ``FTS_MAX_QUERY_TOKENS`` tokens are kept.
    """
    text = query.replace("'", " ").replace("\u2019", " ")
    out: list[str] = []
    for raw in text.split():
        tok = raw.replace('"', "").lower()
        parts = _WORD_RUN.findall(tok)
        if not parts or all(p in FTS_STOPWORDS for p in parts):
            continue
        if tok in out:
            continue
        out.append(tok)
        if len(out) == FTS_MAX_QUERY_TOKENS:
            break
    return out


def fts_min_coverage(n: int) -> int:
    """Return how many of ``n`` query tokens an entry must match.

    All of them when ``n <= 2``; otherwise ``max(ceil(n / 2), n - FTS_MAX_MISSING_TOKENS)``.
    """
    if n <= 2:
        return n
    return max(math.ceil(n / 2), n - FTS_MAX_MISSING_TOKENS)
