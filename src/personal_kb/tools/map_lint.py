"""Advisory fact-free lint for mental_map entry bodies.

A mental_map is a structural orientation node: it should point to where
knowledge lives (kb-XXXXX references, relationships) rather than embed
retrievable values that belong in a factual_reference.

``lint_map_body`` is a pure, deterministic, regex-based heuristic. It performs
NO I/O, makes NO LLM/network call, and never raises. Unlike
``_check_secrets`` in ``kb_store.py`` (whose shape this mirrors), it never
returns an ``Error:`` string and never signals rejection — every warning is
purely advisory and every returned string starts with ``Map lint (advisory):``.

The discriminator (design §7.3): would a reader *act on the value directly*
(forbidden — a retrievable value) or *follow it to a source* (fine — a
pointer)? Counts-of-parts ("three stages", "12 nodes") are pointers in
disguise and are allowed; kb-XXXXX references are the desired content and are
never flagged.
"""

import re

MAP_BODY_ADVISORY_CHARS = 1500

_PREFIX = "Map lint (advisory): "

# kb-XXXXX references are pointers (the desired content) — strip them before
# any other heuristic so their digits never read as a retrievable numeral.
_KB_REF_RE = re.compile(r"kb-\d{5}")

_URL_RE = re.compile(r"https?://\S+")
# ENV_CAPS env-var tokens: all-caps with at least one underscore, e.g. KB_DB_PATH
_ENV_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
# File paths: ~/-rooted, or an absolute /a/b path with at least two segments.
_PATH_RE = re.compile(r"~/[^\s]+|/[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+")
# Dotted code identifiers / signatures: module.func, optionally a "(" call.
_DOTTED_RE = re.compile(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*")
# Quoted literal values (double-quote or backtick pairs; single quotes are
# skipped to avoid flagging ordinary apostrophes in prose).
_QUOTED_RE = re.compile(r'"[^"]+"|`[^`]+`')

_DECIMAL_RE = re.compile(r"\d+\.\d+")
_INT_RE = re.compile(r"\b\d+\b")
# A numeral immediately followed (optionally across one space) by a plural
# noun is a count-of-parts — a pointer-in-disguise, not a retrievable value.
_COUNT_TAIL_RE = re.compile(r"\s*[A-Za-z]+s\b")


def _has_config_numeral(text: str) -> bool:
    """Return True if the text contains a retrievable config-like numeral.

    Decimals (e.g. ``0.06``) and integers with >= 4 digits (e.g. ``8767``,
    ``51820``) or numerals adjacent to ``=``/``:`` are retrievable values.
    Count-of-parts (a numeral followed by a plural noun, any magnitude) are
    excluded — those are pointers-in-disguise.
    """
    if _DECIMAL_RE.search(text):
        return True

    # Remove decimals so their integer parts don't re-trigger below.
    no_dec = _DECIMAL_RE.sub(" ", text)
    for match in _INT_RE.finditer(no_dec):
        if _COUNT_TAIL_RE.match(no_dec[match.end() :]):
            continue  # count-of-parts — allowed
        digits = match.group()
        if len(digits) >= 4:
            return True
        before = no_dec[match.start() - 1 : match.start()]
        if before in ("=", ":"):
            return True
    return False


def lint_map_body(text: str) -> list[str]:
    """Return advisory warnings for a mental_map body (empty list = clean).

    Pure function: no I/O, no LLM, never raises. Every returned string starts
    with ``Map lint (advisory):`` and never with ``Error:``.
    """
    warnings: list[str] = []

    # kb-XXXXX references are pointers, not retrievable values — exclude them
    # from all value heuristics.
    cleaned = _KB_REF_RE.sub(" ", text)

    if _URL_RE.search(cleaned):
        warnings.append(
            _PREFIX + "contains a URL — point to a kb entry instead of embedding a "
            "retrievable link."
        )
    if _PATH_RE.search(cleaned):
        warnings.append(
            _PREFIX + "contains a file path — that is a retrievable value, not an "
            "orientation pointer."
        )
    if _ENV_RE.search(cleaned):
        warnings.append(
            _PREFIX + "contains an ENV_VAR-style token — that is a retrievable config "
            "value, not a pointer."
        )
    if _DOTTED_RE.search(cleaned):
        warnings.append(
            _PREFIX + "contains a dotted code identifier/signature — that is a "
            "retrievable value, not a pointer."
        )
    if _QUOTED_RE.search(cleaned):
        warnings.append(_PREFIX + "contains a quoted literal value.")
    if _has_config_numeral(cleaned):
        warnings.append(
            _PREFIX + "contains a config-like numeral — that is a retrievable value, "
            "not a count of parts."
        )

    if len(text) > MAP_BODY_ADVISORY_CHARS:
        warnings.append(
            _PREFIX + f"body is {len(text)} chars, exceeding the advisory ~1500 "
            "char guideline — a map should orient, not contain."
        )

    return warnings
