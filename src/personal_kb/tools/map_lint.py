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

The body-size guideline is a compositional budget, not a flat cap.
It is a fixed base allowance for the orientation prose plus a per-pointer
allowance for each distinct kb- pointer (`map_body_budget`).
The old flat ~1500-char cap did not scale with pointer count, so enforcing it
would have stripped per-pointer glosses out of exactly the maps that earned the
most pointers.
Those glosses are craft the nightly map-maintenance loop is explicitly
forbidden from regenerating.
Pointers are counted format-agnostically as distinct ``kb-[0-9]{4,5}`` ids,
never by parsing a gloss layout.
The corpus's maps use incompatible gloss conventions (kb-01724 uses inline
bulleted gloss lines; kb-03257 uses a bare id list plus separate prose
paragraphs), so any layout-specific parser works on one map and breaks on the
other.
The literals are validated against the live corpus: both kb-01724 (2319 chars /
11 pointers) and kb-03257 (2540 chars / 20 pointers) pass at their current
sizes, and a thin 1-pointer body now trips a smaller budget than the old cap.
"""

import re

# Compositional body budget (see module docstring for why it is not flat):
# a base allowance for the orientation prose plus a per-pointer allowance for
# each gloss. Literals validated against the live corpus — kb-01724 (2319
# chars / 11 pointers, 82% of its 2825 budget) and kb-03257 (2540 chars / 20
# pointers, 58% of its 4400 budget) both pass, while a thin 1-pointer body
# gets a smaller budget than the old flat ~1500 cap.
MAP_BODY_BASE_CHARS = 900
MAP_BODY_PER_POINTER_CHARS = 175

_PREFIX = "Map lint (advisory): "

# kb-XXXXX references are pointers (the desired content) — strip them before
# any other heuristic so their digits never read as a retrievable numeral.
_KB_REF_RE = re.compile(r"kb-\d{5}")

# Pointer ids for the body-size budget, counted format-agnostically: distinct
# ids matching kb-[0-9]{4,5} wherever they appear in the body, without parsing
# any particular gloss layout (the corpus's best maps use incompatible ones).
_POINTER_ID_RE = re.compile(r"kb-[0-9]{4,5}")

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


def count_map_pointers(text: str) -> int:
    """Count distinct kb- pointer ids in the body, format-agnostically.

    Pure function. Distinct ids matching ``kb-[0-9]{4,5}`` — deliberately NOT
    a parse of any gloss layout, since the corpus's maps gloss their pointers
    in incompatible ways (inline bulleted lists vs. prose paragraphs).
    """
    return len(set(_POINTER_ID_RE.findall(text)))


def map_body_budget(pointer_count: int) -> int:
    """Compositional body budget: prose base plus one allowance per pointer.

    A map with more pointers earns more room for their glosses; a thin map
    gets less room than the old flat cap allowed.
    """
    return MAP_BODY_BASE_CHARS + MAP_BODY_PER_POINTER_CHARS * pointer_count


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

    pointer_count = count_map_pointers(text)
    budget = map_body_budget(pointer_count)
    if len(text) > budget:
        pointers_noun = "pointer" if pointer_count == 1 else "pointers"
        warnings.append(
            _PREFIX + f"body is {len(text)} chars, exceeding the advisory budget of {budget} "
            f"chars for {pointer_count} {pointers_noun} "
            f"({MAP_BODY_BASE_CHARS} base + {MAP_BODY_PER_POINTER_CHARS} per pointer) "
            "— cut orientation prose, not pointer glosses."
        )

    return warnings
