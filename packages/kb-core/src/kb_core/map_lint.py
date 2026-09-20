"""Structured fact-free lint for mental_map entry bodies — single source of truth.

A mental_map is a structural orientation node: it should point to where
knowledge lives (kb-XXXXX references, relationships) rather than embed
retrievable values that belong in a factual_reference.

This module is THE single source of truth for what a valid mental_map body
is (docs/nightly-map-maintenance-design.md, Phase 0 "server-side hard lint
for the machine principal"). Three surfaces consume it, so they can never
disagree about the verdict:

* the MCP channel's advisory lint (``personal_kb.tools.map_lint`` — a thin
  renderer over these findings, one advisory string per finding),
* the web service's hard 422 write gate for the machine principal
  (``kb_service.routes.kb_write_routes``), and
* the web service's dry-run validation endpoint
  (``kb_service.routes.map_lint_routes``), whose non-2xx failure status is
  the somnus nightly loop's ``run_checks`` verdict.

``lint_map_body`` is a pure, deterministic, regex-based heuristic. It
performs NO I/O, makes NO LLM/network call, and never raises. Unlike
``_check_secrets`` in the MCP channel's ``kb_store.py`` (whose shape this
mirrors), it never signals rejection itself: it returns a STRUCTURED result —
a list of findings, each a machine-readable code plus human text — and each
channel decides what to do with them (advise, reject, or report).

The discriminator (design §7.3): would a reader *act on the value directly*
(forbidden — a retrievable value) or *follow it to a source* (fine — a
pointer)? Counts-of-parts ("three stages", "12 nodes") are pointers in
disguise and are allowed; kb-XXXXX references are the desired content and are
never flagged.

The body-size guideline is a compositional budget, not a flat cap.
It is a fixed base allowance for the orientation prose plus a per-pointer
allowance for each distinct kb- pointer (``map_body_budget``).
The old flat ~1500-char cap did not scale with pointer count, so enforcing it
would have stripped per-pointer glosses out of exactly the maps that earned the
most pointers.
Those glosses are craft the nightly map-maintenance loop is explicitly
forbidden from regenerating.
Pointers are counted format-agnostically as distinct ``kb-[0-9]{4,5}`` ids,
never by parsing a gloss layout.
The corpus's maps use incompatible gloss conventions (kb-01724 uses inline
bulleted gloss lines; kb-03257 uses a bare id list plus separate prose
paragraphs), so any layout-specific parser works on one map and breaks on
the other.
The literals are validated against the live corpus: both kb-01724 (2319 chars /
11 pointers) and kb-03257 (2540 chars / 20 pointers) pass at their current
sizes, and a thin 1-pointer body now trips a smaller budget than the old cap.

The constants, regexes, messages and thresholds below are moved VERBATIM from
the MCP channel's ``personal_kb.tools.map_lint`` (the extraction wave's
"move, do not retune" rule): the per-pointer budget was validated against all
27 live maps — zero failures, tightest case kb-01724 at 82% of budget — so a
behaviour change here is a regression, not an improvement.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

# Compositional body budget (see module docstring for why it is not flat):
# a base allowance for the orientation prose plus a per-pointer allowance for
# each gloss. Literals validated against the live corpus — kb-01724 (2319
# chars / 11 pointers, 82% of its 2825 budget) and kb-03257 (2540 chars / 20
# pointers, 58% of its 4400 budget) both pass, while a thin 1-pointer body
# gets a smaller budget than the old flat ~1500 cap.
MAP_BODY_BASE_CHARS = 900
MAP_BODY_PER_POINTER_CHARS = 175

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


class MapLintCode(StrEnum):
    """Machine-readable code for one lint finding.

    Stable across releases: the somnus gate keys on the HTTP status (not the
    codes), but the machine-principal 422 detail and the dry-run response
    both carry these codes, so an agent can branch on them without parsing
    human prose.
    """

    URL = "url"
    PATH = "path"
    ENV_VAR = "env_var"
    DOTTED_IDENTIFIER = "dotted_identifier"
    QUOTED_LITERAL = "quoted_literal"
    CONFIG_NUMERAL = "config_numeral"
    OVER_BUDGET = "over_budget"


@dataclass(frozen=True)
class MapLintFinding:
    """One structured lint finding: machine-readable code plus human text.

    ``message`` carries the human text WITHOUT any channel prefix — each
    channel renders or rejects with its own framing (the MCP channel prepends
    ``Map lint (advisory):``, the write gate embeds it in a 422 detail).
    """

    code: MapLintCode
    message: str


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


def lint_map_body(text: str) -> list[MapLintFinding]:
    """Return structured findings for a mental_map body (empty list = clean).

    Pure function: no I/O, no LLM, never raises. Each finding carries a
    machine-readable code plus the human text; no channel framing (advisory
    prefixes, rejection wording) is baked in here.
    """
    findings: list[MapLintFinding] = []

    # kb-XXXXX references are pointers, not retrievable values — exclude them
    # from all value heuristics.
    cleaned = _KB_REF_RE.sub(" ", text)

    if _URL_RE.search(cleaned):
        findings.append(
            MapLintFinding(
                MapLintCode.URL,
                "contains a URL — point to a kb entry instead of embedding a retrievable link.",
            )
        )
    if _PATH_RE.search(cleaned):
        findings.append(
            MapLintFinding(
                MapLintCode.PATH,
                "contains a file path — that is a retrievable value, not an orientation pointer.",
            )
        )
    if _ENV_RE.search(cleaned):
        findings.append(
            MapLintFinding(
                MapLintCode.ENV_VAR,
                "contains an ENV_VAR-style token — that is a retrievable config "
                "value, not a pointer.",
            )
        )
    if _DOTTED_RE.search(cleaned):
        findings.append(
            MapLintFinding(
                MapLintCode.DOTTED_IDENTIFIER,
                "contains a dotted code identifier/signature — that is a "
                "retrievable value, not a pointer.",
            )
        )
    if _QUOTED_RE.search(cleaned):
        findings.append(
            MapLintFinding(
                MapLintCode.QUOTED_LITERAL,
                "contains a quoted literal value.",
            )
        )
    if _has_config_numeral(cleaned):
        findings.append(
            MapLintFinding(
                MapLintCode.CONFIG_NUMERAL,
                "contains a config-like numeral — that is a retrievable value, "
                "not a count of parts.",
            )
        )

    pointer_count = count_map_pointers(text)
    budget = map_body_budget(pointer_count)
    if len(text) > budget:
        pointers_noun = "pointer" if pointer_count == 1 else "pointers"
        findings.append(
            MapLintFinding(
                MapLintCode.OVER_BUDGET,
                f"body is {len(text)} chars, exceeding the advisory budget of {budget} "
                f"chars for {pointer_count} {pointers_noun} "
                f"({MAP_BODY_BASE_CHARS} base + {MAP_BODY_PER_POINTER_CHARS} per pointer) "
                "— cut orientation prose, not pointer glosses.",
            )
        )

    return findings
