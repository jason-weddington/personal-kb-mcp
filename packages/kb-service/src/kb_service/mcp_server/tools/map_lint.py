"""Advisory MCP-channel renderer over the kb-core map lint.

The lint itself lives in ``kb_core.map_lint`` — the single source of truth
shared with the web service's machine-principal write gate (hard 422) and
its dry-run validation endpoint (``docs/nightly-map-maintenance-design.md``,
Phase 0). This module is the MCP channel's thin adapter: it renders kb-core's
STRUCTURED findings into the advisory strings this channel has always
returned, so the three surfaces can never disagree about what a valid map is.

Observable behaviour is UNCHANGED from the pre-extraction module: the lint
on this channel stays advisory, every returned string starts with
``Map lint (advisory):`` and never with ``Error:`` — it never signals
rejection. ``lint_map_body`` performs NO I/O, makes NO LLM/network call, and
never raises.

The per-pointer budget constants and every purity rule live in kb-core and
were moved VERBATIM — do not retune them here (see the kb-core module
docstring for the corpus calibration).
"""

from kb_core.map_lint import (
    MAP_BODY_BASE_CHARS,
    MAP_BODY_PER_POINTER_CHARS,
    count_map_pointers,
    map_body_budget,
)
from kb_core.map_lint import (
    lint_map_body as _lint_map_findings,
)

_PREFIX = "Map lint (advisory): "

__all__ = [
    "MAP_BODY_BASE_CHARS",
    "MAP_BODY_PER_POINTER_CHARS",
    "count_map_pointers",
    "lint_map_body",
    "map_body_budget",
]


def lint_map_body(text: str) -> list[str]:
    """Return advisory warnings for a mental_map body (empty list = clean).

    Thin renderer: delegates to kb-core's structured lint and prefixes each
    finding's human text. Pure function: no I/O, no LLM, never raises. Every
    returned string starts with ``Map lint (advisory):`` and never with
    ``Error:``.
    """
    return [_PREFIX + finding.message for finding in _lint_map_findings(text)]
