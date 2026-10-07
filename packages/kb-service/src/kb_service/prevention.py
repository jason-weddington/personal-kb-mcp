"""Prevention channels: the SessionStart gotcha slice and the soft-gate index.

Pure helpers over the KB connection — no model or LLM call anywhere.

Resolution record format (the contract of record; future producers must match it)
================================================================================

A *resolution* is an ACTIVE KB entry (``is_active = 1``, ``superseded_by IS
NULL``, ``entry_type != 'mental_map'``) whose ``hints`` JSON carries the key
``"resolution"`` holding a dict. Its project is the entry's ``project_ref``;
there is no schema change to ``knowledge_entries``::

    {"resolution": {
        "corrected_fact": "Push to origin, never to github",
        "wrong_belief": "git push github main is fine",
        "evidence": "pre-push hook refused it",
        "cue": {"tool": "Bash", "target_class": "git push", "args_prefix": "origin"},
        "provenance": {"capture": "deliberate", "grounding": "observed",
                       "event_id": "..."},
        "observed_sessions": 1,
        "scope": "project"
    }}

``corrected_fact`` is required; every other key is optional.
``capture`` is ``deliberate``/``autonomous``, ``grounding`` is
``observed``/``asserted``, ``provenance.event_id`` is reserved for
``grounding=observed``, and ``scope`` is ``project`` (default) or ``global``.

Parse rules (applied after ``json.loads(hints)``); any violation SKIPs the
entry — counted in :attr:`LoadStats.skipped_malformed`, never raised:

* ``resolution`` must be a dict; ``corrected_fact`` a str with ``.strip() != ''``.
* ``wrong_belief`` / ``evidence`` default to ``''``; a non-str value SKIPs.
* ``cue`` is optional; each of ``tool`` / ``target_class`` is ``''`` when
  absent or not a str. ``args_prefix``, when present, must be a non-empty str.
* ``provenance.capture`` must be ``deliberate``/``autonomous`` and
  ``provenance.grounding`` ``observed``/``asserted`` when set.
* ``observed_sessions`` must be an int (not a bool) ``>= 1``, else it is 1.
* ``scope`` must be ``project`` or ``global``. A ``global`` resolution applies
  to every session regardless of the session's project.

Unknown keys in ``resolution`` and ``provenance`` are ignored.

A resolution is OBSERVED-ONCE iff ``provenance.capture == 'autonomous'`` AND
``observed_sessions < 2``.

Only Bash cues with a two-word ``target_class`` (``'git push'``) are admitted
to the gate index; every other resolution still reaches the slice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from kb_service.models import IndexCue, SliceItem

logger = logging.getLogger(__name__)

INDEX_CAP = 200
SLICE_CAP = 20
_TEXT_CAP = 300
_SLICE_TEXT_CAP = 4000

_CAPTURES = frozenset({"deliberate", "autonomous"})
_GROUNDINGS = frozenset({"observed", "asserted"})
_SCOPES = frozenset({"project", "global"})

_RESOLUTIONS_SQL = (
    "SELECT id, updated_at, hints, project_ref FROM knowledge_entries"
    " WHERE is_active = 1 AND superseded_by IS NULL"
    " AND entry_type != 'mental_map'"
    ' AND (project_ref = ? OR hints LIKE \'%"scope": "global"%\''
    ' OR hints LIKE \'%"scope":"global"%\')'
    " AND hints LIKE '%\"resolution\"%'"
    " ORDER BY updated_at DESC, id DESC"
)

_CORRECTIONS_SQL = (
    "SELECT s.id, s.short_title, t.short_title FROM knowledge_entries t"
    " JOIN knowledge_entries s ON s.id = t.superseded_by"
    " WHERE t.project_ref = ? AND s.is_active = 1"
    " ORDER BY s.created_at DESC, s.id DESC LIMIT ?"
)


@dataclass(frozen=True)
class Resolution:
    """A parsed ``hints.resolution`` record."""

    entry_id: str
    updated_at: str
    wrong_belief: str
    corrected_fact: str
    evidence: str
    cue_tool: str
    cue_target_class: str
    capture: str | None
    grounding: str | None
    observed_sessions: int
    observed_once: bool
    scope: str = "project"
    cue_args_prefix: str = ""


@dataclass(frozen=True)
class Correction:
    """A supersedes edge: the superseder's title replaces the superseded one."""

    entry_id: str
    corrected_fact: str
    wrong_belief: str


@dataclass
class LoadStats:
    """Counters from :func:`load_resolutions`.

    ``resolutions_total`` counts well-formed, in-scope resolutions (before
    the observed-once filter).
    """

    resolutions_total: int = 0
    skipped_malformed: int = 0
    skipped_observed_once: int = 0


class _SkipError(Exception):
    """Internal: the entry violates the AC1 parse rules."""


def _opt_str(container: dict[str, Any], key: str) -> str:
    value = container.get(key, "")
    if not isinstance(value, str):
        raise _SkipError(key)
    return value


def parse_resolution(entry_id: str, updated_at: str, hints_raw: object) -> Resolution:
    """Parse one entry's hints into a :class:`Resolution`.

    Raises ``_SkipError`` when the entry violates the parse rules.
    """
    try:
        hints = json.loads(hints_raw) if isinstance(hints_raw, str) else hints_raw
    except (TypeError, ValueError) as exc:
        raise _SkipError("hints") from exc
    if not isinstance(hints, dict):
        raise _SkipError("hints")
    res = hints.get("resolution")
    if not isinstance(res, dict):
        raise _SkipError("resolution")
    corrected = res.get("corrected_fact")
    if not isinstance(corrected, str) or not corrected.strip():
        raise _SkipError("corrected_fact")
    wrong_belief = _opt_str(res, "wrong_belief")
    evidence = _opt_str(res, "evidence")

    cue = res.get("cue")
    cue_d = cue if isinstance(cue, dict) else {}
    cue_tool = cue_d.get("tool") if isinstance(cue_d.get("tool"), str) else ""
    cue_tc = (
        cue_d.get("target_class") if isinstance(cue_d.get("target_class"), str) else ""
    )
    args_prefix = ""
    if "args_prefix" in cue_d:
        raw_prefix = cue_d["args_prefix"]
        if not isinstance(raw_prefix, str) or not raw_prefix.strip():
            raise _SkipError("args_prefix")
        args_prefix = raw_prefix

    capture: str | None = None
    grounding: str | None = None
    prov = res.get("provenance")
    if isinstance(prov, dict):
        if "capture" in prov:
            if prov["capture"] not in _CAPTURES:
                raise _SkipError("capture")
            capture = prov["capture"]
        if "grounding" in prov:
            if prov["grounding"] not in _GROUNDINGS:
                raise _SkipError("grounding")
            grounding = prov["grounding"]

    observed = res.get("observed_sessions")
    if not isinstance(observed, int) or isinstance(observed, bool) or observed < 1:
        observed = 1

    scope = res.get("scope", "project")
    if scope not in _SCOPES:
        raise _SkipError("scope")

    return Resolution(
        entry_id=entry_id,
        updated_at=updated_at,
        wrong_belief=wrong_belief,
        corrected_fact=corrected,
        evidence=evidence,
        cue_tool=str(cue_tool),
        cue_target_class=str(cue_tc),
        capture=capture,
        grounding=grounding,
        observed_sessions=observed,
        observed_once=capture == "autonomous" and observed < 2,
        scope=str(scope),
        cue_args_prefix=args_prefix,
    )


async def load_resolutions(
    db: Any, project: str, include_observed_once: bool
) -> tuple[list[Resolution], LoadStats]:
    """Load the resolutions that apply to *project* (plus global ones).

    Project-scoped resolutions sort before global ones (stable within each
    group, newest first), so project knowledge wins the caps.
    """
    stats = LoadStats()
    cursor = await db.execute(_RESOLUTIONS_SQL, (project,))
    rows = await cursor.fetchall()
    own: list[Resolution] = []
    global_: list[Resolution] = []
    first_skipped: str | None = None
    for row in rows:
        entry_id, updated_at, hints_raw, project_ref = (
            str(row[0]),
            str(row[1]),
            row[2],
            row[3],
        )
        try:
            res = parse_resolution(entry_id, updated_at, hints_raw)
        except _SkipError:
            stats.skipped_malformed += 1
            if first_skipped is None:
                first_skipped = entry_id
            continue
        in_project = project_ref == project
        if not in_project and res.scope != "global":
            continue
        stats.resolutions_total += 1
        if res.observed_once and not include_observed_once:
            stats.skipped_observed_once += 1
            continue
        (own if in_project and res.scope == "project" else global_).append(res)
    if first_skipped is not None:
        logger.warning(
            "prevention skipped malformed resolution entry_id=%s project=%s skipped=%d",
            first_skipped,
            project,
            stats.skipped_malformed,
        )
    return own + global_, stats


async def load_corrections(db: Any, project: str, limit: int) -> list[Correction]:
    """Return the newest supersedes corrections for *project*'s superseded entries."""
    cursor = await db.execute(_CORRECTIONS_SQL, (project, limit))
    rows = await cursor.fetchall()
    return [
        Correction(
            entry_id=str(r[0]),
            corrected_fact=str(r[1] or ""),
            wrong_belief=str(r[2] or ""),
        )
        for r in rows
    ]


def provenance_label(item: Resolution | Correction) -> str:
    """Render the provenance tag shown on a slice line and in a deny reason."""
    if isinstance(item, Correction):
        return "supersedes-edge"
    if item.capture is None and item.grounding is None:
        label = "unlabelled"
    else:
        label = f"{item.capture or '?'}/{item.grounding or '?'}"
    if item.observed_once:
        label += ", observed once, unconfirmed"
    return label


def _gate_trusted(r: Resolution) -> bool:
    """Only trusted resolutions may become a pre-action deny.

    A deny is delivered as a correction, so it requires a deliberate store
    (an agent saving it because the session asked) or observed grounding
    (tool-output evidence). Autonomous + asserted resolutions -- e.g. LLM-drafted
    seeds -- and resolutions without provenance reach the session-start slice
    only, as labelled low-trust context. Fail closed: unknown means untrusted.
    """
    return r.capture == "deliberate" or r.grounding == "observed"


def build_gate_index(resolutions: list[Resolution]) -> tuple[list[IndexCue], int]:
    """Admit Bash two-word-class cues to the gate index.

    Returns ``(index, truncated_count)``.
    """
    admitted = [
        r
        for r in resolutions
        if r.cue_tool == "Bash" and " " in r.cue_target_class and _gate_trusted(r)
    ]
    kept = admitted[:INDEX_CAP]
    index = [
        IndexCue(
            resolution_id=r.entry_id,
            updated_at=r.updated_at,
            tool=r.cue_tool,
            target_class=r.cue_target_class,
            args_prefix=r.cue_args_prefix,
            wrong_belief=r.wrong_belief,
            corrected_fact=r.corrected_fact,
            evidence=r.evidence,
            provenance_label=provenance_label(r),
            observed_once=r.observed_once,
        )
        for r in kept
    ]
    return index, len(admitted) - len(kept)


def build_slice(
    resolutions: list[Resolution], corrections: list[Correction]
) -> tuple[list[SliceItem], int]:
    """Resolutions first, then corrections, de-duplicated; ``(items, dropped)``."""
    seen: set[str] = set()
    sources: list[Resolution | Correction] = [*resolutions, *corrections]
    items: list[SliceItem] = []
    for item in sources:
        if item.entry_id in seen:
            continue
        seen.add(item.entry_id)
        items.append(
            SliceItem(
                entry_id=item.entry_id,
                corrected_fact=item.corrected_fact,
                wrong_belief=item.wrong_belief,
                provenance_label=provenance_label(item),
            )
        )
    return items[:SLICE_CAP], max(0, len(items) - SLICE_CAP)


def render_slice(project: str, items: list[SliceItem]) -> str:
    """Render the gotcha slice text; ``''`` for no items; at most 4000 chars."""
    if not items:
        return ""
    header = (
        f"Known gotchas for {project} from the KB"
        " (each replaces an earlier wrong belief):"
    )
    lines = []
    for item in items:
        was = f" (was: {item.wrong_belief[:_TEXT_CAP]})" if item.wrong_belief else ""
        lines.append(
            f"- {item.corrected_fact[:_TEXT_CAP]}{was}"
            f" [{item.provenance_label}; {item.entry_id}]"
        )
    while lines and len("\n".join([header, *lines])) > _SLICE_TEXT_CAP:
        lines.pop()
    return "\n".join([header, *lines])
