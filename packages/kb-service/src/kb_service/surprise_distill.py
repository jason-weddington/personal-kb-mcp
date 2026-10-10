"""Surprise capture: pure helpers that distill candidates into KB lessons.

No I/O, no DB access and no logging here: ``distill_candidates`` in
``kb_service.surprise_worker`` calls the model, reads and writes the KB and
records each decision in ``surprise_distillations``. This module builds the
distiller prompt, parses its reply, builds the autonomous resolution and the
entry details, and decides whether a candidate matches (and may merge into)
an existing resolution.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from kb_core.cues import target_class
from kb_core.llm.json_parser import parse_json_object

from kb_service.prevention import Resolution
from kb_service.surprise import SurpriseCandidate

# bump on ANY change to SURPRISE_DISTILLER_SYSTEM, DISTILLER_INSTRUCTIONS,
# DISTILLER_SCHEMA_LINE, SHAPE_DESCRIPTIONS, build_distill_prompt,
# parse_distill_response, build_resolution, build_knowledge_details,
# find_exact_match, merge_block_reason, known_sessions or the S0-S13 order
SURPRISE_DISTILLER_VERSION: int = 3

SURPRISE_CONTRIBUTOR = "surprise-capture"
SURPRISE_HINT_KEY = "surprise_capture"
SURPRISE_TAG = "surprise-capture"
REDACTION_MARKER = "[REDACTED:"
GATE_DENY_MARKER = "KB soft gate (deny once"

DISTILL_CONFIDENCE_LEVEL = 0.7
DISTILL_SHORT_TITLE_MAX = 80
DISTILL_LONG_TITLE_MAX = 200
DISTILL_CORRECTED_FACT_MAX = 500
DISTILL_LESSON_MAX = 2000
NOT_DURABLE_WHY_MAX = 200
RESOLUTION_WRONG_BELIEF_MAX = 500
RESOLUTION_EVIDENCE_MAX = 500
SURPRISE_HINT_LIST_CAP = 100
SURPRISE_EVENT_IDS_CAP = 20

SURPRISE_DISTILLER_SYSTEM = (
    "You turn corrected beliefs from coding-agent sessions into durable"
    " knowledge-base lessons. Reply with exactly one JSON object and nothing"
    " else."
)

DISTILLER_INSTRUCTIONS = (
    "Decide whether this correction is a durable lesson for future sessions in"
    " this project. Set durable to false when the correction only applies to"
    " this one moment: a typo, a transient network or service error, a file or"
    " resource that did not exist yet, or a one-off preference; then set why to"
    " a short reason and leave the other fields empty. Set durable to true only"
    " when a future session in this project would plausibly hold the same wrong"
    " belief. corrected_fact is one self-contained sentence a future agent can"
    " act on, at most 300 characters. short_title is at most 80 characters and"
    " long_title at most 200 characters. lesson is at most 1500 characters and"
    " explains why the belief was wrong and how to avoid it. Never include"
    " secrets, tokens, passwords or credentials in any field."
)

DISTILLER_SCHEMA_LINE = (
    '{"durable": true|false, "why": str, "short_title": str, "long_title": str,'
    ' "corrected_fact": str, "lesson": str}'
)

SHAPE_DESCRIPTIONS: dict[int, str] = {
    1: (
        "A shell command failed, and a later command of the same kind succeeded"
        " in the same session."
    ),
    2: (
        "The human's next message corrected a claim or assumption from the"
        " assistant's previous turn."
    ),
    3: (
        "In one turn the agent reached a corrected understanding or a root"
        " cause, backed by tool output, that contradicts what it, the code, a"
        " comment, a doc, a config or the environment had indicated before."
    ),
}

DistillReject = Literal["llm_error", "unparseable", "invalid_fields", "not_durable"]

DistillOutcome = Literal[
    "written",
    "merged",
    "same_session",
    "covered",
    "not_durable",
    "redacted",
    "gate_induced",
    "llm_error",
    "unparseable",
    "invalid_fields",
    "invalid_resolution",
    "secret_detected",
    "no_project",
    "kb_error",
]


@dataclass(frozen=True)
class DistillVerdict:
    """The distiller's accepted reply: the lesson's titles and text."""

    short_title: str
    long_title: str
    corrected_fact: str
    lesson: str


# --- prompt and parser -------------------------------------------------------


def _output_str(candidate: SurpriseCandidate, key: str) -> str:
    return str(candidate.detector_output.get(key) or "")


def build_distill_prompt(candidate: SurpriseCandidate) -> str:
    """Build the distiller's user prompt from the candidate's detector output."""
    return "\n".join(
        [
            SHAPE_DESCRIPTIONS[candidate.shape],
            f"Project: {candidate.project}",
            f"Wrong belief: {_output_str(candidate, 'wrong_belief')}",
            f"Corrected fact: {_output_str(candidate, 'corrected_fact')}",
            f"Evidence: {_output_str(candidate, 'evidence_excerpt')}",
            DISTILLER_INSTRUCTIONS,
            DISTILLER_SCHEMA_LINE,
        ]
    )


def parse_distill_response(
    raw: str | None,
) -> tuple[DistillVerdict | None, DistillReject | None]:
    """Parse the distiller's reply; exactly one element is non-None."""
    if raw is None:
        return None, "llm_error"
    obj = parse_json_object(raw)
    if obj is None:
        return None, "unparseable"
    if obj.get("durable") is not True:
        return None, "not_durable"
    fields: list[str] = []
    for key in ("short_title", "long_title", "corrected_fact", "lesson"):
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, "invalid_fields"
        fields.append(value.strip())
    short_title, long_title, corrected_fact, lesson = fields
    return (
        DistillVerdict(
            short_title=short_title[:DISTILL_SHORT_TITLE_MAX],
            long_title=long_title[:DISTILL_LONG_TITLE_MAX],
            corrected_fact=corrected_fact[:DISTILL_CORRECTED_FACT_MAX],
            lesson=lesson[:DISTILL_LESSON_MAX],
        ),
        None,
    )


def not_durable_reason(raw: str | None) -> str:
    """The distiller's ``why`` for a not-durable reply, else ``''``."""
    if not raw:
        return ""
    obj = parse_json_object(raw)
    if not isinstance(obj, dict):
        return ""
    why = obj.get("why")
    if not isinstance(why, str):
        return ""
    return why.strip()[:NOT_DURABLE_WHY_MAX]


# --- resolution builders -----------------------------------------------------


def shape1_cue(
    wrong_belief: str, cue_target_class: str | None = None
) -> dict[str, str] | None:
    """The Bash cue for a shape-1 wrong belief, when its class is a fixed point.

    *cue_target_class* is the class the detector paired on; when absent the
    class of the whole wrong belief is used.
    """
    tc = cue_target_class or target_class("Bash", wrong_belief)
    if tc != "" and target_class("Bash", tc) == tc:
        return {"tool": "Bash", "target_class": tc}
    return None


def build_resolution(
    candidate: SurpriseCandidate, verdict: DistillVerdict
) -> dict[str, Any]:
    """Build the autonomous ``hints.resolution`` for a new entry."""
    wrong_belief = _output_str(candidate, "wrong_belief").strip()[
        :RESOLUTION_WRONG_BELIEF_MAX
    ]
    evidence = _output_str(candidate, "evidence_excerpt").strip()[
        :RESOLUTION_EVIDENCE_MAX
    ]
    provenance: dict[str, str]
    if candidate.turn_event_ids:
        provenance = {
            "capture": "autonomous",
            "grounding": "observed",
            "event_id": candidate.turn_event_ids[-1],
        }
    else:
        provenance = {"capture": "autonomous", "grounding": "asserted"}
    resolution: dict[str, Any] = {
        "corrected_fact": verdict.corrected_fact,
        "wrong_belief": wrong_belief,
        "evidence": evidence,
        "provenance": provenance,
        "observed_sessions": 1,
        "scope": "project",
    }
    if candidate.shape == 1:
        cue = shape1_cue(wrong_belief, _output_str(candidate, "cue_target_class"))
        if cue is not None:
            resolution["cue"] = cue
    return resolution


def build_knowledge_details(
    candidate: SurpriseCandidate,
    verdict: DistillVerdict,
    resolution: Mapping[str, Any],
) -> str:
    """Build the new entry's ``knowledge_details`` text."""
    return (
        f"{verdict.lesson}\n\nWrong belief: {resolution['wrong_belief']}\n"
        f"Corrected fact: {verdict.corrected_fact}\n"
        f"Evidence: {resolution['evidence']}\n\n"
        f"Captured autonomously by surprise capture (shape {candidate.shape},"
        f" candidate {candidate.id}) from session {candidate.session_id},"
        f" turn events {', '.join(candidate.turn_event_ids)}."
    )


# --- match helpers -----------------------------------------------------------


def normalize_text(s: str) -> str:
    """Collapse whitespace, strip and casefold."""
    return re.sub(r"\s+", " ", s).strip().casefold()


def _cue_tuple(cue: Mapping[str, str] | None) -> tuple[str, str]:
    c = cue or {}
    return c.get("tool", ""), c.get("target_class", "")


def find_exact_match(
    resolutions: Sequence[Resolution],
    wrong_belief: str,
    cue: Mapping[str, str] | None,
) -> Resolution | None:
    """The first resolution with the same normalized wrong belief and cue."""
    wanted = normalize_text(wrong_belief)
    if wanted == "":
        return None
    key = _cue_tuple(cue)
    for r in resolutions:
        if (
            normalize_text(r.wrong_belief) == wanted
            and (r.cue_tool, r.cue_target_class) == key
        ):
            return r
    return None


def _resolution_of(hints: Mapping[str, object]) -> dict[str, Any] | None:
    res = hints.get("resolution")
    return res if isinstance(res, dict) else None


def stored_capture(hints: Mapping[str, object]) -> str | None:
    """The stored resolution's capture; None without one; fail closed."""
    res = _resolution_of(hints)
    if res is None:
        return None
    prov = res.get("provenance")
    if isinstance(prov, dict) and isinstance(prov.get("capture"), str):
        return str(prov["capture"])
    return "deliberate"


def stored_observed_sessions(hints: Mapping[str, object]) -> int:
    """The stored resolution's ``observed_sessions`` (>= 1), else 1."""
    res = _resolution_of(hints) or {}
    value = res.get("observed_sessions")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return 1


def surprise_hint(
    hints: Mapping[str, object],
) -> tuple[list[str], list[int], list[str]]:
    """Read ``hints.surprise_capture`` as (sessions, candidate_ids, event_ids)."""
    block = hints.get(SURPRISE_HINT_KEY)
    if not isinstance(block, dict):
        return [], [], []

    def _list(key: str) -> list[Any]:
        value = block.get(key)
        return value if isinstance(value, list) else []

    sessions = [s for s in _list("sessions") if isinstance(s, str)]
    ids = [
        i
        for i in _list("candidate_ids")
        if isinstance(i, int) and not isinstance(i, bool)
    ]
    event_ids = [e for e in _list("event_ids") if isinstance(e, str)]
    return sessions, ids, event_ids


def known_sessions(hints: Mapping[str, object]) -> set[str]:
    """Sessions already behind an entry (hint list plus provenance event_id)."""
    sessions = set(surprise_hint(hints)[0])
    res = _resolution_of(hints) or {}
    prov = res.get("provenance")
    if isinstance(prov, dict):
        event_id = prov.get("event_id")
        if isinstance(event_id, str) and ":" in event_id:
            sessions.add(event_id.rsplit(":", 1)[0])
    return sessions


def merge_block_reason(
    hints: Mapping[str, object], cue: Mapping[str, str] | None
) -> str | None:
    """Why a matched entry may not absorb a recurrence, else None."""
    res = _resolution_of(hints)
    if res is None:
        return "no_resolution"
    if stored_capture(hints) != "autonomous":
        return "deliberate"
    if res.get("scope", "project") != "project":
        return "global"
    rc = res.get("cue")
    rc_d: dict[str, Any] = rc if isinstance(rc, dict) else {}
    tool = rc_d.get("tool")
    tc = rc_d.get("target_class")
    stored = (
        tool if isinstance(tool, str) else "",
        tc if isinstance(tc, str) else "",
    )
    if stored != _cue_tuple(cue):
        return "cue_mismatch"
    return None


def merged_surprise_hint(
    hints: Mapping[str, object], candidate: SurpriseCandidate, *, new: bool = False
) -> dict[str, Any]:
    """The ``hints.surprise_capture`` value after folding in *candidate*.

    The shape is written only for a *new* entry; a merge keeps the stored value
    (or its absence) and never overwrites or adds it.
    """
    sessions, ids, event_ids = surprise_hint(hints)
    block = hints.get(SURPRISE_HINT_KEY)
    shape_part: dict[str, Any] = {}
    if new:
        shape_part = {"shape": candidate.shape}
    elif isinstance(block, dict) and "shape" in block:
        shape_part = {"shape": block["shape"]}
    return {
        **shape_part,
        "sessions": [*sessions, candidate.session_id][-SURPRISE_HINT_LIST_CAP:],
        "candidate_ids": [*ids, candidate.id][-SURPRISE_HINT_LIST_CAP:],
        "event_ids": (event_ids + candidate.turn_event_ids[-1:])[
            -SURPRISE_EVENT_IDS_CAP:
        ],
    }
