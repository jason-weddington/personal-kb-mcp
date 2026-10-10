"""Surprise capture: pure helpers that distill candidates into KB lessons.

No I/O, no DB access and no logging here: ``distill_candidates`` in
``kb_service.surprise_worker`` calls the model, reads and writes the KB and
records each decision in ``surprise_distillations``. This module builds the
distiller prompt, parses its reply, builds and parses the critic pass that
checks a drafted lesson against its evidence, builds the autonomous resolution and the
entry details, and decides whether a candidate matches (and may merge into)
an existing resolution.
"""

import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from kb_core.cues import bash_segments, target_class
from kb_core.llm.json_parser import parse_json_object

from kb_service.prevention import Resolution
from kb_service.surprise import SurpriseCandidate

# bump on ANY change to SURPRISE_DISTILLER_SYSTEM, DISTILLER_INSTRUCTIONS,
# DISTILLER_SCHEMA_LINE, SHAPE_DESCRIPTIONS, build_distill_prompt,
# parse_distill_response, build_resolution, build_knowledge_details,
# shape1_cue, find_exact_match, merge_block_reason, known_sessions,
# merged_surprise_hint or the S0-S13 order
SURPRISE_DISTILLER_VERSION: int = 5

# bump on ANY change to SURPRISE_CRITIC_SYSTEM, CRITIC_INSTRUCTIONS,
# CRITIC_SCHEMA_LINE, build_critic_prompt or parse_critic_response
SURPRISE_CRITIC_VERSION: int = 2

logger = logging.getLogger(__name__)

# Autonomous lessons expire this many days after the last observation unless
# they reach PERMANENT_OBSERVED_SESSIONS distinct sessions.
SURPRISE_LESSON_TTL_DAYS: int = 30
SURPRISE_LESSON_TTL_ENV = "KB_SURPRISE_LESSON_TTL_DAYS"
PERMANENT_OBSERVED_SESSIONS = 3


def lesson_ttl_days() -> int:
    """TTL in days: env override when an int in 1..3650, else the default."""
    raw = os.environ.get(SURPRISE_LESSON_TTL_ENV, "").strip()
    if raw == "":
        return SURPRISE_LESSON_TTL_DAYS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if 1 <= value <= 3650:
        return value
    logger.warning(
        "surprise_distill bad_lesson_ttl value=%r fallback=%d",
        raw,
        SURPRISE_LESSON_TTL_DAYS,
    )
    return SURPRISE_LESSON_TTL_DAYS


def lesson_expires_at(now: datetime | None = None) -> datetime:
    """``now`` + the lesson TTL (UTC)."""
    return (now or datetime.now(UTC)) + timedelta(days=lesson_ttl_days())


SURPRISE_CONTRIBUTOR = "surprise-capture"
SURPRISE_HINT_KEY = "surprise_capture"
SURPRISE_TAG = "surprise-capture"
REDACTION_MARKER = "[REDACTED:"
GATE_DENY_MARKER = "KB soft gate (deny once"

SURPRISE_MODES = frozenset({"interactive", "headless"})
SHAPE1_ARGS_PREFIX_MAX_TOKENS = 3

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
    " a short reason and leave the other fields empty. Also set durable to false"
    " when the failure was caused by this session's own in-progress changes or"
    " by a transient state of this checkout or environment (for example lint or"
    " test errors present only at that moment, a branch not yet fetched, a file"
    " not yet created, a service that was briefly down), or when the successful"
    " command only narrowed the scope of the same check for this task. A"
    " durable lesson states a fact about the project, its tools or its"
    " environment that will still be true for a fresh session tomorrow. Set"
    " durable to true only"
    " when a future session in this project would plausibly hold the same wrong"
    " belief. State the corrected fact at exactly the scope the evidence"
    " supports: if the user corrected one narrow point, record that narrow"
    " point, never a broader rule. Use only facts present in the evidence; add"
    " no details, motives or events that are not there. If the correction"
    " describes a temporary state, a work-in-progress, or something the user"
    " says will change, set durable to false. If the correction is a priority"
    " or preference for the current task rather than a fact about the project,"
    " its tools or its environment, set durable to false. corrected_fact is"
    " one self-contained sentence a future agent can"
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

SURPRISE_CRITIC_SYSTEM = (
    "You review drafted knowledge-base lessons from coding-agent sessions"
    " against the evidence they were drawn from. Reply with exactly one JSON"
    " object and nothing else."
)

CRITIC_INSTRUCTIONS = (
    "Judge the drafted lesson strictly against the evidence above. supported"
    " is true only when every claim in the draft is present in the evidence;"
    " any added detail, motive, event or cause that the evidence does not"
    " state makes it false. scope_ok is true only when the draft states the"
    " correction at exactly the scope the evidence supports; a narrow"
    " correction stretched into a broader or general rule makes it false."
    " durable is true only when the draft is a fact about the project, its"
    " tools or its environment that will still be true for a fresh session"
    " tomorrow; a temporary state, a work-in-progress, something the user says"
    " will change, or a priority or preference for the current task makes it"
    " false. misleading is true when a future agent acting on the draft would"
    " plausibly do the wrong thing, for example because the cause is misplaced"
    " or the rule is overstated. "
    "A claim the user states hedged or speculatively (for example 'I think',"
    " 'maybe', 'not sure', 'probably', or a question) is not evidence of a"
    " fact: treat such a claim as unsupported unless a tool result confirms"
    " it."
    " reason is one short sentence explaining the"
    " judgement."
)

CRITIC_SCHEMA_LINE = (
    '{"supported": true|false, "scope_ok": true|false, "durable": true|false,'
    ' "misleading": true|false, "reason": str}'
)

CRITIC_REASON_MAX = 200
CRITIC_EVIDENCE_ITEM_MAX = 2000
CRITIC_TOOL_RESULTS_MAX = 20

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


# --- critic ------------------------------------------------------------------

CriticReject = Literal["llm_error", "unparseable"]


@dataclass(frozen=True)
class CriticVerdict:
    """The critic's parsed reply on one drafted lesson."""

    supported: bool
    scope_ok: bool
    durable: bool
    misleading: bool
    reason: str

    @property
    def accepted(self) -> bool:
        """True only when the draft may be written or merged."""
        return self.supported and self.scope_ok and self.durable and not self.misleading


def build_critic_prompt(
    candidate: SurpriseCandidate,
    verdict: DistillVerdict,
    *,
    user_correction: str | None = None,
    tool_results: Sequence[str] = (),
) -> str:
    """Build the critic's user prompt: the candidate's evidence and the draft.

    *user_correction* is the human's message (shape 2) and *tool_results* the
    turn's tool-result excerpts (shape 3), both read from the stored turn
    digests; either may be empty when the digest is gone.
    """
    parts = [
        SHAPE_DESCRIPTIONS[candidate.shape],
        f"Project: {candidate.project}",
        "",
        "Evidence:",
        f"Wrong belief: {_output_str(candidate, 'wrong_belief')}",
        f"Corrected fact: {_output_str(candidate, 'corrected_fact')}",
        f"Evidence excerpt: {_output_str(candidate, 'evidence_excerpt')}",
    ]
    if candidate.shape == 2 and (user_correction or "").strip():
        parts.append(
            "The human's correction: "
            f"{(user_correction or '').strip()[:CRITIC_EVIDENCE_ITEM_MAX]}"
        )
    if candidate.shape == 3:
        excerpts = [t.strip() for t in tool_results if t.strip()]
        if excerpts:
            parts.append("Tool results from the turn:")
            parts.extend(
                f"[tool_result] {t[:CRITIC_EVIDENCE_ITEM_MAX]}"
                for t in excerpts[:CRITIC_TOOL_RESULTS_MAX]
            )
    parts.extend(
        [
            "",
            "Drafted lesson:",
            f"Short title: {verdict.short_title}",
            f"Corrected fact: {verdict.corrected_fact}",
            f"Lesson: {verdict.lesson}",
            "",
            CRITIC_INSTRUCTIONS,
            CRITIC_SCHEMA_LINE,
        ]
    )
    return "\n".join(parts)


def parse_critic_response(
    raw: str | None,
) -> tuple[CriticVerdict | None, CriticReject | None]:
    """Parse the critic's reply; exactly one element is non-None.

    Any missing or non-boolean judgement is ``unparseable`` (fail closed).
    """
    if raw is None:
        return None, "llm_error"
    obj = parse_json_object(raw)
    if obj is None:
        return None, "unparseable"
    flags: list[bool] = []
    for key in ("supported", "scope_ok", "durable", "misleading"):
        value = obj.get(key)
        if not isinstance(value, bool):
            return None, "unparseable"
        flags.append(value)
    reason = obj.get("reason")
    reason_text = reason.strip()[:CRITIC_REASON_MAX] if isinstance(reason, str) else ""
    supported, scope_ok, durable, misleading = flags
    return (
        CriticVerdict(
            supported=supported,
            scope_ok=scope_ok,
            durable=durable,
            misleading=misleading,
            reason=reason_text,
        ),
        None,
    )


# --- resolution builders -----------------------------------------------------


def _segment_args(command: str, cls: str) -> list[str] | None:
    """Args-after-class of the first segment of *command* classed *cls*."""
    for seg_class, seg_args in bash_segments(command):
        if seg_class == cls:
            return seg_args
    return None


def shape1_args_prefix(failed: str, success: str, cls: str) -> str | None:
    """The shortest args prefix of *failed* that *success* does not share.

    F and S are the args after *cls* (flags dropped) of the first segment of
    each command classed *cls*. The prefix is ``' '.join(F[:k])`` for the
    smallest k in 1..min(3, len(F)) with ``F[:k] != S[:k]``. None when either
    segment is missing, F is empty, or no such k exists (the success command
    would match the prefix too, so the cue could not tell them apart).
    """
    f_args = _segment_args(failed, cls)
    s_args = _segment_args(success, cls)
    if not f_args or s_args is None:
        return None
    for k in range(1, min(SHAPE1_ARGS_PREFIX_MAX_TOKENS, len(f_args)) + 1):
        if f_args[:k] != s_args[:k]:
            return " ".join(f_args[:k])
    return None


def shape1_cue(
    wrong_belief: str,
    success_command: str,
    cue_target_class: str | None = None,
) -> dict[str, str] | None:
    """The precise Bash cue for a shape-1 failure, or None.

    *wrong_belief* is the failed command and *success_command* the later
    command of the same class that succeeded (shape 1's ``corrected_fact``).
    *cue_target_class* is the class the detector paired on; when absent the
    class of the whole wrong belief is used. The class must be a fixed point,
    and the cue carries the ``args_prefix`` from :func:`shape1_args_prefix`;
    without one there is no cue, so the lesson reaches the slice only.
    """
    tc = cue_target_class or target_class("Bash", wrong_belief)
    if tc == "" or target_class("Bash", tc) != tc:
        return None
    prefix = shape1_args_prefix(wrong_belief, success_command, tc)
    if not prefix:
        return None
    return {"tool": "Bash", "target_class": tc, "args_prefix": prefix}


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
        cue = shape1_cue(
            wrong_belief,
            _output_str(candidate, "corrected_fact"),
            _output_str(candidate, "cue_target_class"),
        )
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


def _cue_tuple(cue: Mapping[str, Any] | None) -> tuple[str, str, str]:
    c = cue or {}
    out = []
    for key in ("tool", "target_class", "args_prefix"):
        value = c.get(key, "")
        out.append(value if isinstance(value, str) else "")
    return out[0], out[1], out[2]


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
            and (r.cue_tool, r.cue_target_class, r.cue_args_prefix) == key
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
    if _cue_tuple(rc if isinstance(rc, dict) else None) != _cue_tuple(cue):
        return "cue_mismatch"
    return None


def stored_mode(hints: Mapping[str, object]) -> str | None:
    """``hints.surprise_capture.mode`` when it is a known session mode."""
    block = hints.get(SURPRISE_HINT_KEY)
    if not isinstance(block, dict):
        return None
    mode = block.get("mode")
    return mode if isinstance(mode, str) and mode in SURPRISE_MODES else None


def merged_surprise_hint(
    hints: Mapping[str, object],
    candidate: SurpriseCandidate,
    *,
    new: bool = False,
    mode: str | None = None,
) -> dict[str, Any]:
    """The ``hints.surprise_capture`` value after folding in *candidate*.

    The shape and the session *mode* (``interactive``/``headless``, from the
    ``turn_events`` row of the candidate's last turn event) are written only
    for a *new* entry; a merge keeps the stored values (or their absence) and
    never overwrites or adds them. An unknown or missing mode is left absent.
    """
    sessions, ids, event_ids = surprise_hint(hints)
    block = hints.get(SURPRISE_HINT_KEY)
    shape_part: dict[str, Any] = {}
    if new:
        shape_part = {"shape": candidate.shape}
        if mode in SURPRISE_MODES:
            shape_part["mode"] = mode
    else:
        if isinstance(block, dict) and "shape" in block:
            shape_part = {"shape": block["shape"]}
        kept = stored_mode(hints)
        if kept is not None:
            shape_part["mode"] = kept
    return {
        **shape_part,
        "sessions": [*sessions, candidate.session_id][-SURPRISE_HINT_LIST_CAP:],
        "candidate_ids": [*ids, candidate.id][-SURPRISE_HINT_LIST_CAP:],
        "event_ids": (event_ids + candidate.turn_event_ids[-1:])[
            -SURPRISE_EVENT_IDS_CAP:
        ],
    }
