"""Surprise capture: pure detectors for corrected beliefs in turn digests.

Shape 1 (deterministic): a failed Bash call later succeeded by a different
command of the same class. Shape 2 (one model call): the human's next message
says a claim from the assistant's previous turn was wrong. Shape 3 (one model
call): within one turn, after at least one tool result, the agent reaches a
corrected understanding or a root cause backed by tool output; no earlier wrong
claim by the agent is needed.
Shape 4 (deterministic): a harness_correction the harness reported as resolved,
whose resolving tool calls are found in the session up to the current turn,
becomes a candidate; other corrections are recorded only.

No I/O here: the drain in ``kb_service.surprise_worker`` reads digests, calls
the model and stores the results.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard

from kb_core.cues import bash_segments, target_class
from kb_core.llm.json_parser import parse_json_object

DetectionOutcome = Literal[
    "candidate",
    "not_applicable",
    "no_llm",
    "llm_error",
    "unparseable",
    "invalid_fields",
    "no_surprise",
    "low_confidence",
    "ungrounded",
]
ParseReject = Literal["llm_error", "unparseable", "invalid_fields", "no_surprise"]

# bump on ANY change to SURPRISE_DETECTOR_SYSTEM, DETECTOR_SCHEMA_LINE,
# build_shape2_prompt, build_shape3_prompt, render_items, SHAPE3_INSTRUCTIONS,
# SHAPE3_FIELDS, shape3_skip_reason, shape3_prompt_details, parse_detector_response,
# evidence_grounded, the DETECTOR_MIN_CONFIDENCE default, SHAPE1_IGNORED_CLASSES,
# the shape-1 pairing rule or detect_shape4; the effective KB_SURPRISE_MIN_CONFIDENCE
# threshold is recorded per row in surprise_detections.details.min_confidence
# instead
SURPRISE_DETECTOR_VERSION: int = 4
SHAPE1_DETECTOR_MODEL = "rule:shape1"
SHAPE4_DETECTOR_MODEL = "rule:shape4"
DETECTOR_MIN_CONFIDENCE = 0.7
# per-shape defaults (shape 2 recall gain; shape 3 must not drop)
DETECTOR_MIN_CONFIDENCE_BY_SHAPE: dict[int, float] = {2: 0.5, 3: 0.7}
WRONG_BELIEF_MAX = 500
CORRECTED_FACT_MAX = 500
EVIDENCE_EXCERPT_MAX = 1500
RAW_RESPONSE_EXCERPT_MAX = 2000
SHAPE1_IGNORED_CLASSES = frozenset(
    {
        "cat",
        "ls",
        "head",
        "tail",
        "grep",
        "rg",
        "find",
        "which",
        "test",
        "echo",
        "pwd",
        "stat",
        "wc",
        "file",
    }
)
SURPRISE_DETECTOR_SYSTEM = (
    "You detect corrected beliefs in coding-agent transcripts."
    " Reply with exactly one JSON object and nothing else."
)
DETECTOR_SCHEMA_LINE = (
    '{"surprise": true|false, "wrong_belief": str, "corrected_fact": str,'
    ' "evidence_excerpt": str, "confidence": number between 0 and 1}'
)


@dataclass(frozen=True)
class TurnDigest:
    """One turn of one session, as the detectors see it."""

    event_id: str
    session_id: str
    project: str
    turn_index: int
    user_prompt: str | None
    items: list[dict[str, Any]]
    final_message: str | None
    truncated: bool
    ts: str


@dataclass(frozen=True)
class DetectorVerdict:
    """A parsed positive detector response (before threshold and grounding)."""

    wrong_belief: str
    corrected_fact: str
    evidence_excerpt: str
    confidence: float

    def to_output(self) -> dict[str, Any]:
        """Return the stored ``detector_output`` dict (exactly four keys)."""
        return {
            "wrong_belief": self.wrong_belief,
            "corrected_fact": self.corrected_fact,
            "evidence_excerpt": self.evidence_excerpt,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NewCandidate:
    """A candidate about to be inserted into ``surprise_candidates``."""

    shape: int
    turn_event_ids: list[str]
    detector_model: str
    detector_output: dict[str, Any]


@dataclass(frozen=True)
class SurpriseCandidate:
    """A stored ``surprise_candidates`` row."""

    id: int
    shape: int
    session_id: str
    project: str
    turn_event_ids: list[str]
    detector_model: str
    detector_output: dict[str, Any]
    status: str
    entry_id: str | None
    created_at: str


@dataclass(frozen=True)
class DistillResult:
    """What the distiller wrote to the KB."""

    entries_written: list[str] = field(default_factory=list)
    entries_merged: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Shape1Result:
    """Shape-1 candidates for the current digest plus its counters."""

    candidates: list[NewCandidate]
    stats: dict[str, int]


@dataclass(frozen=True)
class DetectionRecord:
    """One detection decision, stored as a ``surprise_detections`` row."""

    shape: int
    outcome: DetectionOutcome
    reason: str
    detector_model: str
    confidence: float | None = None
    candidate: NewCandidate | None = None
    details: dict[str, Any] = field(default_factory=dict)
    raw_response_excerpt: str | None = None
    prompt_chars: int | None = None
    response_chars: int | None = None
    latency_ms: int | None = None


# --- shape 1 -----------------------------------------------------------------


@dataclass(frozen=True)
class _Call:
    event_id: str
    target: str
    classes: tuple[str, ...]
    is_error: bool
    excerpt: str


def detect_shape1(
    session_digests: list[TurnDigest], current_event_id: str
) -> Shape1Result:
    """Pair a failed Bash command with a later, different success of its class.

    *session_digests* holds every digest of one session up to and including
    the current one, ascending ``turn_index``. Calls pair with results only
    within the same digest. Only pairs whose success is in the current digest
    are emitted, and identical (stripped) commands are not a correction.
    """
    stats = {
        "bash_calls": 0,
        "failures": 0,
        "calls_without_result": 0,
        "dropped_ignored_class": 0,
        "pairs": 0,
        "dropped_identical": 0,
    }
    calls: list[_Call] = []
    for digest in session_digests:
        is_current = digest.event_id == current_event_id
        results: dict[str, dict[str, Any]] = {}
        for item in digest.items:
            if item.get("kind") == "tool_result":
                results.setdefault(str(item.get("tool_use_id")), item)
        for item in digest.items:
            if item.get("kind") != "tool_call":
                continue
            result = results.get(str(item.get("tool_use_id")))
            if result is None:
                if is_current:
                    stats["calls_without_result"] += 1
                continue
            if item.get("tool") != "Bash":
                continue
            target = item.get("target") or ""
            is_error = result.get("is_error") is True
            if is_current:
                stats["bash_calls"] += 1
                if is_error:
                    stats["failures"] += 1
            segment_classes = [c for c, _ in bash_segments(target)]
            if not segment_classes:
                fallback = item.get("target_class") or target_class("Bash", target)
                segment_classes = [fallback] if fallback else []
            classes = tuple(
                dict.fromkeys(
                    c for c in segment_classes if c not in SHAPE1_IGNORED_CLASSES
                )
            )
            if not classes:
                if is_current:
                    stats["dropped_ignored_class"] += 1
                continue
            calls.append(
                _Call(
                    event_id=digest.event_id,
                    target=target,
                    classes=classes,
                    is_error=is_error,
                    excerpt=result.get("excerpt") or "",
                )
            )

    candidates: list[NewCandidate] = []
    pending_failure: dict[str, _Call] = {}
    for call in calls:
        if call.is_error:
            for cls in call.classes:
                pending_failure[cls] = call
            continue
        shared = next((c for c in call.classes if c in pending_failure), None)
        if shared is None:
            continue
        failure = pending_failure[shared]
        for cls in failure.classes:
            if pending_failure.get(cls) is failure:
                del pending_failure[cls]
        if call.event_id != current_event_id:
            continue
        stats["pairs"] += 1
        if failure.target.strip() == call.target.strip():
            stats["dropped_identical"] += 1
            continue
        event_ids = list(dict.fromkeys([failure.event_id, call.event_id]))
        output: dict[str, Any] = {
            "wrong_belief": failure.target[:WRONG_BELIEF_MAX],
            "corrected_fact": call.target[:CORRECTED_FACT_MAX],
            "evidence_excerpt": failure.excerpt[:EVIDENCE_EXCERPT_MAX],
            "confidence": 1.0,
        }
        # Only recorded when the cue cannot be re-derived from wrong_belief.
        if shared != target_class("Bash", output["wrong_belief"]):
            output["cue_target_class"] = shared
        candidates.append(
            NewCandidate(
                shape=1,
                turn_event_ids=event_ids,
                detector_model=SHAPE1_DETECTOR_MODEL,
                detector_output=output,
            )
        )
    return Shape1Result(candidates=candidates, stats=stats)


# --- shape 2 -----------------------------------------------------------------


def _assistant_texts(digest: TurnDigest) -> list[str]:
    return [
        str(item.get("text") or "")
        for item in digest.items
        if item.get("kind") == "assistant_text" and str(item.get("text") or "").strip()
    ]


def shape2_skip_reason(prev: TurnDigest | None, cur: TurnDigest) -> str | None:
    """Return why shape 2 does not apply to *cur*, or None when it does."""
    if prev is None and cur.turn_index == 0:
        return "no_prev"
    if prev is None or prev.turn_index != cur.turn_index - 1:
        return "turn_gap"
    if (cur.user_prompt or "").strip() == "":
        return "no_user_prompt"
    if (prev.final_message or "").strip() == "" and not _assistant_texts(prev):
        return "no_prev_text"
    return None


def build_shape2_prompt(prev: TurnDigest, cur: TurnDigest) -> str:
    """Prompt asking whether the human corrected the assistant's prior claim."""
    parts = ["The assistant's previous turn:"]
    parts.extend(_assistant_texts(prev))
    if (prev.final_message or "").strip():
        parts.append(f"Final message: {prev.final_message}")
    parts.append("")
    parts.append("The human's next message:")
    parts.append(cur.user_prompt or "")
    parts.append("")
    parts.append(
        "Set surprise to true only when the human's message says a factual"
        " claim or assumption from the assistant's previous turn was wrong."
        " Set it to false for new requests, preference or style changes,"
        " approvals and follow-up questions."
    )
    parts.append(
        "wrong_belief is the assistant's claim; corrected_fact is what the"
        " human says is true. evidence_excerpt must be copied verbatim from"
        " the human message, with no ellipses."
    )
    parts.append(DETECTOR_SCHEMA_LINE)
    return "\n".join(parts)


# --- shape 3 -----------------------------------------------------------------


def _text_after_first_result(cur: TurnDigest) -> bool:
    first: int | None = None
    for idx, item in enumerate(cur.items):
        kind = item.get("kind")
        if kind == "tool_result" and first is None:
            first = idx
        elif (
            kind == "assistant_text"
            and first is not None
            and str(item.get("text") or "").strip()
        ):
            return True
    return False


def shape3_skip_reason(cur: TurnDigest) -> str | None:
    """Return why shape 3 does not apply to *cur*, or None when it does."""
    if not any(item.get("kind") == "tool_result" for item in cur.items):
        return "no_tool_result"
    if (cur.final_message or "").strip() != "" or _text_after_first_result(cur):
        return None
    return "no_text_after_result"


def _shape3_final(cur: TurnDigest) -> str:
    final = (cur.final_message or "").strip()
    texts = _assistant_texts(cur)
    last = texts[-1].strip() if texts else ""
    return final if final != "" and final != last else ""


def shape3_prompt_details(cur: TurnDigest) -> dict[str, Any]:
    """Return which part of the scope rule admits *cur* and whether its final message is rendered."""  # noqa: E501
    return {
        "shape3_scope": (
            "text_after_result"
            if _text_after_first_result(cur)
            else "final_message_only"
        ),
        "final_rendered": _shape3_final(cur) != "",
    }


def shape3_reasoning_details(cur: TurnDigest) -> dict[str, Any]:
    """Return reasoning-item counts for *cur*; never affects a verdict."""
    reasoning = [i for i in cur.items if i.get("kind") == "reasoning"]
    first: int | None = None
    after = False
    for idx, item in enumerate(cur.items):
        kind = item.get("kind")
        if kind == "tool_result" and first is None:
            first = idx
        elif (
            kind == "reasoning"
            and first is not None
            and str(item.get("text") or "").strip()
        ):
            after = True
            break
    return {
        "reasoning_items": len(reasoning),
        "reasoning_chars": sum(len(str(i.get("text") or "")) for i in reasoning),
        "reasoning_after_result": after,
    }


def render_items(items: list[dict[str, Any]]) -> str:
    """Render a turn's items one line each; unknown kinds are skipped."""
    lines: list[str] = []
    for item in items:
        kind = item.get("kind")
        if kind == "assistant_text":
            lines.append(f"[assistant] {item.get('text') or ''}")
        elif kind == "tool_call":
            lines.append(f"[tool_call {item.get('tool')}] {item.get('target') or ''}")
        elif kind == "tool_result":
            err = "true" if item.get("is_error") is True else "false"
            lines.append(f"[tool_result error={err}] {item.get('excerpt') or ''}")
        elif kind == "reasoning":
            lines.append(f"[reasoning] {item.get('text') or ''}")
    return "\n".join(lines)


SHAPE3_INSTRUCTIONS = (
    "Set surprise to true only when, in this turn, the agent reached a corrected"
    " understanding or a root cause, supported by a tool result in this turn,"
    " that contradicts what had been believed or indicated before, whether by"
    " the agent itself earlier, a code comment, documentation, a config file,"
    " an error message or the apparent state of the environment. The agent's"
    " own conclusion after investigating counts as the trigger (for example"
    " 'Root cause: ...', 'that was wrong', 'it turns out', 'actually'), even"
    " when the agent never stated the wrong belief itself. Set it to false for"
    " routine progress where nothing that was believed or indicated turned out"
    " to be wrong: a failing test fixed by an ordinary code change, a planned"
    " edit, reading files to learn what they contain, or a retry after a"
    " transient error."
)
SHAPE3_FIELDS = (
    "wrong_belief is what was believed or indicated before, naming its source"
    " (for example: the README says the service listens on port 8000)."
    " corrected_fact is what this turn established is actually true, as one"
    " self-contained sentence. evidence_excerpt is the tool output that shows"
    " corrected_fact is true, copied verbatim from the text of one tool result"
    " above without the bracketed label that starts its line, never from"
    " assistant text, with no ellipses."
)


def build_shape3_prompt(cur: TurnDigest) -> str:
    """Prompt asking whether the agent reached a corrected understanding or root cause in this turn."""  # noqa: E501
    parts = ["One turn of a coding agent, in order:", render_items(cur.items)]
    if shape3_prompt_details(cur)["final_rendered"]:
        parts.append(f"[assistant final] {_shape3_final(cur)}")
    parts.extend(["", SHAPE3_INSTRUCTIONS, SHAPE3_FIELDS, DETECTOR_SCHEMA_LINE])
    return "\n".join(parts)


# --- shape 4 -----------------------------------------------------------------


def detect_shape4(
    session_digests: list[TurnDigest], current_event_id: str
) -> list[DetectionRecord]:
    """One record per harness_correction item in the current digest, no model call.

    resolved_by ids are matched against tool_call items in the session's digests
    up to the current turn (first occurrence wins), so a forward reference stays
    unmatched. Harness, mode and trigger are ignored.
    """
    cur = next((d for d in session_digests if d.event_id == current_event_id), None)
    if cur is None:
        return []
    walk = sorted(
        (d for d in session_digests if d.turn_index <= cur.turn_index),
        key=lambda d: d.turn_index,
    )
    calls: dict[str, tuple[str, int, int, str, str]] = {}
    results: dict[str, bool] = {}
    for d in walk:
        for pos, it in enumerate(d.items):
            kind = it.get("kind")
            tid = it.get("tool_use_id")
            if not isinstance(tid, str):
                continue
            if kind == "tool_call" and tid not in calls:
                calls[tid] = (
                    d.event_id,
                    d.turn_index,
                    pos,
                    str(it.get("tool") or ""),
                    str(it.get("target") or ""),
                )
            elif kind == "tool_result" and tid not in results:
                results[tid] = it.get("is_error") is True
    records: list[DetectionRecord] = []
    for item in cur.items:
        if item.get("kind") != "harness_correction":
            continue
        records.append(_shape4_record(cur, item, calls, results))
    return records


def _shape4_record(
    cur: TurnDigest,
    item: dict[str, Any],
    calls: dict[str, tuple[str, int, int, str, str]],
    results: dict[str, bool],
) -> DetectionRecord:
    trigger = str(item.get("trigger") or "")
    detail = str(item.get("detail") or "").strip()
    resolved_by = item.get("resolved_by")
    if not isinstance(resolved_by, list):
        resolved_by = []
    ids = list(dict.fromkeys(i for i in resolved_by if isinstance(i, str) and i))
    matched = [i for i in ids if i in calls]
    resolved = item.get("resolved") is True
    last_error: bool | None = None
    if matched:
        i_last = max(matched, key=lambda i: (calls[i][1], calls[i][2]))
        last_error = results.get(i_last)
    details: dict[str, Any] = {
        "trigger": trigger,
        "resolved": resolved,
        "resolved_by_count": len(ids),
        "resolved_by_matched": len(matched),
        "resolved_by_errored": sum(1 for i in matched if results.get(i) is True),
        "last_resolving_error": last_error,
        "resolved_by_tools": sorted({calls[i][3] for i in matched}),
    }
    reject: tuple[DetectionOutcome, str] | None = None
    if not resolved:
        reject = ("not_applicable", "unresolved")
    elif detail == "":
        reject = ("not_applicable", "empty_detail")
    elif not ids:
        reject = ("not_applicable", "no_resolving_actions")
    elif not matched:
        reject = ("ungrounded", "resolved_by_unmatched")
    if reject is not None:
        return DetectionRecord(
            shape=4,
            outcome=reject[0],
            reason=reject[1],
            detector_model=SHAPE4_DETECTOR_MODEL,
            details=details,
        )
    corrected_fact = "; ".join(f"{calls[i][3]} {calls[i][4]}".strip() for i in matched)[
        :CORRECTED_FACT_MAX
    ]
    earlier = sorted(
        {(calls[i][1], calls[i][0]) for i in matched if calls[i][0] != cur.event_id}
    )
    turn_event_ids = list(dict.fromkeys(e for _, e in earlier))
    turn_event_ids.append(cur.event_id)
    output = {
        "wrong_belief": detail[:WRONG_BELIEF_MAX],
        "corrected_fact": corrected_fact,
        "evidence_excerpt": detail[:EVIDENCE_EXCERPT_MAX],
        "confidence": 1.0,
        "trigger": trigger,
        "resolved_by": matched,
    }
    return DetectionRecord(
        shape=4,
        outcome="candidate",
        reason="",
        detector_model=SHAPE4_DETECTOR_MODEL,
        confidence=1.0,
        candidate=NewCandidate(4, turn_event_ids, SHAPE4_DETECTOR_MODEL, output),
        details=details,
    )


# --- parsing and grounding ---------------------------------------------------


def _valid_confidence(c: Any) -> TypeGuard[int | float]:
    return (
        isinstance(c, (int, float))
        and not isinstance(c, bool)
        and math.isfinite(c)
        and 0.0 <= c <= 1.0
    )


def parse_detector_response(
    raw: str | None,
) -> tuple[DetectorVerdict | None, ParseReject | None]:
    """Parse a detector reply; exactly one element of the result is None.

    Does not apply the confidence threshold.
    """
    if raw is None:
        return None, "llm_error"
    obj = parse_json_object(raw)
    if obj is None:
        return None, "unparseable"
    if obj.get("surprise") is not True:
        return None, "no_surprise"
    strings: list[str] = []
    for key in ("wrong_belief", "corrected_fact", "evidence_excerpt"):
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, "invalid_fields"
        strings.append(value.strip())
    confidence = obj.get("confidence")
    if not _valid_confidence(confidence):
        return None, "invalid_fields"
    return (
        DetectorVerdict(
            wrong_belief=strings[0][:WRONG_BELIEF_MAX],
            corrected_fact=strings[1][:CORRECTED_FACT_MAX],
            evidence_excerpt=strings[2][:EVIDENCE_EXCERPT_MAX],
            confidence=float(confidence),
        ),
        None,
    )


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def evidence_grounded(evidence: str, sources: list[str]) -> bool:
    """True iff the (whitespace/case-normalized) evidence occurs in a source."""
    needle = _normalize(evidence)
    if not needle:
        return False
    return any(needle in _normalize(src) for src in sources)
