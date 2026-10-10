"""Write policy: pure helpers that distill queued stores (shape-5 candidates).

A ``kb_store`` from a headless or autonomous surface is queued by
``kb_service.write_policy`` as a shape-5 ``surprise_candidates`` row whose
``detector_output.request`` holds the agent's entry. The candidate pipeline
(``kb_service.surprise_worker._decide_store_one``) runs a distiller, a critic
and a near-duplicate check before writing it. This module builds those
prompts, parses the distiller reply and builds the entry to write.

No I/O and no logging here.
"""

from datetime import UTC, datetime
from typing import Any

from kb_core.llm.json_parser import parse_json_object
from kb_core.models.entry import EntryType
from kb_core.ttl import compute_expires_at

from kb_service.surprise import SurpriseCandidate
from kb_service.surprise_distill import (
    CRITIC_SCHEMA_LINE,
    DISTILL_CONFIDENCE_LEVEL,
    DISTILL_LONG_TITLE_MAX,
    DISTILL_SHORT_TITLE_MAX,
    LESSON_CLASS_PROMPT,
    LESSON_CLASSES,
    SHAPE_DESCRIPTIONS,
    DistillReject,
    DistillVerdict,
    lesson_expires_at,
)

STORE_CANDIDATE_SHAPE: int = 5
STORE_CANDIDATE_DETECTOR_MODEL = "write-policy"

# bump STORE_DISTILLER_VERSION on ANY change to STORE_DISTILLER_SYSTEM,
# STORE_DISTILLER_INSTRUCTIONS, STORE_DISTILLER_SCHEMA_LINE,
# STORE_PROMPT_DETAILS_MAX, SHAPE_DESCRIPTIONS[5], LESSON_CLASSES,
# LESSON_CLASS_PROMPT, build_store_distill_prompt, parse_store_distill_response,
# build_store_knowledge_details, build_store_kwargs or the W1-W9 order
STORE_DISTILLER_VERSION: int = 1

# bump STORE_CRITIC_VERSION on ANY change to STORE_CRITIC_SYSTEM,
# STORE_CRITIC_INSTRUCTIONS, CRITIC_SCHEMA_LINE, build_store_critic_prompt or
# parse_critic_response
STORE_CRITIC_VERSION: int = 1

STORE_PROMPT_DETAILS_MAX = 8000
WRITE_POLICY_TAG = "write-policy"
WRITE_POLICY_HINT_KEY = "write_policy"

STORE_DISTILLER_SYSTEM = (
    "You review knowledge-base entries that coding agents tried to store from"
    " unattended sessions and decide which are durable. Reply with exactly one"
    " JSON object and nothing else."
)

STORE_DISTILLER_INSTRUCTIONS = (
    "Decide whether this entry is durable knowledge for future sessions in this"
    " project. Set durable to false when it only describes this session: a"
    " progress report, a task status, a work-in-progress state, a transient"
    " error, a file or branch that exists only in this checkout, or a one-off"
    " preference; then set why to a short reason and leave the other fields"
    " empty. A durable entry states a decision, convention, lesson or fact about"
    " the project, its tools or its environment that will still be true for a"
    " fresh session tomorrow. If it is durable, set short_title (at most 80"
    " characters) and long_title (at most 200 characters) to state the entry's"
    " main point using only facts present in the entry; the agent's own titles"
    " may be kept unchanged. Never include secrets, tokens, passwords or"
    " credentials in any field."
)

STORE_DISTILLER_SCHEMA_LINE = (
    '{"durable": true|false, "why": str, "short_title": str, "long_title": str,'
    ' "lesson_class": ' + "|".join(f'"{c}"' for c in LESSON_CLASSES) + "}"
)

STORE_CRITIC_SYSTEM = (
    "You review knowledge-base entries that coding agents tried to store from"
    " unattended sessions. Reply with exactly one JSON object and nothing else."
)

STORE_CRITIC_INSTRUCTIONS = (
    "Judge the entry above strictly. supported is true only when the titles"
    " claim nothing the details do not state and the details present each"
    " claim as established; a claim stated hedged or speculatively (for example"
    " 'I think', 'maybe', 'probably', 'not sure', or a question) makes it false."
    " scope_ok is true only when no claim is stretched beyond what the details"
    " describe into a broader or general rule. durable is true only when the"
    " entry will still be true for a fresh session tomorrow; a progress report,"
    " a task status, a work-in-progress state, or something described as about"
    " to change makes it false. misleading is true when a future agent acting on"
    " the entry would plausibly do the wrong thing. reason is one short sentence"
    " explaining the judgement."
)


def _req(c: SurpriseCandidate) -> dict[str, Any]:
    req = c.detector_output.get("request")
    return req if isinstance(req, dict) else {}


def _etype(req: dict[str, Any]) -> str:
    return str(req.get("entry_type") or "factual_reference")


def _surface(c: SurpriseCandidate) -> str:
    return str(c.detector_output.get("surface") or "")


def _details(req: dict[str, Any]) -> str:
    return str(req["knowledge_details"])[:STORE_PROMPT_DETAILS_MAX]


def build_store_distill_prompt(c: SurpriseCandidate) -> str:
    """The store distiller's user prompt for queued entry *c*."""
    req = _req(c)
    return "\n".join(
        [
            SHAPE_DESCRIPTIONS[STORE_CANDIDATE_SHAPE],
            f"Project: {c.project}",
            f"Entry type: {_etype(req)}",
            f"Short title: {req['short_title']!s}",
            f"Long title: {req['long_title']!s}",
            f"Details: {_details(req)}",
            STORE_DISTILLER_INSTRUCTIONS,
            LESSON_CLASS_PROMPT,
            STORE_DISTILLER_SCHEMA_LINE,
        ]
    )


def parse_store_distill_response(
    raw: str | None,
) -> tuple[DistillVerdict | None, DistillReject | None]:
    """Parse the store distiller's reply; exactly one element is non-None."""
    if raw is None:
        return None, "llm_error"
    obj = parse_json_object(raw)
    if obj is None:
        return None, "unparseable"
    if obj.get("durable") is not True:
        return None, "not_durable"
    st = obj.get("short_title")
    lt = obj.get("long_title")
    if not isinstance(st, str) or not st.strip():
        return None, "invalid_fields"
    if not isinstance(lt, str) or not lt.strip():
        return None, "invalid_fields"
    lesson_class = obj.get("lesson_class")
    if not isinstance(lesson_class, str) or lesson_class not in LESSON_CLASSES:
        return None, "invalid_fields"
    return (
        DistillVerdict(
            short_title=st.strip()[:DISTILL_SHORT_TITLE_MAX],
            long_title=lt.strip()[:DISTILL_LONG_TITLE_MAX],
            corrected_fact="",
            lesson="",
            lesson_class=lesson_class,
        ),
        None,
    )


def build_store_critic_prompt(c: SurpriseCandidate, verdict: DistillVerdict) -> str:
    """The store critic's user prompt: the distilled titles and the details."""
    req = _req(c)
    return "\n".join(
        [
            SHAPE_DESCRIPTIONS[STORE_CANDIDATE_SHAPE],
            f"Project: {c.project}",
            f"Entry type: {_etype(req)}",
            "",
            "Entry:",
            f"Short title: {verdict.short_title}",
            f"Long title: {verdict.long_title}",
            f"Details: {_details(req)}",
            "",
            STORE_CRITIC_INSTRUCTIONS,
            CRITIC_SCHEMA_LINE,
        ]
    )


def build_store_knowledge_details(c: SurpriseCandidate) -> str:
    """The written entry's details: the agent's details plus a provenance line."""
    req = _req(c)
    return (
        f"{req['knowledge_details']}\n\nQueued by the write policy from a"
        f" {_surface(c)} surface (candidate {c.id}) and written after the"
        " distiller and critic reviewed it."
    )


def build_store_kwargs(
    c: SurpriseCandidate, verdict: DistillVerdict, *, now: datetime | None = None
) -> dict[str, Any]:
    """The ``kb.store`` keyword arguments for writing queued entry *c*."""
    now = now or datetime.now(UTC)
    req = _req(c)
    surface = _surface(c)
    source_context = f"write_policy candidate {c.id} ({surface} surface)"
    if req.get("source_context"):
        source_context += f"; {req['source_context']}"
    raw_conf = req.get("confidence_level")
    confidence = min(
        raw_conf if raw_conf is not None else 0.9, DISTILL_CONFIDENCE_LEVEL
    )
    tags = list(
        dict.fromkeys(
            [
                *(req.get("tags") or []),
                WRITE_POLICY_TAG,
                f"surface:{surface}",
                f"lesson-class:{verdict.lesson_class}",
            ]
        )
    )
    hints = {
        **(req.get("hints") or {}),
        WRITE_POLICY_HINT_KEY: {
            "candidate_id": c.id,
            "surface": surface,
            "harness": c.detector_output.get("harness", ""),
            "engine": c.detector_output.get("engine", ""),
            "api_key_id": c.detector_output.get("api_key_id"),
            "op": c.detector_output.get("op"),
            "lesson_class": verdict.lesson_class,
        },
    }
    expires_at = lesson_expires_at(now)
    if req.get("ttl"):
        expires_at = min(expires_at, compute_expires_at(str(req["ttl"]), now))
    return {
        "short_title": verdict.short_title,
        "long_title": verdict.long_title,
        "knowledge_details": build_store_knowledge_details(c),
        "entry_type": EntryType(_etype(req)),
        "project_ref": c.project,
        "source_context": source_context,
        "confidence_level": confidence,
        "tags": tags,
        "hints": hints,
        "contributor": c.detector_output.get("contributor"),
        "team": c.detector_output.get("team"),
        "sensitivity": req.get("sensitivity"),
        "expires_at": expires_at,
        "enrich": False,
    }
