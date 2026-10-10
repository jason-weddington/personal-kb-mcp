"""store_distill: pinned literals, prompts, parser and entry builder (shape 5)."""

import hashlib
import inspect
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from kb_core.models.entry import EntryType

from kb_service import database
from kb_service import store_distill as sd
from kb_service.store_distill import (
    STORE_CANDIDATE_SHAPE,
    build_store_critic_prompt,
    build_store_distill_prompt,
    build_store_knowledge_details,
    build_store_kwargs,
    parse_store_distill_response,
)
from kb_service.surprise import SurpriseCandidate
from kb_service.surprise_distill import (
    CRITIC_SCHEMA_LINE,
    LESSON_CLASS_PROMPT,
    LESSON_CLASSES,
    SHAPE_DESCRIPTIONS,
    DistillVerdict,
    parse_critic_response,
)

# A mismatch here means a shared input of the store prompts changed: bump
# STORE_DISTILLER_VERSION / STORE_CRITIC_VERSION and update the literal.
_DISTILLER_SHARED_SHA = (
    "72bdc8122c389601d89bcdc0072fbaf80fa8c63f3ee8a78cbbbae22c551dfb89"
)
_CRITIC_SHARED_SHA = "a3866dab2c20dc19bf7aeea79a0abeec8b53f11c73dfcb43f2dd2764f3e888d8"


def _cand(**request: Any) -> SurpriseCandidate:
    req: dict[str, Any] = {
        "short_title": "Gate uses uv",
        "long_title": "The gate runs with uv",
        "knowledge_details": "Run the gate with uv run --frozen pytest.",
        "entry_type": "decision",
        "project_ref": "p",
        "source_context": None,
        "confidence_level": None,
        "tags": None,
        "hints": None,
        "sensitivity": None,
        "ttl": None,
    }
    req.update(request)
    return SurpriseCandidate(
        id=7,
        shape=5,
        session_id="key:k1",
        project="p",
        turn_event_ids=[],
        detector_model="write-policy",
        detector_output={
            "kind": "store",
            "op": "store_batch",
            "batch_index": 0,
            "surface": "autonomous",
            "harness": "talos",
            "engine": "glm-5",
            "api_key_id": "k1",
            "contributor": "alice",
            "team": "core",
            "request": req,
        },
        status="pending",
        entry_id=None,
        created_at="t",
    )


_VERDICT = DistillVerdict(
    short_title="Use uv for the gate",
    long_title="The quality gate runs via uv run --frozen",
    corrected_fact="",
    lesson="",
    lesson_class="project_tooling",
)


def test_pinned_literals() -> None:
    assert sd.STORE_CANDIDATE_SHAPE == 5
    assert sd.STORE_CANDIDATE_DETECTOR_MODEL == "write-policy"
    assert sd.STORE_PROMPT_DETAILS_MAX == 8000
    assert sd.WRITE_POLICY_TAG == "write-policy"
    assert sd.WRITE_POLICY_HINT_KEY == "write_policy"
    assert sd.STORE_DISTILLER_SYSTEM == (
        "You review knowledge-base entries that coding agents tried to store from "
        "unattended sessions and decide which are durable. Reply with exactly one "
        "JSON object and nothing else."
    )
    assert sd.STORE_DISTILLER_INSTRUCTIONS == (
        "Decide whether this entry is durable knowledge for future sessions in this "
        "project. Set durable to false when it only describes this session: a "
        "progress report, a task status, a work-in-progress state, a transient "
        "error, a file or branch that exists only in this checkout, or a one-off "
        "preference; then set why to a short reason and leave the other fields "
        "empty. A durable entry states a decision, convention, lesson or fact about "
        "the project, its tools or its environment that will still be true for a "
        "fresh session tomorrow. If it is durable, set short_title (at most 80 "
        "characters) and long_title (at most 200 characters) to state the entry's "
        "main point using only facts present in the entry; the agent's own titles "
        "may be kept unchanged. Never include secrets, tokens, passwords or "
        "credentials in any field."
    )
    schema = (
        '{"durable": true|false, "why": str, "short_title": str, "long_title": str, '
        '"lesson_class": ' + "|".join(f'"{c}"' for c in LESSON_CLASSES) + "}"
    )
    assert schema == sd.STORE_DISTILLER_SCHEMA_LINE
    assert sd.STORE_CRITIC_SYSTEM == (
        "You review knowledge-base entries that coding agents tried to store from "
        "unattended sessions. Reply with exactly one JSON object and nothing else."
    )
    assert sd.STORE_CRITIC_INSTRUCTIONS == (
        "Judge the entry above strictly. supported is true only when the titles "
        "claim nothing the details do not state and the details present each claim "
        "as established; a claim stated hedged or speculatively (for example 'I "
        "think', 'maybe', 'probably', 'not sure', or a question) makes it false. "
        "scope_ok is true only when no claim is stretched beyond what the details "
        "describe into a broader or general rule. durable is true only when the "
        "entry will still be true for a fresh session tomorrow; a progress report, "
        "a task status, a work-in-progress state, or something described as about "
        "to change makes it false. misleading is true when a future agent acting on "
        "the entry would plausibly do the wrong thing. reason is one short sentence "
        "explaining the judgement."
    )


def test_shape5_registered() -> None:
    assert STORE_CANDIDATE_SHAPE in database.SURPRISE_SHAPES
    assert STORE_CANDIDATE_SHAPE in SHAPE_DESCRIPTIONS


def test_distiller_version_and_shared_inputs() -> None:
    shared = json.dumps(
        [list(LESSON_CLASSES), LESSON_CLASS_PROMPT, SHAPE_DESCRIPTIONS[5]]
    )
    assert hashlib.sha256(shared.encode()).hexdigest() == _DISTILLER_SHARED_SHA
    assert sd.STORE_DISTILLER_VERSION == 1


def test_critic_version_and_shared_inputs() -> None:
    shared = json.dumps([CRITIC_SCHEMA_LINE, inspect.getsource(parse_critic_response)])
    assert hashlib.sha256(shared.encode()).hexdigest() == _CRITIC_SHARED_SHA
    assert sd.STORE_CRITIC_VERSION == 1


def test_bump_comments_present() -> None:
    src = inspect.getsource(sd)
    assert "# bump STORE_DISTILLER_VERSION on ANY change to" in src
    assert "# bump STORE_CRITIC_VERSION on ANY change to" in src
    from kb_service import surprise_distill

    sdsrc = inspect.getsource(surprise_distill)
    assert "store_distill.py: bump STORE_DISTILLER_VERSION too)" in sdsrc
    assert "# bump STORE_CRITIC_VERSION too)" in sdsrc


def test_distill_prompt_lines() -> None:
    c = _cand(knowledge_details="x" * 9000)
    assert build_store_distill_prompt(c).split("\n") == [
        SHAPE_DESCRIPTIONS[5],
        "Project: p",
        "Entry type: decision",
        "Short title: Gate uses uv",
        "Long title: The gate runs with uv",
        "Details: " + "x" * 8000,
        sd.STORE_DISTILLER_INSTRUCTIONS,
        LESSON_CLASS_PROMPT,
        sd.STORE_DISTILLER_SCHEMA_LINE,
    ]


def test_distill_prompt_default_entry_type() -> None:
    c = _cand(entry_type=None)
    assert "Entry type: factual_reference" in build_store_distill_prompt(c)


def test_critic_prompt_lines() -> None:
    assert build_store_critic_prompt(_cand(), _VERDICT).split("\n") == [
        SHAPE_DESCRIPTIONS[5],
        "Project: p",
        "Entry type: decision",
        "",
        "Entry:",
        "Short title: Use uv for the gate",
        "Long title: The quality gate runs via uv run --frozen",
        "Details: Run the gate with uv run --frozen pytest.",
        "",
        sd.STORE_CRITIC_INSTRUCTIONS,
        CRITIC_SCHEMA_LINE,
    ]


def _reply(**kw: Any) -> str:
    obj: dict[str, Any] = {
        "durable": True,
        "why": "",
        "short_title": "T",
        "long_title": "L",
        "lesson_class": "none",
    }
    obj.update(kw)
    return json.dumps(obj)


@pytest.mark.parametrize(
    ("raw", "reject"),
    [
        (None, "llm_error"),
        ("no json here", "unparseable"),
        (_reply(durable=False), "not_durable"),
        (_reply(durable="yes"), "not_durable"),
        (_reply(short_title=" "), "invalid_fields"),
        (_reply(short_title=3), "invalid_fields"),
        (_reply(long_title=""), "invalid_fields"),
        (_reply(lesson_class="bogus"), "invalid_fields"),
        (_reply(lesson_class=None), "invalid_fields"),
    ],
)
def test_parse_rejects(raw: str | None, reject: str) -> None:
    assert parse_store_distill_response(raw) == (None, reject)


def test_parse_accepts_and_trims() -> None:
    verdict, reject = parse_store_distill_response(
        _reply(short_title="  T  ", long_title="x" * 300, lesson_class="quality_gates")
    )
    assert reject is None
    assert verdict == DistillVerdict(
        short_title="T",
        long_title="x" * 200,
        corrected_fact="",
        lesson="",
        lesson_class="quality_gates",
    )


def test_knowledge_details() -> None:
    assert build_store_knowledge_details(_cand()) == (
        "Run the gate with uv run --frozen pytest.\n\nQueued by the write policy"
        " from a autonomous surface (candidate 7) and written after the distiller"
        " and critic reviewed it."
    )


_NOW = datetime(2026, 10, 10, tzinfo=UTC)


def test_store_kwargs() -> None:
    c = _cand(
        tags=["gate", "write-policy"],
        hints={"person": "jason"},
        source_context="agent run",
        sensitivity="internal",
        confidence_level=0.95,
    )
    kw = build_store_kwargs(c, _VERDICT, now=_NOW)
    assert set(kw) == {
        "short_title",
        "long_title",
        "knowledge_details",
        "entry_type",
        "project_ref",
        "source_context",
        "confidence_level",
        "tags",
        "hints",
        "contributor",
        "team",
        "sensitivity",
        "expires_at",
        "enrich",
    }
    assert kw["short_title"] == "Use uv for the gate"
    assert kw["entry_type"] is EntryType.DECISION
    assert kw["project_ref"] == "p"
    assert kw["source_context"] == (
        "write_policy candidate 7 (autonomous surface); agent run"
    )
    assert kw["confidence_level"] == 0.7
    assert kw["tags"] == [
        "gate",
        "write-policy",
        "surface:autonomous",
        "lesson-class:project_tooling",
    ]
    assert kw["hints"] == {
        "person": "jason",
        "write_policy": {
            "candidate_id": 7,
            "surface": "autonomous",
            "harness": "talos",
            "engine": "glm-5",
            "api_key_id": "k1",
            "op": "store_batch",
            "lesson_class": "project_tooling",
        },
    }
    assert (kw["contributor"], kw["team"], kw["sensitivity"]) == (
        "alice",
        "core",
        "internal",
    )
    assert kw["expires_at"] == _NOW + timedelta(days=30)
    assert kw["enrich"] is False


@pytest.mark.parametrize(("given", "want"), [(0.95, 0.7), (0.5, 0.5), (None, 0.7)])
def test_confidence_clamp(given: float | None, want: float) -> None:
    kw = build_store_kwargs(_cand(confidence_level=given), _VERDICT, now=_NOW)
    assert kw["confidence_level"] == pytest.approx(want)
    assert kw["source_context"] == "write_policy candidate 7 (autonomous surface)"


def test_ttl_shorter_and_longer() -> None:
    short = build_store_kwargs(_cand(ttl="7d"), _VERDICT, now=_NOW)
    assert short["expires_at"] == _NOW + timedelta(days=7)
    long = build_store_kwargs(_cand(ttl="90d"), _VERDICT, now=_NOW)
    assert long["expires_at"] == _NOW + timedelta(days=30)


def test_store_kwargs_default_now() -> None:
    kw = build_store_kwargs(_cand(), _VERDICT)
    delta = kw["expires_at"] - (datetime.now(UTC) + timedelta(days=30))
    assert abs(delta.total_seconds()) < 60
