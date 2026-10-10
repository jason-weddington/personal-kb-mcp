"""Offline eval harness for the surprise-capture detector.

Scores the production detector, ``kb_service.surprise_worker.detect_digest``,
per shape on a labelled case file. Every case goes through the same
``TurnDigestRequest`` validation, 64 KiB check and ``redact_turn_digest``
redaction as a live digest before any model sees it. The harness writes
nothing to the KB, GTD or any database; its outputs go to ``--out``, which must
lie outside this repo.

Run from the repo root::

    uv run python scripts/surprise_eval/surprise_eval.py validate --cases PATH
    uv run python scripts/surprise_eval/surprise_eval.py run
        --cases PATH --out DIR --model claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic
from kb_core import cues
from kb_core.llm import anthropic as anthropic_llm
from kb_core.llm import json_parser
from kb_service import config as service_config
from kb_service import surprise, surprise_worker, turn_digest
from kb_service.models import TurnDigestRequest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_REFUSAL = "refusing to write surprise eval data inside the repo"
CASES_REFUSAL = "refusing to read non-synthetic cases from inside the repo"
SYNTHETIC_PREFIX = "syn-"
USER_PREFIX = "[user] "
PRECISION_BAR = 0.9
PRECISION_BAR_MIN_PREDICTED = 100
WILSON_Z = 1.96
DEFAULT_CONCURRENCY = 4
MAX_WARNINGS_PER_ROW = 5

OUTCOMES = (
    "candidate",
    "not_applicable",
    "no_llm",
    "llm_error",
    "unparseable",
    "invalid_fields",
    "no_surprise",
    "low_confidence",
    "ungrounded",
)
ERROR_OUTCOMES = ("llm_error", "unparseable", "invalid_fields")
MODEL_POSITIVE_OUTCOMES = ("candidate", "low_confidence", "ungrounded")
NO_CALL_OUTCOMES = ("not_applicable", "no_llm")

COMMON_CASE_KEYS = frozenset(
    {
        "id",
        "shape",
        "label",
        "hard_negative",
        "expected",
        "note",
        "host",
        "project",
        "label_source",
        "frame",
    }
)
DIGEST_CASE_KEYS = COMMON_CASE_KEYS | {"digests"}
MINED_SHAPE2_KEYS = COMMON_CASE_KEYS | {"prev_final_message", "prev_assistant_texts", "user_prompt"}
MINED_SHAPE3_KEYS = COMMON_CASE_KEYS | {"items"}
DIGEST_KEYS = frozenset(
    {
        "event_id",
        "session_id",
        "turn_index",
        "project",
        "user_prompt",
        "items",
        "final_message",
        "truncated",
        "ts",
    }
)
DIGEST_ITEM_KEYS = {
    "assistant_text": {"kind", "text"},
    "tool_call": {"kind", "tool_use_id", "tool", "target", "target_class"},
    "tool_result": {"kind", "tool_use_id", "is_error", "excerpt"},
}
MINED_ITEM_KEYS = {
    "assistant_text": {"kind", "text"},
    "tool_call": {"kind", "tool", "target"},
    "tool_result": {"kind", "is_error", "excerpt"},
}

RESULT_KEYS = (
    "case_id",
    "shape",
    "label",
    "hard_negative",
    "label_source",
    "frame",
    "model",
    "detector_model",
    "detector_version",
    "outcome",
    "reason",
    "predicted",
    "model_positive",
    "confidence",
    "llm_calls",
    "latency_ms",
    "prompt_chars",
    "response_chars",
    "raw_response_excerpt",
    "detector_output",
    "expected",
    "expected_match",
    "redactions",
    "cross_shape_candidates",
    "prompt_sha256",
    "provider_warnings",
    "anomaly",
    "harness_error",
)
CELL_KEYS = (
    "shape",
    "model",
    "n",
    "positives",
    "negatives",
    "hard_negatives",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "precision_wilson95",
    "recall",
    "recall_wilson95",
    "recall_applicable",
    "recall_applicable_wilson95",
    "f1",
    "model_positive_precision",
    "model_positive_recall",
    "recall_durable",
    "fp_hard_negative",
    "outcomes",
    "errors",
    "lost_to_gates",
    "not_applicable_positives",
    "llm_calls",
    "latency_ms_mean",
    "latency_ms_p95",
    "latency_over_production_timeout",
    "prompt_chars_total",
    "response_chars_total",
    "provider_no_text",
    "provider_failed",
    "provider_exception_types",
    "anomalies",
    "cross_shape_fp",
    "harness_errors",
    "tp_expected_mismatch",
    "label_sources",
    "frames",
    "by_label_source",
    "fp_ids",
    "fn_ids",
    "precision_bar",
)

RESULT_FILES = ("results.jsonl", "report.json", "report.md")
PROVIDER_LOGGER = "kb_core.llm.anthropic"
NO_TEXT_MARKER = "no text block"
FAILED_MARKER = "generation failed"
EXC_MARKER = " exc="
_CUT = 300
_MD_CELL_MAX = 120
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_EXPECTED_KEYS = frozenset({"wrong_belief", "corrected_fact", "durable"})

NOTES = (
    "Detector errors count as negatives, as they do in production.",
    "model+ means the model said surprise before the confidence floor and the grounding check.",
    "Precision and recall are case-level: a true positive is any candidate on a positive case;"
    " only shape 1 checks the extracted belief against expected.",
    "R applicable excludes positives the detector's skip rules made not_applicable; mined cases"
    " may carry the miner's own truncation and context window, so read their shape-3 recall as"
    " a lower bound.",
    'Precision bar: precision >= 0.9 once tp+fp >= 100, our reading of "precision is at least'
    ' 0.9 on 100 labelled positives" (proposals/experience-loop-debate-synthesis.md); it reads'
    " not-gating unless every case in the cell has frame unfiltered and a label_source that is"
    " neither unspecified nor silver.",
)
TABLE_HEADER = (
    "| shape | model | n | pos | tp | fp | fn | tn | precision | P 95% CI | recall | R 95% CI"
    " | R applicable | f1 | model+ P | model+ R | errors | n/a | bar |"
)
DISAGREEMENT_HEADER = (
    "| shape | model | case_id | label | outcome | reason | confidence | wrong_belief"
    " | evidence_excerpt |"
)

_EVAL_LOG: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_EVAL_LOG", default=None
)


class CaseError(ValueError):
    """A case file failed validation."""


@dataclass(frozen=True)
class Case:
    """One validated, redacted eval case."""

    id: str
    shape: int
    label: bool
    hard_negative: bool
    label_source: str
    frame: str
    expected: dict[str, Any] | None
    note: str | None
    digests: list[surprise.TurnDigest]
    redactions: list[str]


def inside_repo(p: str | Path) -> bool:
    """Return True when *p* is REPO_ROOT or lies under it.

    Args:
        p: A path given on the command line.

    Returns:
        True when reading private data from (or writing it to) there must be refused.
    """
    resolved = Path(p).expanduser().resolve()
    return resolved == REPO_ROOT or resolved.is_relative_to(REPO_ROOT)


# --- case loading ------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def mined_to_digests(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a validated mined-form case into digest dicts.

    Args:
        case: A mined-form case that passed the common and mined-form checks.

    Returns:
        Two digests (turns 0 and 1) for shape 2, one digest (turn 0) for shape 3.
    """
    project = case.get("project")
    session_id = case["id"]
    if case["shape"] == 2:
        prev_final = case["prev_final_message"]
        return [
            {
                "session_id": session_id,
                "turn_index": 0,
                "project": project,
                "user_prompt": None,
                "items": [
                    {"kind": "assistant_text", "text": t}
                    for t in case.get("prev_assistant_texts", [])
                    if t.strip()
                ],
                "final_message": prev_final if prev_final.strip() else None,
                "truncated": False,
            },
            {
                "session_id": session_id,
                "turn_index": 1,
                "project": project,
                "user_prompt": case["user_prompt"],
                "items": [],
                "final_message": None,
                "truncated": False,
            },
        ]
    mined = case["items"]
    k = -1
    for idx, item in enumerate(mined):
        if item["kind"] == "assistant_text" and item["text"].startswith(USER_PREFIX):
            k = idx
    user_prompt = mined[k]["text"][len(USER_PREFIX) :] if k >= 0 else None
    items: list[dict[str, Any]] = []
    unpaired: list[str] = []
    n_calls = 0
    n_orphans = 0
    for item in mined[k + 1 :]:
        kind = item["kind"]
        if kind == "assistant_text":
            items.append({"kind": "assistant_text", "text": item["text"]})
        elif kind == "tool_call":
            n_calls += 1
            tool = item["tool"]
            target = item["target"]
            tgt = cues.extract_target(
                tool,
                {
                    "command": target,
                    "file_path": target,
                    "notebook_path": target,
                    "pattern": target,
                },
            )
            call_id = f"t{n_calls}"
            items.append(
                {
                    "kind": "tool_call",
                    "tool_use_id": call_id,
                    "tool": tool,
                    "target": tgt,
                    "target_class": cues.target_class(tool, tgt),
                }
            )
            unpaired.append(call_id)
        else:
            if unpaired:
                result_id = unpaired.pop(0)
            else:
                n_orphans += 1
                result_id = f"orphan{n_orphans}"
            items.append(
                {
                    "kind": "tool_result",
                    "tool_use_id": result_id,
                    "is_error": item["is_error"],
                    "excerpt": item["excerpt"],
                }
            )
    return [
        {
            "session_id": session_id,
            "turn_index": 0,
            "project": project,
            "user_prompt": user_prompt,
            "items": items,
            "final_message": None,
            "truncated": False,
        }
    ]


def _check_common(cid: str, obj: dict[str, Any]) -> None:
    shape = obj.get("shape")
    if not _is_int(shape) or shape not in (1, 2, 3):
        raise CaseError(f"{cid}: shape must be 1, 2 or 3")
    if not isinstance(obj.get("label"), bool):
        raise CaseError(f"{cid}: label must be a bool")
    if "hard_negative" in obj and not isinstance(obj["hard_negative"], bool):
        raise CaseError(f"{cid}: hard_negative must be a bool")
    if obj.get("hard_negative") is True and obj["label"] is True:
        raise CaseError(f"{cid}: hard_negative on a positive case")
    if "expected" in obj:
        if obj["label"] is False:
            raise CaseError(f"{cid}: expected on a negative case")
        exp = obj["expected"]
        if (
            not isinstance(exp, dict)
            or not set(exp) <= _EXPECTED_KEYS
            or not isinstance(exp.get("wrong_belief"), str)
            or not isinstance(exp.get("corrected_fact"), str)
            or ("durable" in exp and not isinstance(exp["durable"], bool))
        ):
            raise CaseError(
                f"{cid}: expected must be {{wrong_belief: str, corrected_fact: str,"
                " durable?: bool}"
            )
    for key in ("note", "host", "project"):
        if key in obj and not isinstance(obj[key], str):
            raise CaseError(f"{cid}: {key} must be a str")
    for key in ("label_source", "frame"):
        if key in obj:
            v = obj[key]
            if not isinstance(v, str) or not 1 <= len(v) <= 64:
                raise CaseError(f"{cid}: {key} must be a str of 1-64 chars")


def _unknown_keys(cid: str, obj: dict[str, Any], allowed: frozenset[str] | set[str]) -> None:
    extra = set(obj) - set(allowed)
    if extra:
        raise CaseError(f"{cid}: unknown keys {sorted(extra)}")


def _check_mined(cid: str, obj: dict[str, Any]) -> None:
    shape = obj["shape"]
    if shape == 1:
        raise CaseError(f"{cid}: shape 1 cases need digests")
    if shape == 2:
        _unknown_keys(cid, obj, MINED_SHAPE2_KEYS)
        for key in ("prev_final_message", "user_prompt"):
            if not isinstance(obj.get(key), str):
                raise CaseError(f"{cid}: {key} must be a str")
        if "prev_assistant_texts" in obj:
            texts = obj["prev_assistant_texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise CaseError(f"{cid}: prev_assistant_texts must be a list of str")
        return
    _unknown_keys(cid, obj, MINED_SHAPE3_KEYS)
    items = obj.get("items")
    if not isinstance(items, list) or not items:
        raise CaseError(f"{cid}: items must be a non-empty list")
    for j, item in enumerate(items):
        if not isinstance(item, dict):
            raise CaseError(f"{cid}: item {j} is not an object")
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in MINED_ITEM_KEYS:
            raise CaseError(f"{cid}: item {j} has an unknown kind")
        if set(item) != MINED_ITEM_KEYS[kind]:
            raise CaseError(f"{cid}: item {j} keys must be {sorted(MINED_ITEM_KEYS[kind])}")
        for key in ("text", "target", "excerpt"):
            if key in item and not isinstance(item[key], str):
                raise CaseError(f"{cid}: item {j} {key} must be a str")
        if "tool" in item and (not isinstance(item["tool"], str) or not item["tool"]):
            raise CaseError(f"{cid}: item {j} tool must be a non-empty str")
        if "is_error" in item and not isinstance(item["is_error"], bool):
            raise CaseError(f"{cid}: item {j} is_error must be a bool")


def _check_digest_structure(cid: str, shape: int, digests: Any) -> None:
    if not isinstance(digests, list) or not digests:
        raise CaseError(f"{cid}: digests must be a non-empty list")
    for i, d in enumerate(digests):
        if not isinstance(d, dict):
            raise CaseError(f"{cid}: digest {i} is not an object")
        extra = set(d) - DIGEST_KEYS
        if extra:
            raise CaseError(f"{cid}: digest {i}: unknown keys {sorted(extra)}")
        for key in ("session_id", "turn_index", "items"):
            if key not in d:
                raise CaseError(f"{cid}: digest {i}: missing {key}")
        if not _is_int(d["turn_index"]):
            raise CaseError(f"{cid}: digest {i}: turn_index must be an int")
        if not isinstance(d["items"], list):
            raise CaseError(f"{cid}: digest {i}: items must be a list")
        for j, item in enumerate(d["items"]):
            if not isinstance(item, dict):
                raise CaseError(f"{cid}: digest {i} item {j} is not an object")
            kind = item.get("kind")
            if not isinstance(kind, str) or kind not in DIGEST_ITEM_KEYS:
                raise CaseError(f"{cid}: digest {i} item {j} has an unknown kind")
            if not set(item) <= DIGEST_ITEM_KEYS[kind]:
                raise CaseError(f"{cid}: digest {i} item {j} has unknown keys")
    if len({json.dumps(d["session_id"]) for d in digests}) != 1:
        raise CaseError(f"{cid}: digests must share one session_id")
    turns = [d["turn_index"] for d in digests]
    if any(b <= a for a, b in itertools.pairwise(turns)):
        raise CaseError(f"{cid}: turn_index must be strictly ascending")
    if shape == 2:
        if len(digests) != 2:
            raise CaseError(f"{cid}: shape 2 needs exactly 2 digests")
        if turns[1] != turns[0] + 1:
            raise CaseError(f"{cid}: shape 2 digests must be consecutive turns")
        if any(i.get("kind") == "tool_result" for i in digests[1]["items"]):
            raise CaseError(f"{cid}: shape 2 turn 1 must not hold a tool_result")
    if shape == 3 and len(digests) != 1:
        raise CaseError(f"{cid}: shape 3 needs exactly 1 digest")


def _ingest_digests(
    cid: str, digests: list[dict[str, Any]]
) -> tuple[list[surprise.TurnDigest], list[str]]:
    """Run each digest through the production ingest path (validate, cap, redact)."""
    out: list[surprise.TurnDigest] = []
    redactions: list[str] = []
    for i, d in enumerate(digests):
        payload = {**d, "event_id": d.get("event_id", f"{d['session_id']}:{d['turn_index']}")}
        try:
            req = TurnDigestRequest.model_validate(payload)
        except pydantic.ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors()[:10]
            )
            raise CaseError(f"{cid}: digest {i}: {detail}") from None
        size = len(json.dumps(req.model_dump(mode="json")).encode("utf-8"))
        if size > turn_digest.TURN_DIGEST_MAX_BYTES:
            raise CaseError(f"{cid}: digest {i} exceeds 65536 bytes")
        red = turn_digest.redact_turn_digest(req)
        if red is None:
            raise CaseError(f"{cid}: redaction unavailable")
        r, types = red
        for t in types:
            if t not in redactions:
                redactions.append(t)
        out.append(
            surprise_worker.digest_from_row(
                {
                    "event_id": r.event_id,
                    "session_id": r.session_id,
                    "project": r.project or "",
                    "turn_index": r.turn_index,
                    "user_prompt": r.user_prompt,
                    "items": [item.model_dump() for item in r.items],
                    "final_message": r.final_message,
                    "truncated": r.truncated,
                    "ts": r.ts or "",
                }
            )
        )
    return out, redactions


def load_cases(path: Path) -> list[Case]:
    """Load, validate and redact a JSONL case file.

    Args:
        path: The case file.

    Returns:
        The cases in file order.

    Raises:
        CaseError: On any invalid line or case; the message never carries case text.
    """
    try:
        lines = path.read_bytes().decode("utf-8").split("\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise CaseError(f"cannot read cases file ({type(exc).__name__})") from None
    raw: list[tuple[int, dict[str, Any]]] = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            raise CaseError(f"line {n}: not a JSON object")
        raw.append((n, obj))
    if inside_repo(path) and any(
        not (isinstance(o.get("id"), str) and o["id"].startswith(SYNTHETIC_PREFIX)) for _, o in raw
    ):
        raise CaseError(CASES_REFUSAL)

    cases: list[Case] = []
    seen: set[str] = set()
    for n, obj in raw:
        cid = obj.get("id")
        if not isinstance(cid, str) or not _ID_RE.match(cid):
            raise CaseError(f"line {n}: id must match {_ID_RE.pattern}")
        if cid in seen:
            raise CaseError(f"{cid}: duplicate id")
        seen.add(cid)
        _check_common(cid, obj)
        shape = obj["shape"]
        if "digests" in obj:
            _unknown_keys(cid, obj, DIGEST_CASE_KEYS)
            digest_dicts = obj["digests"]
        else:
            _check_mined(cid, obj)
            digest_dicts = mined_to_digests(obj)
        _check_digest_structure(cid, shape, digest_dicts)
        digests, redactions = _ingest_digests(cid, digest_dicts)
        cases.append(
            Case(
                id=cid,
                shape=shape,
                label=obj["label"],
                hard_negative=obj.get("hard_negative", False),
                label_source=obj.get("label_source", "unspecified"),
                frame=obj.get("frame", "unspecified"),
                expected=obj.get("expected"),
                note=obj.get("note"),
                digests=digests,
                redactions=redactions,
            )
        )
    return cases


# --- statistics --------------------------------------------------------------


def wilson(k: int, n: int, z: float = WILSON_Z) -> list[float] | None:
    """Wilson score interval for k successes in n trials.

    Args:
        k: Successes.
        n: Trials.
        z: The normal quantile.

    Returns:
        ``[lo, hi]`` rounded to 4 decimals, or None when n is 0.
    """
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def precision_bar_status(
    tp: int,
    fp: int,
    precision: float | None,
    frames: Sequence[str],
    label_sources: Sequence[str],
) -> str:
    """Classify a cell against the precision bar.

    Args:
        tp: True positives.
        fp: False positives.
        precision: Unrounded precision, or None.
        frames: The cell's distinct frames.
        label_sources: The cell's distinct label sources.

    Returns:
        One of ``not-gating``, ``insufficient``, ``meets`` or ``below``.
    """
    if any(f != "unfiltered" for f in frames) or any(
        s == "unspecified" or s.startswith("silver") for s in label_sources
    ):
        return "not-gating"
    if tp + fp < PRECISION_BAR_MIN_PREDICTED:
        return "insufficient"
    if precision is not None and precision >= PRECISION_BAR:
        return "meets"
    return "below"


def _ratio(k: int, n: int) -> float | None:
    return k / n if n else None


def _r4(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


# --- model construction ------------------------------------------------------


def make_llm(model: str, timeout: float | None) -> Any:
    """Build the detector client exactly as production does.

    Args:
        model: The Anthropic model id.
        timeout: A per-request timeout override, or None for production's.

    Returns:
        An ``AnthropicLLMClient``.
    """
    cfg = service_config.build_anthropic_config(model=model)
    if timeout is not None:
        cfg = dataclasses.replace(cfg, timeout=timeout)
    return anthropic_llm.AnthropicLLMClient(cfg)


class _EvalLogHandler(logging.Handler):
    """Copies provider warnings into the current evaluation's list."""

    def emit(self, record: logging.LogRecord) -> None:
        sink = _EVAL_LOG.get()
        if sink is None or len(sink) >= MAX_WARNINGS_PER_ROW:
            return
        msg = record.getMessage()[:_CUT]
        if record.exc_info and record.exc_info[0] is not None:
            msg += f"{EXC_MARKER}{record.exc_info[0].__name__}"
        sink.append(msg)


# --- one evaluation ----------------------------------------------------------


def _base_row(case: Case, model: str) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(RESULT_KEYS)
    row.update(
        case_id=case.id,
        shape=case.shape,
        label=case.label,
        hard_negative=case.hard_negative,
        label_source=case.label_source,
        frame=case.frame,
        model=model,
        detector_version=surprise.SURPRISE_DETECTOR_VERSION,
        expected=case.expected,
        redactions=list(case.redactions),
        predicted=False,
        model_positive=False,
        llm_calls=0,
        cross_shape_candidates=[],
        provider_warnings=[],
    )
    return row


async def _score(
    case: Case, llm: Any, min_confidence_by_shape: dict[int, float], row: dict[str, Any]
) -> None:
    records = await surprise_worker.detect_digest(
        llm,
        case.digests[-1],
        case.digests,
        min_confidence=min_confidence_by_shape.get(case.shape),
    )
    for r in records:
        if r.outcome not in OUTCOMES:
            raise RuntimeError(f"unknown detector outcome {r.outcome!r}")
    own = [r for r in records if r.shape == case.shape]
    llm_calls = sum(1 for r in records if r.shape in (2, 3) and r.outcome not in NO_CALL_OUTCOMES)
    anomalies: list[str] = []
    row["llm_calls"] = llm_calls
    if not own:
        row["outcome"] = "not_applicable"
        anomalies.append("own_records=0")
    elif case.shape == 1:
        cands = [r for r in own if r.outcome == "candidate"]
        predicted = bool(cands)
        row["outcome"] = "candidate" if predicted else "no_surprise"
        row["predicted"] = predicted
        row["model_positive"] = predicted
        row["confidence"] = 1.0 if predicted else None
        first = cands[0] if cands else None
        if first is not None and first.candidate is not None:
            row["detector_output"] = first.candidate.detector_output
    else:
        rec = own[0]
        if len(own) > 1:
            anomalies.append(f"own_records={len(own)}")
        row["outcome"] = rec.outcome
        row["predicted"] = rec.outcome == "candidate"
        row["model_positive"] = rec.outcome in MODEL_POSITIVE_OUTCOMES
        row["confidence"] = rec.confidence
        row["latency_ms"] = rec.latency_ms
        row["prompt_chars"] = rec.prompt_chars
        row["response_chars"] = rec.response_chars
        row["raw_response_excerpt"] = rec.raw_response_excerpt
        row["detector_output"] = rec.candidate.detector_output if rec.candidate else None
        if rec.outcome not in NO_CALL_OUTCOMES:
            if case.shape == 2:
                prompt = surprise.build_shape2_prompt(case.digests[0], case.digests[1])
            else:
                prompt = surprise.build_shape3_prompt(case.digests[0])
            row["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if rec.prompt_chars is not None and rec.prompt_chars != len(
                surprise.SURPRISE_DETECTOR_SYSTEM
            ) + len(prompt):
                anomalies.append("prompt_mismatch")
    if (case.shape == 1 and llm_calls != 0) or (case.shape != 1 and llm_calls > 1):
        anomalies.append(f"llm_calls={llm_calls}")
    row["anomaly"] = ",".join(anomalies) or None
    row["cross_shape_candidates"] = sorted(
        r.shape for r in records if r.shape != case.shape and r.outcome == "candidate"
    )
    if case.shape == 1 and case.expected is not None and row["predicted"]:
        out = row["detector_output"] or {}
        row["expected_match"] = (
            out.get("wrong_belief") == case.expected["wrong_belief"]
            and out.get("corrected_fact") == case.expected["corrected_fact"]
        )
    if own:
        row["detector_model"] = own[0].detector_model
        row["reason"] = own[0].reason


async def _evaluate(
    case: Case,
    model: str,
    llm: Any,
    min_confidence_by_shape: dict[int, float],
    sem: asyncio.Semaphore,
    progress: dict[str, int],
) -> dict[str, Any]:
    async with sem:
        log: list[str] = []
        _EVAL_LOG.set(log)
        row = _base_row(case, model)
        try:
            await _score(case, llm, min_confidence_by_shape, row)
        except Exception as exc:
            row = _base_row(case, model)
            row["harness_error"] = f"{type(exc).__name__}: {str(exc)[:_CUT]}"
        row["provider_warnings"] = log
        progress["done"] += 1
        shown = "harness_error" if row["harness_error"] is not None else row["outcome"]
        print(
            f"[{progress['done']}/{progress['total']}] {model} {case.id} {shown}", file=sys.stderr
        )
        if row["anomaly"] is not None:
            print(f"anomaly {model} {case.id} {row['anomaly']}", file=sys.stderr)
        if row["harness_error"] is not None:
            exc_type = row["harness_error"].split(":", 1)[0]
            print(f"harness_error {model} {case.id} {exc_type}", file=sys.stderr)
        return row


# --- cells and warnings ------------------------------------------------------


def _pr(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    tp = sum(1 for r in rows if r["label"] and r[key])
    fp = sum(1 for r in rows if not r["label"] and r[key])
    fn = sum(1 for r in rows if r["label"] and not r[key])
    return _ratio(tp, tp + fp), _ratio(tp, tp + fn)


def _build_cell(
    shape: int, model: str, all_rows: list[dict[str, Any]], production_timeout_s: float
) -> dict[str, Any]:
    rows = [r for r in all_rows if r["harness_error"] is None]
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    tp = sum(1 for r in pos if r["predicted"])
    fn = len(pos) - tp
    fp = sum(1 for r in neg if r["predicted"])
    tn = len(neg) - fp
    fn_applicable = sum(1 for r in pos if not r["predicted"] and r["outcome"] != "not_applicable")
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    recall_applicable = _ratio(tp, tp + fn_applicable)
    if precision is not None and recall is not None:
        f1: float | None = (
            2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )
    else:
        f1 = None
    mp_precision, mp_recall = _pr(rows, "model_positive")
    durable = [r for r in pos if (r["expected"] or {}).get("durable") is True]
    outcomes = dict.fromkeys(OUTCOMES, 0)
    for r in rows:
        outcomes[r["outcome"]] += 1
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    exc_types: dict[str, int] = {}
    for r in rows:
        for w in r["provider_warnings"]:
            if EXC_MARKER in w:
                name = w.rsplit(EXC_MARKER, 1)[1]
                exc_types[name] = exc_types.get(name, 0) + 1
    by_source: dict[str, dict[str, int]] = {}
    for r in rows:
        counts = by_source.setdefault(r["label_source"], {"fn": 0, "fp": 0, "tn": 0, "tp": 0})
        key = ("t" if r["label"] == r["predicted"] else "f") + ("p" if r["predicted"] else "n")
        counts[key] += 1
    frames = sorted({r["frame"] for r in all_rows})
    label_sources = sorted({r["label_source"] for r in all_rows})
    cell: dict[str, Any] = {
        "shape": shape,
        "model": model,
        "n": len(rows),
        "positives": len(pos),
        "negatives": len(neg),
        "hard_negatives": sum(1 for r in neg if r["hard_negative"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _r4(precision),
        "precision_wilson95": wilson(tp, tp + fp),
        "recall": _r4(recall),
        "recall_wilson95": wilson(tp, tp + fn),
        "recall_applicable": _r4(recall_applicable),
        "recall_applicable_wilson95": wilson(tp, tp + fn_applicable),
        "f1": _r4(f1),
        "model_positive_precision": _r4(mp_precision),
        "model_positive_recall": _r4(mp_recall),
        "recall_durable": _r4(_ratio(sum(1 for r in durable if r["predicted"]), len(durable))),
        "fp_hard_negative": sum(1 for r in neg if r["hard_negative"] and r["predicted"]),
        "outcomes": outcomes,
        "errors": sum(outcomes[o] for o in ERROR_OUTCOMES),
        "lost_to_gates": sum(1 for r in pos if r["model_positive"] and not r["predicted"]),
        "not_applicable_positives": sum(1 for r in pos if r["outcome"] == "not_applicable"),
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "latency_ms_mean": _r4(sum(latencies) / len(latencies)) if latencies else None,
        "latency_ms_p95": (latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None),
        "latency_over_production_timeout": sum(
            1 for v in latencies if v > production_timeout_s * 1000
        ),
        "prompt_chars_total": sum(r["prompt_chars"] or 0 for r in rows),
        "response_chars_total": sum(r["response_chars"] or 0 for r in rows),
        "provider_no_text": sum(
            1 for r in rows if any(NO_TEXT_MARKER in w for w in r["provider_warnings"])
        ),
        "provider_failed": sum(
            1 for r in rows if any(FAILED_MARKER in w for w in r["provider_warnings"])
        ),
        "provider_exception_types": exc_types,
        "anomalies": sum(1 for r in rows if r["anomaly"] is not None),
        "cross_shape_fp": sum(1 for r in neg if r["cross_shape_candidates"]),
        "harness_errors": len(all_rows) - len(rows),
        "tp_expected_mismatch": sum(
            1 for r in pos if r["predicted"] and r["expected_match"] is False
        ),
        "label_sources": label_sources,
        "frames": frames,
        "by_label_source": by_source,
        "fp_ids": sorted(r["case_id"] for r in neg if r["predicted"]),
        "fn_ids": sorted(r["case_id"] for r in pos if not r["predicted"]),
        "precision_bar": precision_bar_status(tp, fp, precision, frames, label_sources),
    }
    return cell


def _cell_warnings(cell: dict[str, Any], production_timeout_s: float) -> list[str]:
    p = f"{cell['model']} shape {cell['shape']}: "
    out: list[str] = []
    if cell["errors"] > 0:
        out.append(
            f"{p}{cell['errors']} detector errors (llm_error/unparseable/invalid_fields)"
            " counted as negatives"
        )
    if cell["provider_no_text"] > 0:
        out.append(f"{p}{cell['provider_no_text']} responses had no text block")
    if cell["provider_failed"] > 0:
        types = ", ".join(
            f"{name}={count}" for name, count in sorted(cell["provider_exception_types"].items())
        )
        out.append(f"{p}{cell['provider_failed']} provider failures ({types})")
    n_na = cell["outcomes"]["not_applicable"]
    if n_na > 0:
        out.append(
            f"{p}{n_na} cases were not_applicable under the detector's skip rules,"
            f" {cell['not_applicable_positives']} of them labelled positive"
            " (detector scope, not model quality)"
        )
    if cell["lost_to_gates"] > 0:
        out.append(
            f"{p}{cell['lost_to_gates']} positives lost to the confidence floor or grounding check"
        )
    if cell["latency_over_production_timeout"] > 0:
        out.append(
            f"{p}{cell['latency_over_production_timeout']} calls took longer than the"
            f" production timeout of {production_timeout_s!s}s"
        )
    if cell["anomalies"] > 0:
        out.append(f"{p}{cell['anomalies']} evaluations broke the one-record/one-call invariant")
    if cell["cross_shape_fp"] > 0:
        out.append(
            f"{p}{cell['cross_shape_fp']} negative cases produced a candidate of another shape"
            " (production would write it)"
        )
    if cell["harness_errors"] > 0:
        out.append(
            f"{p}{cell['harness_errors']} evaluations raised in the harness (excluded from metrics)"
        )
    if cell["tp_expected_mismatch"] > 0:
        out.append(
            f"{p}{cell['tp_expected_mismatch']} true positives extracted a different belief"
            " than expected"
        )
    if cell["positives"] == 0:
        out.append(f"{p}no labelled positives")
    if cell["precision_bar"] == "not-gating":
        out.append(
            f"{p}precision is not gating (frames={','.join(cell['frames'])},"
            f" label_sources={','.join(cell['label_sources'])})"
        )
    return out


# --- report.md ---------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return f"{x:.4f}"
    if isinstance(x, list) and len(x) == 2:
        return f"[{x[0]:.4f}, {x[1]:.4f}]"
    return str(x)


def _md_cell(text: Any) -> str:
    if text is None:
        return "-"
    s = str(text)[:_MD_CELL_MAX]
    return s.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _disagreement_fields(row: dict[str, Any]) -> tuple[Any, Any]:
    source = row["detector_output"]
    if not isinstance(source, dict) and row["raw_response_excerpt"] is not None:
        source = json_parser.parse_json_object(row["raw_response_excerpt"])
    if not isinstance(source, dict):
        return None, None
    return source.get("wrong_belief"), source.get("evidence_excerpt")


def _render_md(report: dict[str, Any], rows_by_cell: list[list[dict[str, Any]]]) -> str:
    dirty = report["git_dirty"]
    lines = [
        "# Surprise detector eval",
        f"detector_version={report['detector_version']} cases={report['n_cases']}"
        f" cases_sha256={report['cases_sha256']}"
        f" git_commit={report['git_commit'] or 'unknown'}"
        f" git_dirty={'unknown' if dirty is None else str(dirty).lower()}"
        f" timeout_s={report['timeout_s']} production_timeout_s={report['production_timeout_s']}"
        f" min_confidence={report['min_confidence']:.2f}"
        f" min_confidence_by_shape="
        + ",".join(f"s{k}={v:.2f}" for k, v in report["min_confidence_by_shape"].items()),
        "",
        TABLE_HEADER,
        "|" + "---|" * 19,
    ]
    for c in report["cells"]:
        values = [
            c["shape"],
            c["model"],
            c["n"],
            c["positives"],
            c["tp"],
            c["fp"],
            c["fn"],
            c["tn"],
            c["precision"],
            c["precision_wilson95"],
            c["recall"],
            c["recall_wilson95"],
            c["recall_applicable"],
            c["f1"],
            c["model_positive_precision"],
            c["model_positive_recall"],
            c["errors"],
            c["outcomes"]["not_applicable"],
            c["precision_bar"],
        ]
        lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")
    lines += [
        "",
        "## Outcomes",
        "",
        "| shape | model | " + " | ".join(OUTCOMES) + " |",
        "|" + "---|" * (2 + len(OUTCOMES)),
    ]
    for c in report["cells"]:
        values = [c["shape"], c["model"], *(c["outcomes"][o] for o in OUTCOMES)]
        lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")
    lines += ["", "## Disagreements", ""]
    disagreements: list[str] = []
    for cell_rows in rows_by_cell:
        scored = [
            r for r in cell_rows if r["harness_error"] is None and r["label"] != r["predicted"]
        ]
        for r in sorted(scored, key=lambda r: r["case_id"]):
            wrong_belief, evidence = _disagreement_fields(r)
            values = [
                _fmt(r["shape"]),
                _md_cell(r["model"]),
                _md_cell(r["case_id"]),
                _fmt(r["label"]),
                _md_cell(r["outcome"]),
                _md_cell(r["reason"] or None),
                _fmt(r["confidence"]),
                _md_cell(wrong_belief),
                _md_cell(evidence),
            ]
            disagreements.append("| " + " | ".join(values) + " |")
    if disagreements:
        lines += [DISAGREEMENT_HEADER, "|" + "---|" * 9, *disagreements]
    else:
        lines.append("none")
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in report["warnings"]] or ["none"]
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in NOTES]
    return "\n".join(lines) + "\n"


# --- provenance --------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO_ROOT), *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- commands ----------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(Path(args.cases))
    except CaseError as exc:
        print(f"validate: {exc}", file=sys.stderr)
        return 2
    by_source: dict[str, int] = {}
    by_shape = {str(s): {"negatives": 0, "positives": 0} for s in (1, 2, 3)}
    for c in cases:
        by_source[c.label_source] = by_source.get(c.label_source, 0) + 1
        by_shape[str(c.shape)]["positives" if c.label else "negatives"] += 1
    summary = {
        "by_label_source": by_source,
        "by_shape": by_shape,
        "cases": len(cases),
        "redacted_cases": sum(1 for c in cases if c.redactions),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _arg_error(args: argparse.Namespace) -> tuple[str | None, list[int]]:
    parts = args.shapes.split(",")
    if not parts or any(p not in ("1", "2", "3") for p in parts) or len(set(parts)) != len(parts):
        return "--shapes must be a comma list of distinct members of 1,2,3", []
    shapes = sorted(int(p) for p in parts)
    if args.concurrency < 1:
        return "--concurrency must be >= 1", shapes
    if args.limit is not None and args.limit < 1:
        return "--limit must be >= 1", shapes
    if args.timeout is not None and not args.timeout > 0:
        return "--timeout must be > 0", shapes
    if len(set(args.model)) != len(args.model):
        return "--model values must be distinct", shapes
    return None, shapes


def _cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if inside_repo(out):
        print(OUT_REFUSAL, file=sys.stderr)
        return 2
    reason, shapes = _arg_error(args)
    if reason is not None:
        print(f"run: {reason}", file=sys.stderr)
        return 2
    if not args.force and any((out / name).exists() for name in RESULT_FILES):
        print(f"run: refusing to overwrite results in {out}; pass --force", file=sys.stderr)
        return 2
    cases_path = Path(args.cases)
    try:
        cases = load_cases(cases_path)
    except CaseError as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 2
    cases = [c for c in cases if c.shape in shapes]
    if args.limit is not None:
        cases = cases[: args.limit]
    if any(c.shape in (2, 3) for c in cases) and not args.model:
        print("run: --model is required for shape 2 and 3 cases", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        for name in RESULT_FILES:
            (out / name).unlink(missing_ok=True)
    return asyncio.run(_run(args, cases_path, cases, shapes, out))


async def _run(
    args: argparse.Namespace,
    cases_path: Path,
    cases: list[Case],
    shapes: list[int],
    out: Path,
) -> int:
    started_at = _now()
    min_confidence = surprise_worker.detector_min_confidence()
    min_confidence_by_shape = {
        shape: surprise_worker.detector_min_confidence_for(shape) for shape in (2, 3)
    }
    production_timeout_s = service_config.get_anthropic_timeout()
    timeout_s = args.timeout if args.timeout is not None else production_timeout_s
    models: list[str] = list(args.model)
    model_cases = [c for c in cases if c.shape in (2, 3)]
    shape1_cases = [c for c in cases if c.shape == 1]

    llms: list[tuple[str, Any]] = []
    if model_cases:
        llms = [(m, make_llm(m, args.timeout)) for m in models]
    jobs: list[tuple[Case, str, Any]] = [
        (c, surprise.SHAPE1_DETECTOR_MODEL, None) for c in shape1_cases
    ]
    for model, llm in llms:
        jobs += [(c, model, llm) for c in model_cases]

    sem = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "total": len(jobs)}
    provider_logger = logging.getLogger(PROVIDER_LOGGER)
    handler = _EvalLogHandler(level=logging.WARNING)
    old_level = provider_logger.level
    provider_logger.addHandler(handler)
    if provider_logger.getEffectiveLevel() > logging.WARNING:
        provider_logger.setLevel(logging.WARNING)
    try:
        rows = await asyncio.gather(
            *(_evaluate(c, m, llm, min_confidence_by_shape, sem, progress) for c, m, llm in jobs)
        )
    finally:
        provider_logger.removeHandler(handler)
        provider_logger.setLevel(old_level)
        for _, llm in llms:
            await llm.close()
    finished_at = _now()

    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    cell_order = [(1, surprise.SHAPE1_DETECTOR_MODEL)] + [(s, m) for s in (2, 3) for m in models]
    cells: list[dict[str, Any]] = []
    rows_by_cell: list[list[dict[str, Any]]] = []
    for shape, model in cell_order:
        cell_rows = [r for r in rows if r["shape"] == shape and r["model"] == model]
        if not cell_rows:
            continue
        cells.append(_build_cell(shape, model, cell_rows, production_timeout_s))
        rows_by_cell.append(cell_rows)
    warnings: list[str] = []
    for cell in cells:
        warnings += _cell_warnings(cell, production_timeout_s)
    default_model = surprise_worker.SURPRISE_DETECTOR_DEFAULT_MODEL
    if any(r["shape"] in (2, 3) for r in rows) and default_model not in models:
        warnings.append(f"production default detector {default_model} was not measured")

    prompt_pairs = {
        f"{r['case_id']}:{r['prompt_sha256']}" for r in rows if r["prompt_sha256"] is not None
    }
    git_commit = _git("rev-parse", "HEAD")
    git_status = _git("status", "--porcelain")
    report: dict[str, Any] = {
        "schema": 1,
        "detector_version": surprise.SURPRISE_DETECTOR_VERSION,
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "n_cases": len(cases),
        "shapes": shapes,
        "models": models,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": git_commit.strip() if git_commit is not None else None,
        "git_dirty": bool(git_status.strip()) if git_status is not None else None,
        "args": {
            "concurrency": args.concurrency,
            "force": args.force,
            "limit": args.limit,
            "shapes": shapes,
            "timeout": args.timeout,
        },
        "versions": {
            name: _version(name) for name in ("anthropic", "kb-core", "personal-kb-web-service")
        },
        "timeout_s": timeout_s,
        "production_timeout_s": production_timeout_s,
        "production_default_model": default_model,
        "min_confidence": min_confidence,
        "min_confidence_by_shape": {str(k): v for k, v in min_confidence_by_shape.items()},
        "prompt_set_sha256": (
            hashlib.sha256("\n".join(sorted(prompt_pairs)).encode("utf-8")).hexdigest()
            if prompt_pairs
            else None
        ),
        "cells": cells,
        "warnings": warnings,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "report.md").write_text(_render_md(report, rows_by_cell), encoding="utf-8")

    broken = any(r["anomaly"] is not None or r["harness_error"] is not None for r in rows)
    code = 4 if broken else 0
    for model in models:
        mrows = [r for r in rows if r["model"] == model and r["shape"] in (2, 3)]
        called = [r for r in mrows if r["llm_calls"] > 0]
        if called and all(r["outcome"] == "llm_error" for r in called):
            print(f"run: detector unavailable for {model}", file=sys.stderr)
        elif mrows and not called:
            print(
                f"run: no detector calls made for {model}"
                " (every shape-2/3 case was not_applicable)",
                file=sys.stderr,
            )
        else:
            continue
        if code == 0:
            code = 3
    return code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surprise_eval", description="Offline eval of the surprise-capture detector."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="validate a case file; no model calls, no files")
    v.add_argument("--cases", required=True)
    r = sub.add_parser("run", help="score the detector on a case file")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", action="append", default=[])
    r.add_argument("--shapes", default="1,2,3")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    r.add_argument("--timeout", type=float, default=None)
    r.add_argument("--force", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the ``validate`` and ``run`` subcommands.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    args = _parser().parse_args(argv)
    if args.command == "validate":
        return _cmd_validate(args)
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
