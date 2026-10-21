#!/usr/bin/env python3
"""Generate the wire-contract fixtures under ``contracts/``.

The fixtures are golden request/response JSON for ``GET /api/kb/prevention``,
``POST /api/kb/turn``, ``POST /api/kb/event`` and ``POST /api/kb/belief-check``.
They are written only by this script and never edited by hand; see
``contracts/README.md``.

A breaking change to a contract (see :func:`breaking_changes`) makes this script
refuse to write until ``CONTRACT_VERSIONS`` is bumped.

Usage (from the repo root):
  uv run python packages/kb-service/scripts/gen_contracts.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kb_core.cues import build_cue

from kb_service import harness_tools
from kb_service.belief_check import (
    BELIEF_CHECK_MATCHER_VERSION,
    BELIEF_CHECK_THRESHOLD,
)
from kb_service.main import app
from kb_service.models import (
    BeliefCheckRequest,
    BeliefCheckResponse,
    EventRequest,
    EventResponse,
    GateSettings,
    IndexCue,
    PreventionDiagnostics,
    PreventionResponse,
    SliceItem,
    TurnDigestRequest,
    TurnDigestResponse,
)
from kb_service.prevention import render_slice
from kb_service.routes.prevention_routes import (
    DEFAULT_MAX_DENIES_PER_HOUR,
    DEFAULT_MAX_DENIES_PER_TURN,
    DEFAULT_REARM_HOURS,
    LEGACY_MAX_DENIES,
)
from kb_service.turn_digest import TURN_DIGEST_MAX_BYTES

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "contracts"
GENERATED_BY = "packages/kb-service/scripts/gen_contracts.py"
CONTRACT_VERSIONS: dict[str, int] = {
    "prevention": 1,
    "turn": 1,
    "event": 1,
    "belief_check": 1,
}
# A literal, so the build does not depend on KB_EMBEDDING_MODEL.
_CONTRACT_EMBED_MODEL = "qwen3-embedding:0.6b"

_CLAUDE_CODE_ITEMS: list[dict[str, Any]] = [
    {"kind": "assistant_text", "text": "Pushing the branch now."},
    {
        "kind": "tool_call",
        "tool_use_id": "toolu_contract_1",
        "tool": "Bash",
        "target": "git push mirror feat/x",
        "target_class": "git push",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "toolu_contract_1",
        "is_error": True,
        "excerpt": "remote: Permission to mirror denied",
    },
]

_TALOS_ITEMS: list[dict[str, Any]] = [
    {
        "kind": "tool_call",
        "tool_use_id": "toolu_contract_t1",
        "tool": "bash",
        "target": "git push mirror feat/x",
        "target_class": "",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "toolu_contract_t1",
        "is_error": True,
        "excerpt": "remote: Permission to mirror denied",
    },
    {
        "kind": "reasoning",
        "text": "The mirror is release-only, so origin is the right remote.",
        "truncated": False,
    },
    {
        "kind": "tool_call",
        "tool_use_id": "toolu_contract_t2",
        "tool": "edit_file",
        "target": "src/a.py",
        "target_class": "",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "toolu_contract_t2",
        "is_error": False,
        "excerpt": "",
    },
    {
        "kind": "harness_correction",
        "trigger": "gate_red",
        "detail": "FAILED tests/test_contract.py::test_example",
        "resolved_by": ["toolu_contract_t2"],
        "resolved": True,
    },
]

_CUE_PUSH = IndexCue(
    resolution_id="kb-00101",
    updated_at="2026-10-02T09:00:00+00:00",
    tool="Bash",
    target_class="git push",
    args_prefix="",
    wrong_belief="Push feature branches to the mirror remote.",
    corrected_fact=(
        "Push feature branches to origin; the mirror remote receives release"
        " pushes only."
    ),
    evidence="remote: Permission to mirror denied",
    provenance_label="deliberate/observed",
    observed_once=False,
)
_CUE_REMOTE = IndexCue(
    resolution_id="kb-00102",
    updated_at="2026-10-01T09:00:00+00:00",
    tool="Bash",
    target_class="git remote",
    args_prefix="add",
    wrong_belief="Add a new remote with git remote add.",
    corrected_fact=(
        "Create remotes with ./scripts/add_remote.sh so the server-side"
        " repository exists first."
    ),
    evidence="",
    provenance_label="autonomous/observed",
    observed_once=False,
)
_S1 = SliceItem(
    entry_id="kb-00101",
    corrected_fact=_CUE_PUSH.corrected_fact,
    wrong_belief=_CUE_PUSH.wrong_belief,
    provenance_label="deliberate/observed",
)
_S_REMOTE = SliceItem(
    entry_id="kb-00102",
    corrected_fact=_CUE_REMOTE.corrected_fact,
    wrong_belief=_CUE_REMOTE.wrong_belief,
    provenance_label="autonomous/observed",
)
_S2 = SliceItem(
    entry_id="kb-00103",
    corrected_fact="The local KB service listens on port 8765.",
    wrong_belief="",
    provenance_label="supersedes-edge",
)
_S3 = SliceItem(
    entry_id="kb-00104",
    corrected_fact=(
        "Run the hook tests from packages/personal-kb-hook with --project ../.."
    ),
    wrong_belief="Run the hook tests from the repo root.",
    provenance_label="autonomous/observed, observed once, unconfirmed",
)


def _gate(*, enabled: bool, shadow: bool) -> GateSettings:
    return GateSettings(
        enabled=enabled,
        shadow=shadow,
        max_denies=LEGACY_MAX_DENIES,
        max_denies_per_turn=DEFAULT_MAX_DENIES_PER_TURN,
        max_denies_per_hour=DEFAULT_MAX_DENIES_PER_HOUR,
        rearm_hours=DEFAULT_REARM_HOURS,
    )


def turn_item_kinds() -> list[str]:
    """Return the TurnItem kinds, in union order."""
    schema = TurnDigestRequest.model_json_schema()
    return list(schema["properties"]["items"]["items"]["discriminator"]["mapping"])


def prevention_query_schema() -> dict[str, Any]:
    """Build the JSON Schema of the prevention query string from OpenAPI."""
    params = app.openapi()["paths"]["/api/kb/prevention"]["get"]["parameters"]
    query = [p for p in params if p["in"] == "query"]
    return {
        "type": "object",
        "properties": {p["name"]: p["schema"] for p in query},
        "required": [p["name"] for p in query if p.get("required")],
        "additionalProperties": False,
    }


def _header(contract: str, method: str, path: str) -> dict[str, Any]:
    return {
        "contract": contract,
        "contract_version": CONTRACT_VERSIONS[contract],
        "generated_by": GENERATED_BY,
        "method": method,
        "path": path,
    }


def build_prevention() -> dict[str, Any]:
    """Build the ``prevention.json`` document."""
    armed = PreventionResponse(
        project="personal-kb",
        gate=_gate(enabled=True, shadow=False),
        index=[_CUE_PUSH, _CUE_REMOTE],
        slice=[_S1, _S_REMOTE, _S2, _S3],
        slice_text=render_slice("personal-kb", [_S1, _S_REMOTE, _S2, _S3]),
        diagnostics=PreventionDiagnostics(
            resolutions_total=3, index_excluded_observed_once=1
        ),
        surprise_capture="shadow",
    )
    disabled = PreventionResponse(
        project="personal-kb",
        gate=_gate(enabled=False, shadow=True),
        index=[],
        slice=[_S1],
        slice_text=render_slice("personal-kb", [_S1]),
        diagnostics=PreventionDiagnostics(resolutions_total=1),
        surprise_capture="off",
    )
    no_project = PreventionResponse(
        project="",
        gate=_gate(enabled=False, shadow=True),
        index=[],
        slice=[],
        slice_text="",
        diagnostics=PreventionDiagnostics(),
        surprise_capture="on",
    )
    talos = armed.model_copy(update={"tool_map": harness_tools.tool_map("talos")})
    talos_off = talos.model_copy(
        update={"gate": _gate(enabled=False, shadow=True), "slice": [_S1]}
    )
    talos_off = talos_off.model_copy(
        update={"slice_text": render_slice("personal-kb", [_S1])}
    )
    examples = [
        ("gate_armed", armed),
        ("gate_disabled_slice_only", disabled),
        ("no_project", no_project),
        ("talos_gate_armed", talos),
        ("talos_gate_switch_off", talos_off),
    ]
    return {
        **_header("prevention", "GET", "/api/kb/prevention"),
        "request": {
            "query_schema": prevention_query_schema(),
            "examples": [
                {
                    "name": "project_and_cwd",
                    "query": {
                        "project": "personal-kb",
                        "cwd": "/home/dev/git/personal_kb",
                        "session_id": "contract-session-1",
                    },
                },
                {
                    "name": "talos",
                    "query": {
                        "project": "personal-kb",
                        "session_id": "contract-session-2",
                        "harness": "talos",
                    },
                },
            ],
        },
        "response": {
            "schema": PreventionResponse.model_json_schema(),
            "examples": [
                {"name": n, "body": r.model_dump(mode="json")} for n, r in examples
            ],
        },
    }


def build_turn() -> dict[str, Any]:
    """Build the ``turn.json`` document."""
    missing = sorted(
        set(turn_item_kinds()) - {i["kind"] for i in _CLAUDE_CODE_ITEMS + _TALOS_ITEMS}
    )
    if missing:
        raise ValueError(
            f"no example item for TurnItem kind(s) {missing}; add one to"
            " _CLAUDE_CODE_ITEMS (kinds the Claude Code hook emits) or"
            " _TALOS_ITEMS in packages/kb-service/scripts/gen_contracts.py"
        )
    raws: list[tuple[str, dict[str, Any]]] = [
        (
            "claude_code_turn",
            {
                "event_id": "contract-session-1:3",
                "session_id": "contract-session-1",
                "harness": "claude-code",
                "mode": "interactive",
                "engine": None,
                "host": "contract-host",
                "hook_version": "0.0.0-contract",
                "project": "personal-kb",
                "turn_index": 3,
                "ts": "2026-10-10T12:00:00+00:00",
                "user_prompt": "Push the feature branch.",
                "items": _CLAUDE_CODE_ITEMS,
                "final_message": (
                    "The mirror rejected the push; pushed to origin instead."
                ),
                "truncated": False,
            },
        ),
        (
            "talos_turn",
            {
                "event_id": "contract-session-2:0",
                "session_id": "contract-session-2",
                "harness": "talos",
                "mode": "headless",
                "engine": "talos-glm-flash",
                "host": "contract-host",
                "hook_version": None,
                "project": "personal-kb",
                "turn_index": 0,
                "ts": "2026-10-10T12:05:00+00:00",
                "user_prompt": "Implement acceptance criterion 1.",
                "items": _TALOS_ITEMS,
                "final_message": "Done; the gate is green.",
                "truncated": True,
            },
        ),
        (
            # One SubagentStop digest: the parent session's read-only turn
            # index, the agent event_id form, and the agent fields set.
            "claude_code_turn_subagent",
            {
                "event_id": "contract-session-1:agent:agent-contract-1",
                "session_id": "contract-session-1",
                "harness": "claude-code",
                "mode": "interactive",
                "engine": None,
                "host": "contract-host",
                "hook_version": "0.0.0-contract",
                "project": "personal-kb",
                "turn_index": 3,
                "ts": "2026-10-10T12:10:00+00:00",
                "user_prompt": "Find the failing test in the hook suite.",
                "items": _CLAUDE_CODE_ITEMS,
                "final_message": "The failure was the missing drop line.",
                "truncated": False,
                "agent_id": "agent-contract-1",
                "agent_type": "general-purpose",
            },
        ),
        (
            "minimal",
            {
                "event_id": "contract-session-1:0",
                "session_id": "contract-session-1",
                "turn_index": 0,
            },
        ),
    ]
    response_schema = TurnDigestResponse.model_json_schema()
    reasons = response_schema["properties"]["reason"]["enum"]
    return {
        **_header("turn", "POST", "/api/kb/turn"),
        "request": {
            "max_body_bytes": TURN_DIGEST_MAX_BYTES,
            "schema": TurnDigestRequest.model_json_schema(),
            "examples": [
                {
                    "name": name,
                    "body": TurnDigestRequest.model_validate(raw).model_dump(
                        mode="json"
                    ),
                }
                for name, raw in raws
            ],
        },
        "response": {
            "schema": response_schema,
            "examples": [
                {
                    "name": r,
                    "body": TurnDigestResponse(
                        recorded=(r == "recorded"),
                        reason=r,
                        redactions=["Secret Keyword"] if r == "recorded" else [],
                    ).model_dump(mode="json"),
                }
                for r in reasons
            ],
        },
    }


# Fields the server computes (via ``build_cue`` and the route) and stores in
# ``failure_events``; none is accepted from the client.
_EVENT_SERVER_COMPUTED = [
    "cue_key",
    "normalizer_version",
    "host_class",
    "project_source",
    "target",
    "target_class",
    "normalized_error",
    "error_rule",
    "anomaly",
    "raw_error_excerpt",
    "received_ts",
]

_EVENT_RAWS: list[tuple[str, dict[str, Any]]] = [
    (
        "claude_code_bash_failure",
        {
            "type": "post_tool",
            "event_id": "contract-session-1:toolu_contract_1",
            "session_id": "contract-session-1",
            "harness": "claude-code",
            "mode": "interactive",
            "engine": None,
            "host": "contract-host",
            "hook_version": "0.0.0-contract",
            "cwd": "/home/dev/git/personal_kb",
            "project": "personal-kb",
            "ts": "2026-10-10T12:00:00+00:00",
            "tool_name": "Bash",
            "tool_input": {"command": "git push mirror feat/x"},
            "tool_use_id": "toolu_contract_1",
            "error": "Exit code 1\nremote: Permission to mirror denied",
            "is_error": True,
            "is_interrupt": False,
            "duration_ms": 120,
        },
    ),
    (
        # One subagent failure: the parent session_id, the subagent's own
        # agent fields, and a tool_use_id that only its transcript holds.
        "claude_code_subagent_failure",
        {
            "type": "post_tool",
            "event_id": "contract-session-3:toolu_contract_1",
            "session_id": "contract-session-3",
            "harness": "claude-code",
            "mode": "interactive",
            "engine": None,
            "host": "contract-host",
            "hook_version": "0.0.0-contract",
            "cwd": "/home/dev/git/personal_kb",
            "project": "personal-kb",
            "ts": "2026-10-10T12:15:00+00:00",
            "tool_name": "Bash",
            "tool_input": {"command": "git push mirror feat/x"},
            "tool_use_id": "toolu_contract_1",
            "agent_id": "agent-contract-1",
            "agent_type": "general-purpose",
            "error": "Exit code 1\nremote: Permission to mirror denied",
            "is_error": True,
            "is_interrupt": False,
            "duration_ms": 120,
        },
    ),
    (
        "talos_bash_failure",
        {
            "type": "post_tool",
            "event_id": "contract-session-2:toolu_contract_t1",
            "session_id": "contract-session-2",
            "harness": "talos",
            "mode": "headless",
            "engine": "talos-glm-flash",
            "host": "contract-host",
            "cwd": "/home/dev/git/personal_kb",
            "project": "personal-kb",
            "ts": "2026-10-10T12:05:00+00:00",
            "tool_name": "bash",
            "tool_input": {"command": "git push mirror feat/x"},
            "tool_use_id": "toolu_contract_t1",
            "error": (
                "exit 1 (0.01s)\n--- stdout ---\n\n--- stderr ---\n"
                "remote: Permission to mirror denied\n"
            ),
            "is_error": True,
            "duration_ms": 95,
        },
    ),
    (
        "talos_edit_file_failure",
        {
            "type": "post_tool",
            "event_id": "contract-session-2:toolu_contract_t2",
            "session_id": "contract-session-2",
            "harness": "talos",
            "mode": "headless",
            "engine": "talos-glm-flash",
            "host": "contract-host",
            "cwd": "/home/dev/git/personal_kb",
            "project": "personal-kb",
            "ts": "2026-10-10T12:06:00+00:00",
            "tool_name": "edit_file",
            "tool_input": {"path": "src/a.py"},
            "tool_use_id": "toolu_contract_t2",
            "error": "old_string not found in src/a.py",
            "is_error": True,
        },
    ),
]


def event_stored(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the computed ``failure_events`` fields the route stores for *raw*."""
    body = EventRequest.model_validate(raw)
    name, error = body.tool_name or "", body.error or ""
    tool = harness_tools.canonical_tool(body.harness, name)
    tool_input = harness_tools.canonical_tool_input(body.harness, name, body.tool_input)
    cue = build_cue(tool, tool_input, error, body.project, body.cwd)
    return {
        "cue_key": cue.cue_key,
        "normalizer_version": cue.normalizer_version,
        "host_class": cue.host_class,
        "project_source": cue.project_source,
        "tool": cue.tool,
        "target": cue.target,
        "target_class": cue.target_class,
        "normalized_error": cue.normalized_error,
        "error_rule": cue.error_rule,
        "anomaly": None,
        "raw_error_excerpt": error[:2000],
    }


def build_event() -> dict[str, Any]:
    """Build the ``event.json`` document."""
    request_schema = EventRequest.model_json_schema()
    response_schema = EventResponse.model_json_schema()
    reasons = response_schema["properties"]["reason"]["enum"]
    examples = []
    for name, raw in _EVENT_RAWS:
        stored = event_stored(raw)
        examples.append(
            {
                "name": name,
                "body": EventRequest.model_validate(raw).model_dump(mode="json"),
                "stored": stored,
                "response": EventResponse(
                    recorded=True,
                    cue_key=stored["cue_key"],
                    normalizer_version=stored["normalizer_version"],
                    reason="recorded",
                ).model_dump(mode="json"),
            }
        )
    return {
        **_header("event", "POST", "/api/kb/event"),
        "fields": {
            "client_supplied": list(request_schema["properties"]),
            "server_computed": _EVENT_SERVER_COMPUTED,
        },
        "request": {
            "schema": request_schema,
            "examples": [{"name": e["name"], "body": e["body"]} for e in examples],
        },
        "stored": {e["name"]: e["stored"] for e in examples},
        "response": {
            "schema": response_schema,
            "examples": [{"name": e["name"], "body": e["response"]} for e in examples]
            + [
                {
                    "name": r,
                    "body": EventResponse(recorded=False, reason=r).model_dump(
                        mode="json"
                    ),
                }
                for r in reasons
                if r in ("duplicate", "not-failure", "unsupported-type")
            ],
        },
    }


_BC_REQUESTS: list[tuple[str, dict[str, Any]]] = [
    (
        "talos_turn",
        {
            "text": "Tests are green. Push feature branches to the mirror remote.",
            "reasoning": (
                "The branch is ready for review, so it should go to the shared"
                " remote now."
            ),
            "project": "personal-kb",
            "pending_calls": [
                {"tool": "bash", "input_summary": "git push mirror feat/x"}
            ],
            "session_id": "contract-session-2",
            "turn_index": 4,
        },
    ),
    (
        "minimal",
        {
            "text": "Push feature branches to the mirror remote.",
            "project": "personal-kb",
        },
    ),
]

_BC_ENTRIES: dict[str, tuple[str, str, str]] = {
    "kb-00101": (
        _CUE_PUSH.wrong_belief,
        _CUE_PUSH.corrected_fact,
        "deliberate/observed",
    ),
    "kb-00102": (
        _CUE_REMOTE.wrong_belief,
        _CUE_REMOTE.corrected_fact,
        "autonomous/observed",
    ),
    "kb-00105": (
        "Iterate with the bare uv run pytest.",
        "Iterate with uv run pytest -m 'not eval'; the bare command runs the"
        " live-API eval tests.",
        "deliberate/asserted",
    ),
}


def _bc_match(entry_id: str, score: float) -> dict[str, Any]:
    wrong, fact, label = _BC_ENTRIES[entry_id]
    return {
        "entry_id": entry_id,
        "score": score,
        "wrong_belief": wrong,
        "corrected_fact": fact,
        "provenance_label": label,
    }


def _bc_response(
    reason: str,
    matches: list[dict[str, Any]],
    candidates: int,
    scored: int,
    embed_model: str = _CONTRACT_EMBED_MODEL,
) -> dict[str, Any]:
    return {
        "matches": matches,
        "threshold": BELIEF_CHECK_THRESHOLD,
        "matcher_version": BELIEF_CHECK_MATCHER_VERSION,
        "reason": reason,
        "candidates": candidates,
        "scored": scored,
        "embed_model": embed_model,
    }


def _check_belief_examples(examples: list[tuple[str, dict[str, Any]]]) -> None:
    """Fail closed when the examples stop illustrating the contract."""
    by_name = dict(examples)
    match_top = by_name["match"]["matches"][0]["score"]
    near_top = by_name["near_miss"]["matches"][0]["score"]
    if not match_top >= BELIEF_CHECK_THRESHOLD > near_top:
        raise ValueError(
            "belief_check examples: match must reach the threshold and near_miss"
            f" must not (match={match_top}, near_miss={near_top},"
            f" threshold={BELIEF_CHECK_THRESHOLD})"
        )
    for name, body in examples:
        matches = body["matches"]
        if body["reason"] == "checked":
            scores = [m["score"] for m in matches]
            if not 1 <= len(matches) <= 3:
                raise ValueError(f"belief_check example {name}: 1..3 matches")
            if scores != sorted(scores, reverse=True):
                raise ValueError(f"belief_check example {name}: scores not descending")
            if body["scored"] > body["candidates"]:
                raise ValueError(f"belief_check example {name}: scored > candidates")
        for m in matches:
            label = m["provenance_label"]
            if not m["wrong_belief"] or "observed once" in label:
                raise ValueError(f"belief_check example {name}: untrusted match")
            if "supersedes-edge" in label:
                raise ValueError(f"belief_check example {name}: supersedes-edge match")
    partial = by_name["partial"]
    if not partial["scored"] < partial["candidates"]:
        raise ValueError("belief_check example partial: scored must be < candidates")


def build_belief_check() -> dict[str, Any]:
    """Build the ``belief_check.json`` document."""
    response_schema = BeliefCheckResponse.model_json_schema()
    reasons = response_schema["properties"]["reason"]["enum"]
    raws: list[tuple[str, dict[str, Any]]] = [
        (
            "match",
            _bc_response(
                "checked",
                [
                    _bc_match("kb-00101", 0.9132),
                    _bc_match("kb-00102", 0.6120),
                    _bc_match("kb-00105", 0.5507),
                ],
                3,
                3,
            ),
        ),
        (
            "near_miss",
            _bc_response(
                "checked",
                [
                    _bc_match("kb-00101", 0.8013),
                    _bc_match("kb-00102", 0.5120),
                    _bc_match("kb-00105", 0.4710),
                ],
                3,
                3,
            ),
        ),
        (
            "partial",
            _bc_response(
                "checked",
                [_bc_match("kb-00102", 0.6120), _bc_match("kb-00105", 0.5507)],
                3,
                2,
            ),
        ),
    ]
    for r in reasons:
        if r == "checked":
            continue
        raws.append(
            (
                r,
                _bc_response(
                    r,
                    [],
                    3 if r in ("timeout", "warming", "embed-failed") else 0,
                    0,
                    "" if r == "no-embedder" else _CONTRACT_EMBED_MODEL,
                ),
            )
        )
    _check_belief_examples(raws)
    return {
        **_header("belief_check", "POST", "/api/kb/belief-check"),
        "request": {
            "schema": BeliefCheckRequest.model_json_schema(),
            "examples": [
                {
                    "name": n,
                    "body": BeliefCheckRequest.model_validate(raw).model_dump(
                        mode="json"
                    ),
                }
                for n, raw in _BC_REQUESTS
            ],
        },
        "response": {
            "schema": response_schema,
            "examples": [
                {
                    "name": n,
                    "body": BeliefCheckResponse.model_validate(raw).model_dump(
                        mode="json"
                    ),
                }
                for n, raw in raws
            ],
        },
    }


def build_contracts() -> dict[str, dict[str, Any]]:
    """Return every generated document, keyed by file name."""
    return {
        "prevention.json": build_prevention(),
        "turn.json": build_turn(),
        "event.json": build_event(),
        "belief_check.json": build_belief_check(),
    }


def render(doc: dict[str, Any]) -> str:
    """Render *doc* exactly as it is written to disk."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Return the path of the first difference between *a* and *b*, or None."""
    if type(a) is not type(b):
        return path or "<root>"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            child = k if path == "" else f"{path}.{k}"
            if k not in a or k not in b:
                return child
            found = first_difference(a[k], b[k], child)
            if found is not None:
                return found
        return None
    if isinstance(a, list):
        for i in range(min(len(a), len(b))):
            found = first_difference(a[i], b[i], f"{path}[{i}]")
            if found is not None:
                return found
        if len(a) != len(b):
            return f"{path}[{min(len(a), len(b))}]"
        return None
    return None if a == b else (path or "<root>")


_MAX_KEYS = ("maxLength", "maxItems", "maximum")
_MIN_KEYS = ("minLength", "minItems", "minimum")


def _is_num(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _node_changes(
    old: dict[str, Any], new: dict[str, Any], p: str, request: bool
) -> list[str]:
    out: list[str] = []
    old_props, new_props = old.get("properties"), new.get("properties")
    if isinstance(old_props, dict) and isinstance(new_props, dict):
        out += [
            f"{p}.properties.{k} (property-removed)"
            for k in old_props
            if k not in new_props
        ]
    old_req, new_req = old.get("required", []), new.get("required", [])
    if isinstance(old_req, list) and isinstance(new_req, list):
        if request:
            out += [
                f"{p}.required.{n} (required-added)"
                for n in new_req
                if n not in old_req
            ]
        else:
            out += [
                f"{p}.required.{n} (required-removed)"
                for n in old_req
                if n not in new_req
            ]
    old_enum, new_enum = old.get("enum"), new.get("enum", [])
    if isinstance(old_enum, list) and isinstance(new_enum, list):
        out += [f"{p}.enum.{m} (enum-removed)" for m in old_enum if m not in new_enum]
        if not request:
            out += [f"{p}.enum.{m} (enum-added)" for m in new_enum if m not in old_enum]
    for key in _MAX_KEYS:
        if _is_num(old.get(key)) and _is_num(new.get(key)) and new[key] < old[key]:
            out.append(f"{p}.{key} (limit-tightened)")
    for key in _MIN_KEYS:
        if _is_num(old.get(key)) and _is_num(new.get(key)) and new[key] > old[key]:
            out.append(f"{p}.{key} (limit-tightened)")
    if request:
        out += [
            f"{p}.{key} (limit-added)"
            for key in _MAX_KEYS + _MIN_KEYS
            if key in new and key not in old
        ]
    old_disc, new_disc = old.get("discriminator"), new.get("discriminator")
    if isinstance(old_disc, dict) and isinstance(new_disc, dict):
        old_map, new_map = old_disc.get("mapping"), new_disc.get("mapping")
        if isinstance(old_map, dict) and isinstance(new_map, dict):
            out += [
                f"{p}.discriminator.mapping.{k} (kind-removed)"
                for k in old_map
                if k not in new_map
            ]
    old_defs, new_defs = old.get("$defs"), new.get("$defs")
    if isinstance(old_defs, dict) and isinstance(new_defs, dict):
        out += [f"{p}.$defs.{k} (def-removed)" for k in old_defs if k not in new_defs]
    if (
        isinstance(old.get("type"), str)
        and isinstance(new.get("type"), str)
        and old["type"] != new["type"]
    ):
        out.append(f"{p}.type (type-changed)")
    return out


def _walk(old: dict[str, Any], new: dict[str, Any], p: str, request: bool) -> list[str]:
    out = _node_changes(old, new, p, request)
    for k, ov in old.items():
        if k == "examples" or k not in new:
            continue
        nv = new[k]
        if isinstance(ov, dict) and isinstance(nv, dict):
            out += _walk(ov, nv, f"{p}.{k}", request)
        elif isinstance(ov, list) and isinstance(nv, list):
            for i in range(min(len(ov), len(nv))):
                if isinstance(ov[i], dict) and isinstance(nv[i], dict):
                    out += _walk(ov[i], nv[i], f"{p}.{k}[{i}]", request)
    return out


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Return the detectable breaking changes from *old* to *new* (sorted)."""
    out: list[str] = []
    for zone in ("request", "response"):
        o, n = old.get(zone), new.get(zone)
        if isinstance(o, dict) and isinstance(n, dict):
            out += _walk(o, n, zone, zone == "request")
    try:
        if new["request"]["max_body_bytes"] < old["request"]["max_body_bytes"]:
            out.append("request.max_body_bytes (limit-tightened)")
    except (KeyError, TypeError):
        pass
    for key in ("path", "method"):
        if old.get(key) != new.get(key):
            out.append(f"{key} (endpoint-changed)")
    return sorted(set(out))


def _violations(name: str, doc: dict[str, Any], old: dict[str, Any]) -> list[str]:
    contract = name.removesuffix(".json")
    current = CONTRACT_VERSIONS[contract]
    if old["contract_version"] > current:
        return [
            f"breaking: {name} contract_version went down"
            f" ({old['contract_version']} -> {current})"
        ]
    if current <= old["contract_version"]:
        return [
            f'breaking: {name} at {change}; bump CONTRACT_VERSIONS["{contract}"]'
            for change in breaking_changes(old, doc)
        ]
    return []


def _load(path: Path) -> dict[str, Any] | None:
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(old, dict) or type(old.get("contract_version")) is not int:
        return None
    return old


def main(argv: list[str] | None = None) -> int:
    """Write or check the contract fixtures; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=CONTRACTS_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    contracts = build_contracts()
    problems: list[str] = []
    violations = False
    for name in sorted(contracts):
        doc = contracts[name]
        path = out_dir / name
        if not path.exists():
            if args.check:
                problems.append(f"missing: {name}")
            continue
        old = _load(path)
        if old is None:
            if args.check:
                problems.append(f"drifted: {name} (unparsable)")
            continue
        found = _violations(name, doc, old)
        violations = violations or bool(found)
        problems += found
        if args.check and path.read_text(encoding="utf-8") != render(doc):
            where = first_difference(old, doc) or "formatting only"
            problems.append(f"drifted: {name} (first difference at {where})")
    if args.check:
        problems += [
            f"unexpected: {p.name}"
            for p in sorted(out_dir.glob("*.json"))
            if p.name not in contracts
        ]
        for line in problems:
            print(line)
        if problems:
            return 1
        print("contracts up to date")
        return 0
    if violations:
        for line in problems:
            print(line)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(contracts):
        (out_dir / name).write_text(render(contracts[name]), encoding="utf-8")
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
