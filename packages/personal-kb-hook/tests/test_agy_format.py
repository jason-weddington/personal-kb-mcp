"""Tests for the Antigravity (agy) output mode: ``--format=agy-json --event <Event>``.

agy (the Gemini agent harness) ships one camelCase JSON object on stdin and
expects one JSON object on stdout, and its stdin carries NO event name — so the
mode is input plus output adapter around the existing Claude Code pipeline:
:mod:`personal_kb_hook.agy` translates both ways and ``cli._run_agy`` drives
``cli._pipeline`` with the translated payload.

Every test isolates ``HOME`` under ``tmp_path`` (all cache paths live under
``~/.cache/personal_kb/``) and replaces ``urllib.request.urlopen`` with the
same ``_Server`` fake ``tests/test_prevention.py`` uses (prevention GET,
decisions POST, maps index).
"""

from __future__ import annotations

import copy
import inspect
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import agy, cli, events, listener, paths, prevention, telemetry, turn_digest

_URL = "https://kb.example.test"
_KEY = "test-key"
_SID = "c1"
_AGY_EVENTS = ("SessionStart", "PreInvocation", "PreToolUse", "PostToolUse", "Stop")
_DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "intelligent-kb.md"

# The maps index is a plain HTTP body here (``_Server`` answers it).
_KB1 = {"id": "kb-1", "short_title": "alpha", "long_title": ""}
_KB2 = {"id": "kb-2", "short_title": "beta", "long_title": ""}
_EM_DASH = "\u2014"


# ─── fake server (same routing as tests/test_prevention.py) ─────────────────


class _Resp:
    def __init__(self, status: int = 200, body: bytes = b"{}") -> None:
        self.status = status
        self._body = body

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Server:
    """Fake urlopen routing by path: prevention GET, decisions POST, maps index."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.prevention: dict[str, Any] | BaseException = _prevention_body()
        self.decisions_status = 200
        self.maps: dict[str, Any] = {"projects": []}
        self.event_exc: BaseException | None = None

    def __call__(self, req: Any, timeout: float = 30.0) -> _Resp:
        url = req.full_url
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        self.calls.append({"url": url, "method": req.get_method(), "body": body})
        if "/api/kb/event" in url and self.event_exc is not None:
            raise self.event_exc
        if "/api/kb/prevention/decisions" in url:
            return _Resp(self.decisions_status)
        if "/api/kb/prevention?" in url:
            if isinstance(self.prevention, BaseException):
                raise self.prevention
            return _Resp(200, json.dumps(self.prevention).encode())
        return _Resp(200, json.dumps(self.maps).encode())

    def events(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["url"].endswith("/api/kb/event")]


def _cue(rid: str = "kb-00001", tc: str = "git push", **kw: Any) -> dict[str, Any]:
    out = {
        "resolution_id": rid,
        "updated_at": "2026-10-07T00:00:00",
        "tool": "Bash",
        "target_class": tc,
        "args_prefix": "",
        "wrong_belief": "push to github",
        "corrected_fact": "Push to origin; github is release-only",
        "evidence": "",
        "provenance_label": "deliberate/observed",
        "observed_once": False,
    }
    out.update(kw)
    return out


def _prevention_body(
    *,
    enabled: bool = True,
    shadow: bool = False,
    index: list[dict[str, Any]] | None = None,
    slice_text: str = "",
    per_turn: int = 1,
    per_hour: int = 6,
    rearm_hours: int = 24,
) -> dict[str, Any]:
    return {
        "project": "personal-kb",
        "gate": {
            "enabled": enabled,
            "shadow": shadow,
            "max_denies": 1000,
            "max_denies_per_turn": per_turn,
            "max_denies_per_hour": per_hour,
            "rearm_hours": rearm_hours,
        },
        "index": [_cue()] if index is None else index,
        "slice": [{"entry_id": "kb-1"}] if slice_text else [],
        "slice_text": slice_text,
        "diagnostics": {},
    }


def _maps(*maps: dict[str, str]) -> dict[str, Any]:
    return {"projects": [{"project_ref": "personal-kb", "maps": list(maps)}]}


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Server:
    """Isolate HOME/XDG under tmp_path and serve HTTP from the fake server."""
    monkeypatch.setenv("PERSONAL_KB_URL", _URL)
    monkeypatch.setenv("PERSONAL_KB_API_KEY", _KEY)
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    monkeypatch.delenv("PERSONAL_KB_LISTENER", raising=False)
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "kb" / "knowledge.db"))
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True)
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    srv = _Server()
    monkeypatch.setattr(urllib.request, "urlopen", srv)
    return srv


# ─── helpers ────────────────────────────────────────────────────────────────


def _run(monkeypatch: pytest.MonkeyPatch, payload: Any, args: list[str]) -> str:
    """Run ``cli.main`` with ``payload`` on stdin; return captured stdout."""
    stdin_text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    cli.main(args)
    return out.getvalue()


def _agy(tmp_path: Path, event: str | None = None, **over: Any) -> list[str]:
    """The agy argv for ``event`` (omitted entirely when ``event is None``)."""
    args = ["--format=agy-json"]
    if event is not None:
        args.append(f"--event={event}")
    return args


def _payload(tmp_path: Path, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "conversationId": _SID,
        "workspacePaths": [str(tmp_path / "repo")],
        "transcriptPath": "/t.jsonl",
        "modelName": "gemini-2.5-pro",
        "lastUserInput": "hi",
    }
    base.update(over)
    return base


def _tool_payload(
    tmp_path: Path, name: str, args: dict[str, Any], step: int | None = 1, **over: Any
) -> dict[str, Any]:
    payload = _payload(tmp_path, **over)
    payload["toolCall"] = {"name": name, "args": args}
    if step is not None:
        payload["stepIdx"] = step
    return payload


def _push(step: int = 1) -> dict[str, Any]:
    return {"name": "run_command", "args": {"CommandLine": "git push origin main"}}


def _pre_payload(tmp_path: Path, step: int = 1, sid: str = _SID) -> dict[str, Any]:
    payload = _tool_payload(tmp_path, "run_command", {"CommandLine": "git push origin main"}, step)
    payload["conversationId"] = sid
    return payload


def _rows(sid: str = _SID) -> list[dict[str, Any]]:
    path = paths.get_gate_log_path(sid)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _telemetry_rows(sid: str = _SID) -> list[dict[str, Any]]:
    path = paths.get_whisper_log_path(sid)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _seed_telemetry(sid: str, map_id: str) -> None:
    telemetry.append_row(
        sid,
        {
            "session_id": sid,
            "host": "h",
            "surface": "roster",
            "map_id": map_id,
            "source_kb": "personal",
            "cwd_project": "personal-kb",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )


def _claude_session_start(tmp_path: Path, sid: str) -> dict[str, Any]:
    return {
        "hook_event_name": "SessionStart",
        "session_id": sid,
        "cwd": str(tmp_path / "repo"),
        "source": "startup",
    }


# ─── AC 1: CLI surface ──────────────────────────────────────────────────────


def test_formats_tuple() -> None:
    assert cli._FORMATS == ("text", "claude-json", "agy-json")


def test_parse_args_format_choices() -> None:
    assert cli._parse_args(["--format=agy-json"]).format == "agy-json"
    assert cli._parse_args([]).format == "text"
    with pytest.raises(SystemExit):
        cli._parse_args(["--format=bogus"])


def test_parse_args_event_is_free_form() -> None:
    args = cli._parse_args(["--format=agy-json", "--event=Bogus"])
    assert args.event == "Bogus"
    assert cli._parse_args([]).event is None


def test_parse_args_docstring_names_event() -> None:
    doc = inspect.getdoc(cli._parse_args) or ""
    assert "The only supported flag is ``--format``" not in doc
    assert "--event" in doc


# ─── AC 2: dispatch order ──────────────────────────────────────────────────


def test_empty_stdin_still_writes_one_object(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(monkeypatch, "", ["--format=agy-json"])
    assert out == "{}"


def test_no_event_empty_stdin_writes_one_object(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(monkeypatch, "", ["--format=agy-json"]) == "{}"


# ─── AC 3: unknown --event never reaches the pipeline ──────────────────────


def test_unknown_event_never_reaches_pipeline(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = _payload(tmp_path)
    bogus = _run(monkeypatch, payload, ["--format=agy-json", "--event=Bogus"])
    assert json.loads(bogus) == {}
    ok = _run(monkeypatch, payload, ["--format=agy-json", "--event=SessionStart"])
    injected = json.loads(ok)["injectSteps"][0]["systemMessage"]["systemMessage"]
    assert "Maps for personal-kb" in injected


# ─── AC 4: constants ───────────────────────────────────────────────────────


def test_constants_by_value() -> None:
    assert isinstance(agy.AGY_EVENTS, frozenset)
    assert set(agy.AGY_EVENTS) == {
        "SessionStart",
        "PreInvocation",
        "PreToolUse",
        "PostToolUse",
        "Stop",
    }
    assert agy.EVENT_MAP == {
        "SessionStart": "SessionStart",
        "PreInvocation": "UserPromptSubmit",
        "PreToolUse": "PreToolUse",
        "PostToolUse": "PostToolUse",
        "Stop": None,
    }
    assert agy.TOOL_NAME_MAP == {
        "run_command": "Bash",
        "read_file": "Read",
        "view_file": "Read",
        "write_to_file": "Write",
    }
    assert set(agy.EVENT_MAP) == set(agy.AGY_EVENTS)
    assert "call_mcp_tool" not in agy.TOOL_NAME_MAP


# ─── AC 5: event mapping ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("event", "error", "expected"),
    [
        ("SessionStart", None, "SessionStart"),
        ("SessionStart", "", "SessionStart"),
        ("SessionStart", "boom", "SessionStart"),
        ("PreInvocation", None, "UserPromptSubmit"),
        ("PreInvocation", "", "UserPromptSubmit"),
        ("PreInvocation", "boom", "UserPromptSubmit"),
        ("PreToolUse", None, "PreToolUse"),
        ("PreToolUse", "", "PreToolUse"),
        ("PreToolUse", "boom", "PreToolUse"),
        ("PostToolUse", None, "PostToolUse"),
        ("PostToolUse", "", "PostToolUse"),
        ("PostToolUse", "boom", "PostToolUseFailure"),
        ("Stop", None, None),
        ("Stop", "", None),
        ("Stop", "boom", None),
    ],
)
def test_claude_event_for(event: str, error: str | None, expected: str | None) -> None:
    payload: dict[str, Any] = {}
    if error is not None:
        payload["error"] = error
    assert agy.claude_event_for(event, payload) == expected


@pytest.mark.parametrize("error", [None, "", 123, True])
def test_claude_event_for_nested_error_is_ignored(error: Any) -> None:
    payload: dict[str, Any] = {"toolCall": {"name": "run_command", "error": "boom"}}
    if error is not None:
        payload["error"] = error
    assert agy.claude_event_for("PostToolUse", payload) == "PostToolUse"


@pytest.mark.parametrize("event", [None, "", "preinvocation", "Bogus"])
def test_claude_event_for_unknown_event(event: Any) -> None:
    assert agy.claude_event_for(event, {"error": "boom"}) is None


# ─── AC 6: common-field mapping ────────────────────────────────────────────


def test_to_claude_payload_empty() -> None:
    assert agy.to_claude_payload({}, "SessionStart") == {
        "hook_event_name": "SessionStart",
        "harness": "antigravity",
    }


def test_to_claude_payload_full() -> None:
    payload = {
        "conversationId": "c1",
        "workspacePaths": ["/repo", "/other"],
        "transcriptPath": "/t.jsonl",
        "lastUserInput": "hi",
    }
    assert agy.to_claude_payload(payload, "SessionStart") == {
        "session_id": "c1",
        "cwd": "/repo",
        "transcript_path": "/t.jsonl",
        "prompt": "hi",
        "hook_event_name": "SessionStart",
        "harness": "antigravity",
    }


@pytest.mark.parametrize(
    ("over", "absent"),
    [
        ({"conversationId": ""}, "session_id"),
        ({"conversationId": 7}, "session_id"),
        ({"workspacePaths": []}, "cwd"),
        ({"workspacePaths": [""]}, "cwd"),
        ({"workspacePaths": "/repo"}, "cwd"),
        ({"transcriptPath": ""}, "transcript_path"),
        ({"lastUserInput": None}, "prompt"),
    ],
)
def test_to_claude_payload_omits_unusable_keys(over: dict[str, Any], absent: str) -> None:
    payload: dict[str, Any] = {
        "conversationId": "c1",
        "workspacePaths": ["/repo"],
        "transcriptPath": "/t.jsonl",
        "lastUserInput": "hi",
        **over,
    }
    result = agy.to_claude_payload(payload, "SessionStart")
    assert absent not in result
    assert None not in result.values()


def test_to_claude_payload_does_not_mutate() -> None:
    payload = _tool_payload(Path("/repo"), "run_command", {"CommandLine": "git push"})
    snapshot = copy.deepcopy(payload)
    result = agy.to_claude_payload(payload, "PreToolUse")
    assert result is not payload
    assert payload == snapshot


# ─── AC 7: tool-event presence rules ───────────────────────────────────────


@pytest.mark.parametrize("claude_event", ["PreToolUse", "PostToolUse", "PostToolUseFailure"])
@pytest.mark.parametrize(
    ("tool_call", "expected_input"),
    [
        (None, {}),
        ({}, {}),
        ({"name": "", "args": {}}, {}),
        ({"name": 7, "args": {"X": 1}}, {"X": 1}),
        ({"args": {"X": 1}}, {"X": 1}),
    ],
)
def test_tool_keys_omitted_without_usable_name(
    claude_event: str, tool_call: dict[str, Any] | None, expected_input: dict[str, Any]
) -> None:
    payload: dict[str, Any] = {"conversationId": "c1", "stepIdx": 3}
    if tool_call is not None:
        payload["toolCall"] = tool_call
    result = agy.to_claude_payload(payload, claude_event)
    assert "tool_name" not in result
    assert result["tool_input"] == expected_input
    assert result["tool_use_id"] == "agy-3"
    raw_args = (tool_call or {}).get("args")
    if isinstance(raw_args, dict):
        assert result["tool_input"] is not raw_args


@pytest.mark.parametrize("claude_event", ["SessionStart", "UserPromptSubmit", None])
def test_non_tool_events_carry_no_tool_keys(claude_event: str | None) -> None:
    payload: dict[str, Any] = {"conversationId": "c1", "stepIdx": 3, "toolCall": _push()}
    result = agy.to_claude_payload(payload, claude_event)
    for key in ("tool_name", "tool_input", "tool_use_id"):
        assert key not in result
    assert result["hook_event_name"] is claude_event


# ─── AC 8: tool-name and argument mapping ──────────────────────────────────


_ALIAS_ARGS = {"CommandLine": "git push origin main", "WaitMsBeforeAsync": 0, "toolSummary": "push"}
_ALIAS_OUT = {"command": "git push origin main", "WaitMsBeforeAsync": 0, "toolSummary": "push"}
_READ_ARGS = {"Path": "/repo/a.py", "StartLine": 3}
_READ_OUT = {"file_path": "/repo/a.py", "StartLine": 3}

_TOOL_TABLE: list[tuple[str, Any, str, dict[str, Any]]] = [
    ("run_command", _ALIAS_ARGS, "Bash", _ALIAS_OUT),
    ("read_file", _READ_ARGS, "Read", _READ_OUT),
    ("view_file", _READ_ARGS, "Read", _READ_OUT),
    ("write_to_file", {"TargetFile": "/repo/b.py"}, "Write", {"file_path": "/repo/b.py"}),
    ("write_to_file", {"Path": "/repo/b.py"}, "Write", {"file_path": "/repo/b.py"}),
    (
        "write_to_file",
        {"TargetFile": "/repo/b.py", "Path": "/repo/c.py"},
        "Write",
        {"file_path": "/repo/b.py"},
    ),
    (
        "call_mcp_tool",
        {
            "ServerName": "personal-kb",
            "ToolName": "kb_get",
            "Arguments": {"entry_id": ["kb-1"]},
        },
        "mcp__personal-kb__kb_get",
        {"entry_id": ["kb-1"]},
    ),
    (
        "call_mcp_tool",
        {"ToolName": "kb_get", "Arguments": {"entry_id": "kb-1"}},
        "call_mcp_tool",
        {"entry_id": "kb-1"},
    ),
    (
        "call_mcp_tool",
        {"ServerName": "personal-kb", "Arguments": {"entry_id": "kb-1"}},
        "call_mcp_tool",
        {"entry_id": "kb-1"},
    ),
    (
        "call_mcp_tool",
        {"ServerName": "personal-kb", "ToolName": "kb_get", "Arguments": "nope"},
        "mcp__personal-kb__kb_get",
        {},
    ),
    ("call_mcp_tool", {}, "call_mcp_tool", {}),
    ("some_new_tool", {"X": 1}, "some_new_tool", {"X": 1}),
    ("read_file", {"StartLine": 3}, "Read", {"StartLine": 3}),
    ("run_command", {"WaitMsBeforeAsync": 0}, "Bash", {"WaitMsBeforeAsync": 0}),
    ("run_command", "not-a-dict", "Bash", {}),
    ("some_new_tool", None, "some_new_tool", {}),
]


@pytest.mark.parametrize(("name", "args", "expected_name", "expected_input"), _TOOL_TABLE)
def test_tool_name_and_argument_mapping(
    name: str, args: Any, expected_name: str, expected_input: dict[str, Any]
) -> None:
    payload: dict[str, Any] = {
        "conversationId": "c1",
        "stepIdx": 2,
        "toolCall": {"name": name, "args": args},
    }
    result = agy.to_claude_payload(payload, "PreToolUse")
    assert result["tool_name"] == expected_name
    assert result["tool_input"] == expected_input
    assert result["tool_input"] is not args


# ─── AC 9: synthesized tool_use_id ─────────────────────────────────────────


def test_tool_use_id_from_step_idx(tmp_path: Path) -> None:
    third = agy.to_claude_payload(
        _tool_payload(tmp_path, "run_command", _ALIAS_ARGS, 3), "PreToolUse"
    )
    assert third["tool_use_id"] == "agy-3"
    zeroth = agy.to_claude_payload(
        _tool_payload(tmp_path, "run_command", _ALIAS_ARGS, 0), "PreToolUse"
    )
    assert zeroth["tool_use_id"] == "agy-0"


@pytest.mark.parametrize("step", ["absent", None, "3", True])
def test_tool_use_id_hash_fallback(tmp_path: Path, step: Any) -> None:
    payload = _tool_payload(tmp_path, "run_command", {"CommandLine": "git push origin main"}, None)
    if step != "absent":
        payload["stepIdx"] = step
    result = agy.to_claude_payload(payload, "PreToolUse")
    assert re.fullmatch(r"agy-[0-9a-f]{12}", str(result["tool_use_id"]))


def test_tool_use_id_ignores_nested_step_idx(tmp_path: Path) -> None:
    payload = _tool_payload(tmp_path, "run_command", {"CommandLine": "git push"}, None)
    payload["toolCall"]["stepIdx"] = 5
    result = agy.to_claude_payload(payload, "PreToolUse")
    assert re.fullmatch(r"agy-[0-9a-f]{12}", str(result["tool_use_id"]))


def test_tool_use_id_is_deterministic_and_distinct(tmp_path: Path) -> None:
    payload = _tool_payload(tmp_path, "run_command", {"CommandLine": "git push origin main"}, None)
    first = agy.to_claude_payload(payload, "PreToolUse")["tool_use_id"]
    second = agy.to_claude_payload(payload, "PreToolUse")["tool_use_id"]
    assert first == second
    other = _tool_payload(tmp_path, "run_command", {"CommandLine": "git push origin dev"}, None)
    assert agy.to_claude_payload(other, "PreToolUse")["tool_use_id"] != first


# ─── AC 10: render_reply ───────────────────────────────────────────────────


def _envelope(event: str, **fields: Any) -> str:
    return json.dumps({"hookSpecificOutput": {"hookEventName": event, **fields}})


def test_render_reply_direct_rows() -> None:
    assert agy.render_reply("SessionStart", "") == {}
    assert agy.render_reply("SessionStart", "not-json") == {}
    assert (
        agy.render_reply(
            "SessionStart", '{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}'
        )
        == {}
    )
    assert agy.render_reply("PreToolUse", "") == {"decision": "allow"}
    deny = _envelope("PreToolUse", permissionDecision="deny", permissionDecisionReason="X")
    assert agy.render_reply("PreToolUse", deny) == {"decision": "allow", "reason": "X"}
    failure = _envelope("PostToolUseFailure", additionalContext="Y")
    assert agy.render_reply("PostToolUse", failure) == {}
    assert agy.render_reply("PostToolUseFailure", failure) == {}
    assert agy.render_reply(None, "anything") == {}


def test_render_reply_never_copies_a_deny() -> None:
    deny = _envelope(
        "PreToolUse",
        permissionDecision="deny",
        permissionDecisionReason="X",
        additionalContext="Y",
    )
    reply = agy.render_reply("PreToolUse", deny)
    assert reply == {"decision": "allow", "reason": "X"}
    assert "deny" not in json.dumps(reply)


def test_render_reply_ignores_blank_and_non_dict_shapes() -> None:
    assert agy.render_reply("SessionStart", "[]") == {}
    assert agy.render_reply("SessionStart", _envelope("SessionStart", additionalContext="")) == {}
    assert agy.render_reply("UserPromptSubmit", _envelope("UserPromptSubmit")) == {}


# ─── AC 11: end-to-end reply shapes ────────────────────────────────────────


def test_session_start_reply_shape(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = _payload(tmp_path)
    out = _run(monkeypatch, payload, _agy(tmp_path, "SessionStart"))
    reply = json.loads(out)
    assert list(reply) == ["injectSteps"]
    assert len(reply["injectSteps"]) == 1
    step = reply["injectSteps"][0]
    assert list(step) == ["systemMessage"]
    assert list(step["systemMessage"]) == ["systemMessage"]
    injected = step["systemMessage"]["systemMessage"]
    assert "Maps for personal-kb" in injected

    twin = agy.to_claude_payload(payload, "SessionStart")
    twin["session_id"] = "baseline"
    baseline = _run(monkeypatch, twin, ["--format=claude-json"])
    assert json.loads(baseline)["hookSpecificOutput"]["additionalContext"] == injected


def test_pre_invocation_reply_shape(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = _payload(tmp_path)
    out = _run(monkeypatch, payload, _agy(tmp_path, "PreInvocation"))
    assert "systemMessage" not in out
    reply = json.loads(out)
    assert list(reply) == ["injectSteps"]
    step = reply["injectSteps"][0]
    assert list(step) == ["ephemeralMessage"]
    injected = step["ephemeralMessage"]
    assert "Maps for personal-kb" in injected

    twin = agy.to_claude_payload(payload, "UserPromptSubmit")
    twin["session_id"] = "baseline"
    baseline = _run(monkeypatch, twin, ["--format=claude-json"])
    assert json.loads(baseline)["hookSpecificOutput"]["additionalContext"] == injected


def test_tool_and_stop_replies_are_empty(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for event in ("PostToolUse", "Stop"):
        out = _run(monkeypatch, _payload(tmp_path), _agy(tmp_path, event))
        assert json.loads(out) == {}
    failed = _tool_payload(
        tmp_path, "run_command", {"CommandLine": "git push origin main"}, 1, error="boom"
    )
    assert json.loads(_run(monkeypatch, failed, _agy(tmp_path, "PostToolUse"))) == {}


def test_nothing_to_inject_is_an_empty_object(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "noproj"
    plain.mkdir()
    payload = _payload(tmp_path, workspacePaths=[str(plain)])
    for event in ("SessionStart", "PreInvocation"):
        assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, event))) == {}
    server.prevention = _prevention_body(slice_text="X")
    out = _run(monkeypatch, payload, _agy(tmp_path, "SessionStart"))
    assert json.loads(out) == {"injectSteps": [{"systemMessage": {"systemMessage": "X"}}]}


# ─── AC 12: suppression reused; PreInvocation is a delta fallback ──────────


def test_pre_invocation_first_emission_then_suppressed(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = _payload(tmp_path, conversationId="c9")
    args = _agy(tmp_path, "PreInvocation")
    first = json.loads(_run(monkeypatch, payload, args))
    assert "Maps for personal-kb" in first["injectSteps"][0]["ephemeralMessage"]
    assert json.loads(_run(monkeypatch, payload, args)) == {}


def test_pre_invocation_new_map_delta(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = _payload(tmp_path, conversationId="c-delta")
    args = _agy(tmp_path, "PreInvocation")
    assert json.loads(_run(monkeypatch, payload, args))
    server.maps = _maps(_KB1, _KB2)
    out = _run(monkeypatch, payload, args)
    assert json.loads(out) == {
        "injectSteps": [{"ephemeralMessage": f"New map {_EM_DASH} personal/[kb-2] beta"}]
    }


def test_pre_invocation_after_session_start_is_suppressed(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = _payload(tmp_path)
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "SessionStart")))
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PreInvocation"))) == {}


# ─── AC 13: PreToolUse always allows ───────────────────────────────────────


@pytest.mark.parametrize("case", ["empty", "not-json", "list", "no-cache", "no-toolcall", "raised"])
def test_pre_tool_use_always_allow(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    stdin: Any
    if case == "empty":
        stdin = ""
    elif case == "not-json":
        stdin = "not-json"
    elif case == "list":
        stdin = "[]"
    else:
        payload = _payload(tmp_path)
        if case != "no-toolcall":
            payload["toolCall"] = _push()
        payload["stepIdx"] = 1
        stdin = json.dumps(payload)
    if case == "raised":

        def _boom(_payload: Any) -> str | None:
            raise RuntimeError("boom")

        monkeypatch.setattr(prevention, "pre_tool", _boom)
    out = _run(monkeypatch, stdin, _agy(tmp_path, "PreToolUse"))
    obj = json.loads(out)
    assert isinstance(obj, dict)
    assert obj["decision"] == "allow"
    assert obj == {"decision": "allow"}
    assert "deny" not in out
    assert "injectSteps" not in out


# ─── AC 14: a real gate deny becomes an allow with the gate text ───────────


def test_gate_deny_becomes_allow_with_reason(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _payload(tmp_path), _agy(tmp_path, "SessionStart"))
    out = _run(monkeypatch, _pre_payload(tmp_path, 1), _agy(tmp_path, "PreToolUse"))
    obj = json.loads(out)
    assert set(obj) == {"decision", "reason"}
    assert obj["decision"] == "allow"
    reason = obj["reason"]
    assert reason.startswith(prevention.REASON_PREFIX)
    assert reason.endswith(prevention.REASON_SUFFIX)
    assert '"deny"' not in out

    _run(monkeypatch, _claude_session_start(tmp_path, "twin"), ["--format=claude-json"])
    twin = {
        "hook_event_name": "PreToolUse",
        "session_id": "twin",
        "tool_name": "Bash",
        "tool_use_id": "t-1",
        "tool_input": {"command": "git push origin main"},
    }
    baseline = _run(monkeypatch, twin, ["--format=claude-json"])
    hso = json.loads(baseline)["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert hso["permissionDecisionReason"] == reason


# ─── AC 15: shadow stays bare-allow; deny-once never blocks ────────────────


def test_shadow_mode_replies_bare_allow(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(shadow=True)
    _run(monkeypatch, _payload(tmp_path), _agy(tmp_path, "SessionStart"))
    out = _run(monkeypatch, _pre_payload(tmp_path, 1), _agy(tmp_path, "PreToolUse"))
    assert json.loads(out) == {"decision": "allow"}
    assert "reason" not in out
    assert _rows()[-1]["decision"] == "would_deny"
    assert _rows()[-1]["shadow"] is True


def test_deny_once_then_identical_retry_allows(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _payload(tmp_path), _agy(tmp_path, "SessionStart"))
    denied = json.loads(_run(monkeypatch, _pre_payload(tmp_path, 1), _agy(tmp_path, "PreToolUse")))
    assert denied["decision"] == "allow"
    assert denied["reason"]
    again = _run(monkeypatch, _pre_payload(tmp_path, 2), _agy(tmp_path, "PreToolUse"))
    assert json.loads(again) == {"decision": "allow"}
    later = [r for r in _rows() if r.get("tool_use_id") == "agy-2"]
    assert [r["decision"] for r in later] == ["retry", "overridden", "skipped_already_denied"]


# ─── AC 16: PostToolUse kb_get consumption ─────────────────────────────────


def test_post_tool_use_consumes_kb_get(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_telemetry(_SID, "kb-A")
    _seed_telemetry(_SID, "kb-B")
    payload = _tool_payload(
        tmp_path,
        "call_mcp_tool",
        {
            "ServerName": "personal-kb",
            "ToolName": "kb_get",
            "Arguments": {"entry_id": "kb-A"},
        },
        4,
    )
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PostToolUse"))) == {}
    rows = {r["map_id"]: r for r in _telemetry_rows()}
    assert rows["kb-A"]["consumed"] is True
    assert rows["kb-A"]["consumed_ts"] is not None
    assert rows["kb-B"]["consumed"] is False

    plain = _tool_payload(tmp_path, "run_command", {"CommandLine": "ls"}, 5)
    assert json.loads(_run(monkeypatch, plain, _agy(tmp_path, "PostToolUse"))) == {}
    rows = {r["map_id"]: r for r in _telemetry_rows()}
    assert rows["kb-B"]["consumed"] is False


# ─── AC 17: the failure path ───────────────────────────────────────────────


def test_failure_path_calls_post_failure_once(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(events, "post_failure", calls.append)
    payload = _tool_payload(
        tmp_path,
        "run_command",
        {"CommandLine": "git push origin main"},
        3,
        error="bash: nope",
    )
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PostToolUse"))) == {}
    assert len(calls) == 1
    body = calls[0]
    assert body["hook_event_name"] == "PostToolUseFailure"
    assert body["session_id"] == "c1"
    assert body["tool_use_id"] == "agy-3"
    assert body["tool_name"] == "Bash"
    assert body["error"] == "bash: nope"


def test_failure_event_posts_antigravity_harness(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _tool_payload(
        tmp_path,
        "run_command",
        {"CommandLine": "git push origin main"},
        3,
        error="bash: nope",
    )
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PostToolUse"))) == {}
    posts = server.events()
    assert len(posts) == 1
    body = posts[0]["body"]
    assert body["harness"] == "antigravity"
    assert body["tool_name"] == "Bash"
    assert body["event_id"] == "cc:c1:agy-3"
    assert body["tool_input"] == {"command": "git push origin main"}


def test_empty_error_is_the_success_path(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(events, "post_failure", calls.append)
    payload = _tool_payload(
        tmp_path, "run_command", {"CommandLine": "git push origin main"}, 3, error=""
    )
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PostToolUse"))) == {}
    assert calls == []
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PostToolUse"))) == {}


def test_failure_context_reply_is_discarded_but_claude_json_prints(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    _run(monkeypatch, _payload(tmp_path), _agy(tmp_path, "SessionStart"))
    seen: list[str | None] = []
    real = prevention.failure_context

    def spy(payload: dict[str, Any]) -> str | None:
        value = real(payload)
        seen.append(value)
        return value

    monkeypatch.setattr(prevention, "failure_context", spy)
    payload = _tool_payload(
        tmp_path, "run_command", {"CommandLine": "git push origin main"}, 2, error="boom"
    )
    assert json.loads(_run(monkeypatch, payload, _agy(tmp_path, "PostToolUse"))) == {}
    assert seen and seen[0] is not None

    twin_src = {**payload, "conversationId": "twin"}
    twin_event = agy.claude_event_for("PostToolUse", twin_src)
    assert twin_event == "PostToolUseFailure"
    twin = agy.to_claude_payload(twin_src, twin_event)
    _run(monkeypatch, _claude_session_start(tmp_path, "twin"), ["--format=claude-json"])
    baseline = _run(monkeypatch, twin, ["--format=claude-json"])
    envelope = json.loads(baseline)["hookSpecificOutput"]
    assert envelope["hookEventName"] == "PostToolUseFailure"
    assert envelope["additionalContext"]


# ─── AC 18: no usable conversationId never touches shared state ────────────


@pytest.mark.parametrize("over", [{}, {"conversationId": ""}, {"conversationId": 7}])
def test_missing_conversation_id_writes_nothing(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over: dict[str, Any]
) -> None:
    server.maps = _maps(_KB1)
    payload: dict[str, Any] = {"workspacePaths": [str(tmp_path / "repo")], **over}
    for event, expected in (
        ("SessionStart", {}),
        ("PreToolUse", {"decision": "allow"}),
        ("Stop", {}),
    ):
        out = _run(monkeypatch, payload, _agy(tmp_path, event))
        assert json.loads(out) == expected
    assert list((tmp_path / ".cache" / "personal_kb").iterdir()) == []


# ─── AC 19: Stop is a no-op with a reply ───────────────────────────────────


def test_stop_touches_nothing(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("PERSONAL_KB_LISTENER", "KB_LISTENER_HEADLESS", "HEADLESS_BUILD_ENGINE"):
        monkeypatch.delenv(name, raising=False)
    recorded: list[str] = []

    def _spy(label: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            recorded.append(label)

        return record

    monkeypatch.setattr(telemetry, "flush_session", _spy("flush_session"))
    monkeypatch.setattr(prevention, "stop", _spy("prevention.stop"))
    monkeypatch.setattr(prevention, "flush_gate_log", _spy("flush_gate_log"))
    monkeypatch.setattr(prevention, "refresh", _spy("refresh"))
    monkeypatch.setattr(turn_digest, "stop", _spy("turn_digest.stop"))
    monkeypatch.setattr(listener, "spawn_worker", _spy("spawn_worker"))

    popen_calls: list[Any] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popen_calls.append((a, k)))

    def _no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Stop must not make a network call")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)

    out = _run(monkeypatch, _payload(tmp_path), _agy(tmp_path, "Stop"))
    assert json.loads(out) == {}
    assert recorded == []
    assert popen_calls == []
    assert not paths.get_turn_state_path(_SID).exists()
    assert not paths.get_gate_log_path(_SID).exists()


# ─── AC 20: stdout discipline ──────────────────────────────────────────────


@pytest.mark.parametrize("event", _AGY_EVENTS)
@pytest.mark.parametrize("shape", ["valid", "empty", "not-json"])
def test_stdout_is_exactly_one_object(
    server: _Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    shape: str,
) -> None:
    plain = tmp_path / "noproj"
    plain.mkdir(exist_ok=True)
    valid: dict[str, Any] = {
        "conversationId": "c-stdout",
        "workspacePaths": [str(plain)],
        "transcriptPath": "/t.jsonl",
        "lastUserInput": "hi",
    }
    if event in ("PreToolUse", "PostToolUse"):
        valid["toolCall"] = {"name": "run_command", "args": {"CommandLine": "ls"}}
        valid["stepIdx"] = 1
    stdin = {"valid": json.dumps(valid), "empty": "", "not-json": "not-json"}[shape]
    out = _run(monkeypatch, stdin, _agy(tmp_path, event))
    assert out
    assert "\n" not in out
    obj = json.loads(out)
    assert isinstance(obj, dict)
    assert obj == ({"decision": "allow"} if event == "PreToolUse" else {})


@pytest.mark.parametrize("shape", ["valid", "empty", "not-json"])
def test_no_event_always_writes_empty_object(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    valid: dict[str, Any] = {
        "conversationId": "c-no-event",
        "workspacePaths": [str(tmp_path / "repo")],
        "transcriptPath": "/t.jsonl",
        "lastUserInput": "hi",
        "toolCall": {"name": "run_command", "args": {"CommandLine": "git push origin main"}},
    }
    stdin = {"valid": json.dumps(valid), "empty": "", "not-json": "not-json"}[shape]
    out = _run(monkeypatch, stdin, ["--format=agy-json"])
    assert out == "{}"


# ─── AC 21: text and claude-json are unchanged ─────────────────────────────


def test_text_and_claude_json_ignore_event(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps(_KB1)
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "unchanged",
        "cwd": str(tmp_path / "repo"),
        "source": "compact",
    }
    text_plain = _run(monkeypatch, payload, ["--format=text"])
    text_event = _run(monkeypatch, payload, ["--format=text", "--event=SessionStart"])
    assert text_plain
    assert text_plain == text_event

    json_plain = _run(monkeypatch, payload, ["--format=claude-json"])
    json_event = _run(monkeypatch, payload, ["--format=claude-json", "--event=PreInvocation"])
    assert json_plain
    assert json_plain == json_event
    assert json.loads(json_plain)["hookSpecificOutput"]["additionalContext"] == text_plain


def test_claude_pre_tool_use_text_format_prints_deny_envelope(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _claude_session_start(tmp_path, "gate"), ["--format=claude-json"])
    pre = {
        "hook_event_name": "PreToolUse",
        "session_id": "gate",
        "tool_name": "Bash",
        "tool_use_id": "t-1",
        "tool_input": {"command": "git push origin main"},
    }
    out = _run(monkeypatch, pre, ["--format=text"])
    hso = json.loads(out)["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert hso["permissionDecisionReason"].startswith(prevention.REASON_PREFIX)


def test_supported_events_unchanged() -> None:
    assert isinstance(cli._SUPPORTED_EVENTS, frozenset)
    assert set(cli._SUPPORTED_EVENTS) == {
        "PreToolUse",
        "SessionStart",
        "UserPromptSubmit",
        "Stop",
        "SubagentStop",
        "PostToolUse",
        "PostToolUseFailure",
    }


# ─── AC 22c: the documented hooks.json sample ──────────────────────────────


def _agy_section() -> str:
    text = _DOC_PATH.read_text(encoding="utf-8")
    start = text.index("### Antigravity (agy) mode")
    return text[start:]


def _hooks_sample() -> dict[str, Any]:
    match = re.search(r"```json\n(.*?)\n```", _agy_section(), re.S)
    assert match is not None, "no ```json fence in the agy section"
    sample: dict[str, Any] = json.loads(match.group(1))
    return sample


def test_documented_hooks_json_sample() -> None:
    sample = _hooks_sample()
    assert list(sample) == ["personal-kb"]
    inner = sample["personal-kb"]
    assert inner["enabled"] is True
    assert set(inner) == {
        "enabled",
        "SessionStart",
        "PreInvocation",
        "PreToolUse",
        "PostToolUse",
        "Stop",
    }
    for event in _AGY_EVENTS:
        entries = inner[event]
        assert len(entries) == 1
        assert entries[0]["matcher"] == "*"
        hooks = entries[0]["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["type"] == "command"
        assert f"--format=agy-json --event={event}" in hooks[0]["command"]
        if event in ("PreInvocation", "PreToolUse", "PostToolUse"):
            assert hooks[0]["timeout"] == 5
        else:
            assert "timeout" not in hooks[0]


def test_agy_section_documents_the_three_gaps() -> None:
    section = _agy_section()
    assert "turn digest" in section
    assert "listener" in section
    assert "gate log" in section
    assert "cache refresh" in section or "refresh the prevention cache" in section
    assert "failure context" in section
    assert "prevention.new_turn" in section
    for name in agy.TOOL_NAME_MAP:
        assert f"`{name}`" in section
    assert "call_mcp_tool" in section
    assert "mcp__<ServerName>__<ToolName>" in section


# ─── AC 24e: the failure body stamps the adapter's harness ────────────────


def _event_payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "session_id": "c1",
        "tool_use_id": "t-1",
        "tool_name": "Bash",
        "tool_input": {"command": "git push origin main"},
        "error": "boom",
    }
    base.update(over)
    return base


def test_build_body_harness_rows() -> None:
    assert events._build_body(_event_payload())["harness"] == "claude-code"
    assert events._build_body(_event_payload(harness="antigravity"))["harness"] == "antigravity"
    assert events._build_body(_event_payload(harness=""))["harness"] == "claude-code"
    assert events._build_body(_event_payload(harness=7))["harness"] == "claude-code"
