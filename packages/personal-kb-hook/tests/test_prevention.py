"""Prevention channels in the hook: SessionStart slice, PreToolUse gate, Stop flush.

Every test forces its env with monkeypatch (PERSONAL_KB_URL, PERSONAL_KB_API_KEY,
HEADLESS_BUILD_ENGINE), points HOME at tmp_path (every cache path is under
``~/.cache/personal_kb/``) and patches ``urllib.request.urlopen``.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli, prevention
from personal_kb_hook.paths import (
    get_event_drop_log_path,
    get_failure_context_state_path,
    get_gate_log_path,
    get_prevention_cache_path,
)

if TYPE_CHECKING:
    from pathlib import Path

_URL = "https://kb.example.test"
_KEY = "test-key"
_SID = "sess-1"


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

    def posts(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if "/decisions" in c["url"]]

    def gets(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if "/api/kb/prevention?" in c["url"]]

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


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Server:
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


def _run(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], args: list[str]) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    cli.main(args)
    return out.getvalue()


def _ss(tmp_path: Path, sid: str = _SID) -> dict[str, Any]:
    return {
        "hook_event_name": "SessionStart",
        "session_id": sid,
        "cwd": str(tmp_path / "repo"),
        "source": "startup",
    }


def _pre(command: str, tool_use_id: str, tool: str = "Bash", sid: str = _SID) -> dict[str, Any]:
    key = "command" if tool == "Bash" else "file_path"
    return {
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": tool,
        "tool_use_id": tool_use_id,
        "tool_input": {key: command},
    }


def _stop(tmp_path: Path, sid: str = _SID) -> dict[str, Any]:
    return {
        "hook_event_name": "Stop",
        "session_id": sid,
        "cwd": str(tmp_path / "repo"),
        "transcript_path": "",
    }


def _rows(sid: str = _SID) -> list[dict[str, Any]]:
    path = get_gate_log_path(sid)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _cache(sid: str = _SID) -> dict[str, Any]:
    return json.loads(get_prevention_cache_path(sid).read_text())


def _drops() -> list[dict[str, Any]]:
    path = get_event_drop_log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _maps() -> dict[str, Any]:
    return {
        "projects": [
            {
                "project_ref": "personal-kb",
                "maps": [{"id": "kb-9", "short_title": "Arch", "long_title": "Architecture"}],
            }
        ]
    }


# ─── (a) SessionStart output ─────────────────────────────────────────────────


def test_session_start_slice_only_envelope(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(slice_text="X")
    out = _run(monkeypatch, _ss(tmp_path), ["--format=claude-json"])
    envelope = json.loads(out)
    assert envelope["hookSpecificOutput"]["additionalContext"] == "X"
    armed = [r for r in _rows() if r["decision"] == "armed"]
    assert len(armed) == 1
    assert armed[0]["index_len"] == 1
    assert armed[0]["slice_len"] == 1
    assert armed[0]["decision_id"].startswith(f"cc:{_SID}:armed:")


def test_session_start_slice_plus_directory(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps()
    with monkeypatch.context() as m:
        m.setattr(prevention, "session_start", lambda payload: None)
        baseline = _run(monkeypatch, _ss(tmp_path, "base"), ["--format=text"])
    assert baseline
    server.prevention = _prevention_body(slice_text="X")
    out = _run(monkeypatch, _ss(tmp_path), ["--format=claude-json"])
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"] == "X\n" + baseline
    text_out = _run(monkeypatch, _ss(tmp_path, "s-text"), ["--format=text"])
    assert text_out == "X\n" + baseline


def test_session_start_text_format_bare(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(slice_text="X")
    assert _run(monkeypatch, _ss(tmp_path), ["--format=text"]) == "X"


@pytest.mark.parametrize("fmt", ["--format=text", "--format=claude-json"])
def test_session_start_empty_slice_byte_identical(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: str
) -> None:
    server.maps = _maps()
    with monkeypatch.context() as m:
        m.setattr(prevention, "session_start", lambda payload: None)
        baseline = _run(monkeypatch, _ss(tmp_path, "base"), [fmt])
    out = _run(monkeypatch, _ss(tmp_path, "real"), [fmt])
    assert out == baseline
    assert server.gets()


def test_session_start_slice_survives_suppression(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps()
    server.prevention = _prevention_body(slice_text="X")
    _run(monkeypatch, _ss(tmp_path), ["--format=text"])
    # Same session, same maps: the directory is suppressed, the slice is not.
    resumed = {**_ss(tmp_path), "source": "resume"}
    out = _run(monkeypatch, resumed, ["--format=text"])
    assert out.startswith("X")


def test_session_start_no_project_still_emits_slice(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(slice_text="X")
    payload = {**_ss(tmp_path), "cwd": str(tmp_path)}
    assert _run(monkeypatch, payload, ["--format=text"]) == "X"


def test_session_start_slice_emitted_on_unexpected_error(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(slice_text="X")

    def _boom(*args: Any) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(cli.http_index, "load_index", _boom)
    assert _run(monkeypatch, _ss(tmp_path), ["--format=text"]) == "X"


# ─── (b) cache preserved across re-fetch ─────────────────────────────────────


def test_second_session_start_preserves_state(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    cache = _cache()
    assert cache["deny_state"] == {}
    assert cache["turn_denies"] == 0
    assert cache["pending_retry"] is None
    out = _run(monkeypatch, _pre("git push origin main", "t1"), [])
    assert out
    before = _cache()
    server.prevention = _prevention_body(index=[_cue(), _cue("kb-00002", "git pull")])
    _run(monkeypatch, _ss(tmp_path), [])
    after = _cache()
    for key in ("deny_state", "deny_timestamps", "turn_denies", "pending_retry"):
        assert after[key] == before[key]
    assert "kb-00001" in after["deny_state"]
    assert len(after["index"]) == 2


# ─── (c)/(d)/(e)/(f) gate decisions ──────────────────────────────────────────


def test_deny_then_identical_retry_allowed(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    command = "cd /x && git push origin main"
    out = _run(monkeypatch, _pre(command, "t1"), ["--format=text"])
    hso = json.loads(out)["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    reason = hso["permissionDecisionReason"]
    assert "Push to origin; github is release-only" in reason
    assert reason.endswith(prevention.REASON_SUFFIX)
    denied = [r for r in _rows() if r["decision"] == "denied"]
    assert len(denied) == 1
    assert denied[0]["target"] == command
    assert denied[0]["reason_excerpt"] == reason
    assert denied[0]["decision_id"] == f"cc:{_SID}:t1:denied"
    assert denied[0]["resolution_updated_at"] == "2026-10-07T00:00:00"

    out = _run(monkeypatch, _pre(command, "t2"), [])
    assert out == ""
    later = [r for r in _rows() if r.get("tool_use_id") == "t2"]
    assert [r["decision"] for r in later] == ["retry", "overridden", "skipped_already_denied"]
    assert later[0]["retry_changed_command"] is False
    assert later[1]["resolution_id"] == "kb-00001"
    assert later[2]["reason"] == "overridden"
    assert _cache()["deny_state"]["kb-00001"]["overridden"] is True
    assert later[0]["prior_target"] == command
    assert _cache()["pending_retry"] is None


def test_other_tool_does_not_consume_retry(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push origin main", "t1"), [])
    assert _run(monkeypatch, _pre("/x/a.py", "t2", tool="Read"), []) == ""
    assert _cache()["pending_retry"] is not None
    _run(monkeypatch, _pre("git push --force-with-lease origin main", "t3"), [])
    retry = [r for r in _rows() if r["decision"] == "retry"]
    assert len(retry) == 1
    assert retry[0]["retry_changed_command"] is True
    assert retry[0]["tool_use_id"] == "t3"


def test_cap_after_two_denies_in_one_turn(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(
        index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull"), _cue("kb-3", "git fetch")],
        per_turn=2,
    )
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    assert _run(monkeypatch, _pre("git pull", "t2"), [])
    assert _run(monkeypatch, _pre("git fetch", "t3"), []) == ""
    assert _rows()[-1]["decision"] == "skipped_cap"
    assert _rows()[-1]["reason"] == "per_turn"
    assert _cache()["turn_denies"] == 2


def test_shadow_mode_would_deny(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(shadow=True)
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push origin main", "t1"), []) == ""
    row = _rows()[-1]
    assert row["decision"] == "would_deny"
    assert row["shadow"] is True
    assert row["reason_excerpt"].endswith(prevention.REASON_SUFFIX)
    cache = _cache()
    assert cache["turn_denies"] == 1
    assert cache["deny_state"]["kb-00001"]["overridden"] is False
    assert cache["pending_retry"] is None


def test_observed_once_prefix(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(index=[_cue(observed_once=True, wrong_belief="")])
    _run(monkeypatch, _ss(tmp_path), [])
    reason = json.loads(_run(monkeypatch, _pre("git push", "t1"), []))["hookSpecificOutput"][
        "permissionDecisionReason"
    ]
    assert reason.startswith(prevention.REASON_PREFIX_OBSERVED_ONCE)
    assert "Earlier wrong belief" not in reason


# ─── (g) silent paths ────────────────────────────────────────────────────────


def test_no_cache_is_silent(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(monkeypatch, _pre("git push", "t1"), []) == ""
    assert _rows() == []


def test_gate_disabled_is_silent(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(enabled=False)
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), []) == ""
    assert [r["decision"] for r in _rows()] == ["armed"]


@pytest.mark.parametrize(
    "payload",
    [
        _pre("git pull", "t1"),
        _pre("*.py", "t1", tool="Glob"),
        {k: v for k, v in _pre("git push", "t1").items() if k != "tool_use_id"},
        _pre("npm run build", "t1"),
    ],
)
def test_non_matching_calls_are_silent(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, payload, []) == ""
    assert [r["decision"] for r in _rows()] == ["armed"]


# ─── args_prefix ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "prefix", "denied"),
    [
        ("git remote add origin git@h:r", "add", True),
        ("cd /x && git remote add origin u", "add", True),
        ("git remote -v", "add", False),
        ("git remote show origin", "add", False),
        ("git remote add origin git@h:r", "", True),
        ("git remote -v", "", True),
        ("git remote show origin", "", True),
    ],
)
def test_args_prefix_matching(
    server: _Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    prefix: str,
    denied: bool,
) -> None:
    server.prevention = _prevention_body(index=[_cue(tc="git remote", args_prefix=prefix)])
    _run(monkeypatch, _ss(tmp_path), [])
    out = _run(monkeypatch, _pre(command, "t1"), [])
    assert bool(out) is denied


# ─── (h) no network on PreToolUse ────────────────────────────────────────────


def test_pre_tool_never_calls_urlopen(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    before = len(server.calls)
    for i in range(3):
        _run(monkeypatch, _pre("git push origin main", f"t{i}"), [])
    _run(monkeypatch, _pre("/a.py", "t9", tool="Edit"), [])
    assert len(server.calls) == before


# ─── (i) fetch failure ───────────────────────────────────────────────────────


def test_timeout_without_cache(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.maps = _maps()
    with monkeypatch.context() as m:
        m.setattr(prevention, "session_start", lambda payload: None)
        baseline = _run(monkeypatch, _ss(tmp_path, "base"), ["--format=text"])
    server.prevention = TimeoutError("slow")
    out = _run(monkeypatch, _ss(tmp_path), ["--format=text"])
    assert out == baseline
    assert not get_prevention_cache_path(_SID).exists()
    drops = _drops()
    assert drops[-1]["op"] == "prevention_fetch"
    assert drops[-1]["reason"] == "timeout"
    assert drops[-1]["tool_use_id"] is None


def test_failure_keeps_existing_cache(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    before = get_prevention_cache_path(_SID).read_bytes()
    server.prevention = urllib.error.URLError("refused")
    _run(monkeypatch, _ss(tmp_path), [])
    assert get_prevention_cache_path(_SID).read_bytes() == before
    assert _drops()[-1]["reason"] == "urlerror"


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (urllib.error.HTTPError(_URL, 503, "x", {}, None), "http_503"),  # type: ignore[arg-type]
        (ValueError("bad"), "error"),
        (urllib.error.URLError(TimeoutError()), "timeout"),
    ],
)
def test_fetch_drop_reasons(
    server: _Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
    reason: str,
) -> None:
    server.prevention = exc
    assert prevention.session_start(_ss(tmp_path)) is None
    assert _drops()[-1]["reason"] == reason


def test_no_url_key(server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAL_KB_API_KEY")
    assert prevention.session_start(_ss(tmp_path)) is None
    assert server.calls == []
    assert _drops()[-1]["reason"] == "no_url_key"


def test_malformed_response_is_error(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = {"gate": "nope"}
    assert prevention.session_start(_ss(tmp_path)) is None
    assert _drops()[-1]["reason"] == "error"


def test_session_start_requires_session_id(server: _Server) -> None:
    assert prevention.session_start({"session_id": ""}) is None
    assert server.calls == []


# ─── (j) Stop flush ──────────────────────────────────────────────────────────


def test_stop_flush_success(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    _run(monkeypatch, _pre("git push origin main", "t1"), [])
    assert _run(monkeypatch, _stop(tmp_path), []) == ""
    posts = server.posts()
    assert len(posts) == 1
    decisions = [r["decision"] for r in posts[0]["body"]["rows"]]
    assert decisions == ["armed", "denied", "retry", "summary"]
    abandoned = posts[0]["body"]["rows"][2]
    assert abandoned["retry_changed_command"] is None
    assert abandoned["decision_id"] == f"cc:{_SID}:t1:retry"
    assert abandoned["prior_target"] == "git push origin main"
    assert abandoned["target"] == ""
    summary = posts[0]["body"]["rows"][3]
    assert summary["pre_tool_calls"] == 1
    assert summary["tool"] == "Stop"
    flushing = get_gate_log_path(_SID).with_name(f"gate-log-{_SID}.flushing.jsonl")
    assert not flushing.exists()
    # Refresh re-armed after the flush: one new armed row in a fresh log.
    assert [r["decision"] for r in _rows()] == ["armed"]
    cache = _cache()
    assert cache["pending_retry"] is None
    assert cache["pre_tool_calls"] == 0


def test_stop_flush_failure_keeps_flushing_file(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    server.decisions_status = 500
    _run(monkeypatch, _stop(tmp_path), [])
    flushing = get_gate_log_path(_SID).with_name(f"gate-log-{_SID}.flushing.jsonl")
    assert flushing.exists()
    assert any(d["op"] == "gate_flush" and d["reason"] == "http_500" for d in _drops())
    # Next flush appends the new log behind the kept rows and delivers both.
    server.decisions_status = 200
    prevention.flush_gate_log(_SID)
    assert not flushing.exists()
    rows = server.posts()[-1]["body"]["rows"]
    assert [r["decision"] for r in rows] == ["armed", "summary", "armed"]


def test_flush_chunks_and_skips_malformed(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = get_gate_log_path(_SID)
    lines = [json.dumps({"decision_id": f"d{i}", "decision": "armed"}) for i in range(600)]
    path.write_text("\n".join([*lines, "not json", "[1]"]) + "\n")
    prevention.flush_gate_log(_SID)
    posts = server.posts()
    assert [len(p["body"]["rows"]) for p in posts] == [500, 100]


def test_flush_no_url_key(server: _Server, monkeypatch: pytest.MonkeyPatch) -> None:
    get_gate_log_path(_SID).write_text('{"decision": "armed"}\n')
    monkeypatch.delenv("PERSONAL_KB_API_KEY")
    prevention.flush_gate_log(_SID)
    assert server.calls == []
    assert _drops()[-1] == {**_drops()[-1], "op": "gate_flush", "reason": "no_url_key"}


# ─── (k) Stop refresh delivers a switch flip ─────────────────────────────────


def test_stop_refresh_switch_flip(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull")])
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    server.prevention = _prevention_body(enabled=False)
    _run(monkeypatch, _stop(tmp_path), [])
    assert _cache()["gate"]["enabled"] is False
    assert _run(monkeypatch, _pre("git pull", "t2"), []) == ""
    assert list(_cache()["deny_state"]) == ["kb-1"]
    assert len(_cache()["deny_timestamps"]) == 1


# ─── (l) error inside matching ───────────────────────────────────────────────


def test_error_inside_matching(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])

    def _boom(*args: Any) -> str:
        raise KeyError("x")

    monkeypatch.setattr(prevention.cues_lite, "bash_segments", _boom)
    assert _run(monkeypatch, _pre("git push", "t1"), []) == ""
    cache = _cache()
    assert cache["pre_tool_errors"] == 1
    assert cache["last_error_type"] == "KeyError"
    assert cache["pre_tool_calls"] == 1


# ─── (m) orphan sweep ────────────────────────────────────────────────────────


def _age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_orphan_sweep_bounds(server: _Server) -> None:
    cache_dir = get_gate_log_path("x").parent
    for i in range(25):
        p = cache_dir / f"gate-log-old{i:02d}.jsonl"
        p.write_text('{"decision": "armed"}\n')
        _age(p, 7200)
    fresh = cache_dir / "gate-log-fresh.jsonl"
    fresh.write_text('{"decision": "armed"}\n')
    current = cache_dir / f"gate-log-{_SID}.jsonl"
    current.write_text('{"decision": "armed"}\n')
    _age(current, 7200)
    stale_cache = get_prevention_cache_path("old")
    stale_cache.write_text("{}")
    _age(stale_cache, 8 * 86400)
    young_cache = get_prevention_cache_path("young")
    young_cache.write_text("{}")

    prevention.orphan_sweep(_SID)
    assert len(server.posts()) == 20
    assert fresh.exists()
    assert current.exists()
    assert not stale_cache.exists()
    assert young_cache.exists()
    assert len(list(cache_dir.glob("gate-log-old*.jsonl"))) == 5


def test_orphan_sweep_budget(server: _Server, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = get_gate_log_path("x").parent
    p = cache_dir / "gate-log-old.flushing.jsonl"
    p.write_text('{"decision": "armed"}\n')
    _age(p, 7200)
    monkeypatch.setattr(prevention.telemetry, "_ORPHAN_SWEEP_BUDGET_SECONDS", -1.0)
    prevention.orphan_sweep(_SID)
    assert server.posts() == []
    assert p.exists()


def test_orphan_sweep_via_cli(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = get_gate_log_path("other")
    p.write_text('{"decision": "armed"}\n')
    _age(p, 7200)
    _run(monkeypatch, _ss(tmp_path), [])
    assert not p.exists()
    assert len(server.posts()) == 1


# ─── (n) mode ────────────────────────────────────────────────────────────────


def test_mode_headless_and_interactive(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADLESS_BUILD_ENGINE", "claude-code")
    _run(monkeypatch, _ss(tmp_path), [])
    assert _rows()[-1]["mode"] == "headless"
    assert _rows()[-1]["engine"] == "claude-code"
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE")
    _run(monkeypatch, _ss(tmp_path, "s2"), [])
    assert _rows("s2")[-1]["mode"] == "interactive"


# ─── (o) reason cap ──────────────────────────────────────────────────────────


def test_reason_cap() -> None:
    reason = prevention.build_reason(_cue(corrected_fact="c" * 2000))
    assert len(reason) == 1000
    assert reason.endswith(prevention.REASON_SUFFIX)


# ─── log rotation ────────────────────────────────────────────────────────────


def test_gate_log_rotation(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = get_gate_log_path(_SID)
    path.write_text("x" * 262145)
    _run(monkeypatch, _ss(tmp_path), [])
    assert [r["decision"] for r in _rows()] == ["armed"]
    assert any(d["op"] == "gate_log_rotated" for d in _drops())


def test_drop_log_rotation(server: _Server, tmp_path: Path) -> None:
    drop = get_event_drop_log_path()
    drop.write_text("x" * 262145)
    server.prevention = TimeoutError()
    prevention.session_start(_ss(tmp_path))
    assert len(_drops()) == 1


# ─── compound Bash commands: every segment is matched ────────────────────────


@pytest.mark.parametrize(
    ("tc", "prefix", "command", "denied"),
    [
        (
            "systemctl restart",
            "",
            "caddy validate --config /etc/caddy/Caddyfile && systemctl restart caddy",
            True,
        ),
        ("git push", "github", "git commit -qam x && git push github main", True),
        ("git push", "github", "git commit -qam x && git push origin main", False),
        ("make smoke", "", "cd /app\nmake smoke 2>&1 | tail -5", True),
        ("systemctl restart", "", "sudo systemctl restart caddy; echo ok", True),
        ("git push", "", "git commit -qam x && echo hi", False),
    ],
)
def test_compound_segments_match(
    server: _Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tc: str,
    prefix: str,
    command: str,
    denied: bool,
) -> None:
    server.prevention = _prevention_body(index=[_cue(tc=tc, args_prefix=prefix)])
    _run(monkeypatch, _ss(tmp_path), [])
    out = _run(monkeypatch, _pre(command, "t1"), [])
    assert bool(out) is denied
    if denied:
        rows = [r for r in _rows() if r["decision"] == "denied"]
        assert rows[0]["target_class"] == tc


def test_bash_segments() -> None:
    from personal_kb_hook import cues_lite

    assert cues_lite.bash_segments("cd /a && FOO=1 git push -f github main; ls | wc -l") == [
        ("git push", ["github", "main"]),
        ("ls", []),
        ("wc", []),
    ]
    assert cues_lite.bash_segments("cd /a") == []


# ─── KB_GOTCHA_SLICE ─────────────────────────────────────────────────────────


def test_gotcha_slice_off_still_arms_gate(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_GOTCHA_SLICE", "0")
    server.prevention = _prevention_body(slice_text="X")
    assert prevention.session_start(_ss(tmp_path)) is None
    assert _cache()["index"] == [_cue()]
    out = _run(monkeypatch, _pre("git push origin main", "t1"), ["--format=text"])
    hso = json.loads(out)["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert any(r["decision"] == "denied" and r.get("tool_use_id") == "t1" for r in _rows())


def test_gotcha_slice_unset_returns_slice(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KB_GOTCHA_SLICE", raising=False)
    server.prevention = _prevention_body(slice_text="X")
    assert prevention.session_start(_ss(tmp_path)) == "X"


def test_gotcha_slice_off_keeps_tool_inventory(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "scripts"
    tools.mkdir()
    tool = tools / "mytool"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("KB_TOOL_DIRS", str(tools))
    monkeypatch.setenv("KB_TOOL_INVENTORY", "1")
    monkeypatch.setenv("KB_GOTCHA_SLICE", "0")
    server.prevention = _prevention_body(slice_text="SLICE-X")
    out = _run(monkeypatch, _ss(tmp_path), ["--format=text"])
    assert "mytool" in out
    assert "SLICE-X" not in out


# ─── surprise_capture caching + turn GC ──────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [("shadow", "shadow"), ("on", "on"), (None, "off"), ("ON", "off"), ("off", "off")],
)
def test_surprise_capture_cached_top_level(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: Any, expected: str
) -> None:
    body = _prevention_body()
    if value is not None:
        body["surprise_capture"] = value
    server.prevention = body
    _run(monkeypatch, _ss(tmp_path), ["--format=claude-json"])
    assert _cache()["surprise_capture"] == expected


def test_surprise_capture_ignores_gate_key(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _prevention_body()
    body["gate"]["surprise_capture"] = "on"
    server.prevention = body
    _run(monkeypatch, _ss(tmp_path), ["--format=claude-json"])
    assert _cache()["surprise_capture"] == "off"


def test_surprise_capture_flip_on_stop_and_failed_refresh_keeps_value(
    server: _Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    turn_spawns: list[tuple[str, str]],
) -> None:
    _run(monkeypatch, _ss(tmp_path), ["--format=claude-json"])
    assert _cache()["surprise_capture"] == "off"
    body = _prevention_body()
    body["surprise_capture"] = "shadow"
    server.prevention = body
    _run(monkeypatch, _stop(tmp_path), ["--format=claude-json"])
    assert _cache()["surprise_capture"] == "shadow"
    assert len(turn_spawns) == 1
    server.prevention = urllib.error.URLError("x")
    _run(monkeypatch, _stop(tmp_path), ["--format=claude-json"])
    assert _cache()["surprise_capture"] == "shadow"
    drops = _drops()
    assert len([d for d in drops if d["op"] == "prevention_fetch"]) == 1
    digest = [d for d in drops if d["op"] == "turn_digest"]
    assert len(digest) == 2
    assert {d["reason"] for d in digest} == {"transcript_unreadable"}


def test_orphan_sweep_gcs_turn_files(server: _Server, tmp_path: Path) -> None:
    cache_dir = get_gate_log_path("x").parent
    now = time.time()

    def make(name: str, age_days: float) -> Path:
        p = cache_dir / name
        p.write_text("{}")
        os.utime(p, (now - age_days * 86400, now - age_days * 86400))
        return p

    old_state = make("turn-state-other.json", 92)
    young_state = make("turn-state-young.json", 8)
    month_state = make("turn-state-month.json", 32)
    own_state = make(f"turn-state-{_SID}.json", 92)
    old_log = make("turn-digest-log-other.jsonl", 8)
    young_log = make("turn-digest-log-young.jsonl", 1)
    prevention.orphan_sweep(_SID)
    assert not old_state.exists()
    assert young_state.exists()
    assert month_state.exists()  # outlives the server's 90-day digest retention
    assert own_state.exists()
    assert not old_log.exists()
    assert young_log.exists()


# ─── PostToolUseFailure: failure context ─────────────────────────────────────

_COMPOUND = "git commit -qam x && git push github main"


def _fail(command: str = "git push github main", tool_use_id: str = "toolu_f1") -> dict[str, Any]:
    return {
        "hook_event_name": "PostToolUseFailure",
        "session_id": _SID,
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
        "tool_input": {"command": command},
        "error": "Exit code 1",
    }


def _fc_state() -> Path:
    return get_failure_context_state_path(_SID)


def _fc_rows(decision: str) -> list[dict[str, Any]]:
    return [r for r in _rows() if r["decision"] == decision]


@pytest.fixture
def armed(server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Server:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    prevention.session_start(_ss(tmp_path))
    return server


def test_failure_context_state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert get_failure_context_state_path("sess-1") == (
        tmp_path / ".cache/personal_kb/failure-context-sess-1.json"
    )


def test_build_failure_context_text() -> None:
    assert prevention.build_failure_context(_cue()) == (
        "KB: this failure matches a known correction: Push to origin; github is release-only"
        " Earlier wrong belief: push to github [deliberate/observed; kb-00001]"
    )
    capped = prevention.build_failure_context(_cue(corrected_fact="c" * 2000))
    assert len(capped) == 1000
    assert prevention.REASON_SUFFIX not in capped
    assert prevention.build_failure_context(_cue(observed_once=True)).startswith(
        prevention.FAILURE_CONTEXT_PREFIX_OBSERVED_ONCE
    )


def test_build_reason_uses_shared_body() -> None:
    entry = _cue()
    assert prevention.build_reason(entry) == (
        prevention.REASON_PREFIX + prevention._reason_body(entry) + prevention.REASON_SUFFIX
    )


def test_find_match_order_and_miss() -> None:
    entries = [_cue("kb-a", tc="git commit"), _cue("kb-b")]
    match, tc = prevention._find_match("Bash", _COMPOUND, entries)
    assert match is not None
    assert (match["resolution_id"], tc) == ("kb-a", "git commit")
    assert prevention._find_match("Bash", "ls -la", entries) == (None, "")
    assert prevention._find_match("Read", "/x", entries) == (None, "")


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "maybe"])
def test_failure_context_switch_off_is_silent(
    armed: _Server, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("KB_FAILURE_CONTEXT", raising=False)
    else:
        monkeypatch.setenv("KB_FAILURE_CONTEXT", value)
    before = _rows()
    drops = _drops()
    assert prevention.failure_context(_fail()) is None
    assert _rows() == before
    assert _drops() == drops
    assert not _fc_state().exists()


@pytest.mark.parametrize("value", ["1", "true", " YES ", "on"])
def test_failure_context_switch_on_delivers(
    armed: _Server, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", value)
    assert prevention.failure_context(_fail()) is not None


def _mutate_cache(**kw: Any) -> None:
    cache = _cache()
    cache.update(kw)
    get_prevention_cache_path(_SID).write_text(json.dumps(cache))


@pytest.mark.parametrize(
    "case",
    [
        "no_sid",
        "not_bash",
        "no_tuid",
        "interrupt",
        "empty_target",
        "no_cache",
        "disabled",
        "pending",
        "no_match",
    ],
)
def test_failure_context_silent_cases(armed: _Server, case: str) -> None:
    payload = _fail()
    if case == "no_sid":
        payload["session_id"] = ""
    elif case == "not_bash":
        payload["tool_name"] = "Read"
        payload["tool_input"] = {"file_path": "git push github main"}
    elif case == "no_tuid":
        payload["tool_use_id"] = ""
    elif case == "interrupt":
        payload["is_interrupt"] = True
    elif case == "empty_target":
        payload["tool_input"] = "nope"
    elif case == "no_cache":
        get_prevention_cache_path(_SID).unlink()
    elif case == "disabled":
        _mutate_cache(gate={"enabled": False, "shadow": False, "max_denies": 2})
    elif case == "pending":
        _mutate_cache(pending_retry={"tool": "Bash", "tool_use_id": "toolu_f1"})
    elif case == "no_match":
        payload["tool_input"] = {"command": "ls -la"}
    before = _rows()
    drops = _drops()
    calls = len(armed.calls)
    assert prevention.failure_context(payload) is None
    assert _rows() == before
    assert _drops() == drops
    assert len(armed.calls) == calls
    assert not _fc_state().exists()


def test_failure_context_delivers_without_touching_cache(armed: _Server) -> None:
    cache_before = get_prevention_cache_path(_SID).read_bytes()
    calls = len(armed.calls)
    out = prevention.failure_context(_fail())
    assert out is not None
    assert json.loads(out) == {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUseFailure",
            "additionalContext": prevention.build_failure_context(_cue()),
        }
    }
    assert get_prevention_cache_path(_SID).read_bytes() == cache_before
    assert len(armed.calls) == calls
    assert list(json.loads(_fc_state().read_text())["delivered"]) == ["kb-00001"]
    (row,) = _fc_rows("failure_context")
    assert row["decision_id"] == f"cc:{_SID}:toolu_f1:failure_context"
    assert row["reason_excerpt"] == prevention.build_failure_context(_cue())
    assert row["resolution_id"] == "kb-00001"
    assert row["target_class"] == "git push"
    assert row["tool"] == "Bash"
    assert row["shadow"] is False


def test_failure_context_ignores_shadow(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    server.prevention = _prevention_body(shadow=True)
    prevention.session_start(_ss(tmp_path))
    assert prevention.failure_context(_fail()) is not None
    assert _fc_rows("failure_context")[0]["shadow"] is False


def test_failure_context_ignores_deny_budget(armed: _Server) -> None:
    _mutate_cache(
        turn_denies=5,
        deny_state={"kb-00001": {"last_deny_ts": "2099-01-01T00:00:00+00:00"}},
    )
    assert prevention.failure_context(_fail()) is not None


def test_failure_context_repeat(armed: _Server) -> None:
    assert prevention.failure_context(_fail()) is not None
    state = _fc_state().read_bytes()
    assert prevention.failure_context(_fail(tool_use_id="toolu_f2")) is None
    assert _fc_state().read_bytes() == state
    assert len(_fc_rows("failure_context")) == 1
    (rep,) = _fc_rows("failure_context_repeat")
    assert rep.get("reason_excerpt") is None
    assert rep["decision_id"] == f"cc:{_SID}:toolu_f2:failure_context_repeat"
    assert rep["resolution_id"] == "kb-00001"
    assert rep["shadow"] is False


@pytest.mark.parametrize(
    "content", ["[]", '{"delivered_resolution_ids": "kb-00001"}', '{"delivered": 3}', "{bad"]
)
def test_failure_context_malformed_state_delivers(armed: _Server, content: str) -> None:
    _fc_state().write_text(content)
    assert prevention.failure_context(_fail()) is not None
    assert list(json.loads(_fc_state().read_text())["delivered"]) == ["kb-00001"]


def test_failure_context_state_keeps_str_members(armed: _Server) -> None:
    ts = prevention.telemetry.now_ts()
    _fc_state().write_text(json.dumps({"delivered": {"kb-9": ts, "kb-8": 3}}))
    assert prevention._load_failure_state(_SID) == {"kb-9": ts}
    assert prevention.failure_context(_fail()) is not None
    assert set(json.loads(_fc_state().read_text())["delivered"]) == {"kb-9", "kb-00001"}


def test_failure_context_redelivers_after_window(armed: _Server) -> None:
    assert prevention.failure_context(_fail()) is not None
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    _fc_state().write_text(json.dumps({"delivered": {"kb-00001": old}}))
    assert prevention.failure_context(_fail(tool_use_id="toolu_f2")) is not None
    assert json.loads(_fc_state().read_text())["delivered"]["kb-00001"] > old
    assert len(_fc_rows("failure_context")) == 2


def test_failure_context_unparseable_timestamp_is_expired(armed: _Server) -> None:
    _fc_state().write_text(json.dumps({"delivered": {"kb-00001": "garbage"}}))
    assert prevention.failure_context(_fail()) is not None


def test_failure_context_legacy_state_suppresses(armed: _Server) -> None:
    _fc_state().write_text(json.dumps({"delivered_resolution_ids": ["kb-00001"]}))
    assert prevention.failure_context(_fail()) is None
    assert len(_fc_rows("failure_context_repeat")) == 1


@pytest.mark.parametrize("source", ["compact", "resume", "clear", "startup"])
def test_failure_context_rearm_on_session_start(
    armed: _Server, tmp_path: Path, source: str
) -> None:
    assert prevention.failure_context(_fail()) is not None
    prevention.session_start({**_ss(tmp_path), "source": source})
    if source == "startup":
        assert _fc_state().exists()
        assert prevention.failure_context(_fail(tool_use_id="toolu_f2")) is None
        assert not _fc_rows("rearmed")
        return
    assert not _fc_state().exists()
    (rearmed,) = _fc_rows("rearmed")
    assert rearmed["failure_context_cleared"] == 1
    assert prevention.failure_context(_fail(tool_use_id="toolu_f2")) is not None


def test_failure_context_state_write_failure(
    armed: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: Any) -> None:
        raise OSError("disk")

    monkeypatch.setattr(prevention, "_atomic_write", _boom)
    assert prevention.failure_context(_fail()) is None
    assert _fc_rows("failure_context") == []
    assert _drops()[-1]["op"] == "failure_context"
    assert _drops()[-1]["reason"] == "state_write"


def test_failure_context_record_failure_still_delivers(
    armed: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = prevention._record

    def _flaky(session_id: str, cache: dict[str, Any], decision: str, **kw: Any) -> Any:
        if decision == "failure_context":
            raise OSError("log")
        return real(session_id, cache, decision, **kw)

    monkeypatch.setattr(prevention, "_record", _flaky)
    assert prevention.failure_context(_fail()) is not None
    assert _fc_state().exists()
    assert any(
        d.get("op") == "failure_context" and d["reason"] == "record_failed" for d in _drops()
    )


def test_failure_context_error_row(armed: _Server, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(prevention, "_find_match", _boom)
    assert prevention.failure_context(_fail()) is None
    (row,) = _fc_rows("failure_context_error")
    assert row["last_error_type"] == "RuntimeError"
    assert row["decision_id"] == f"cc:{_SID}:toolu_f1:failure_context_error"
    assert row["shadow"] is False
    assert not _fc_state().exists()
    fc_drops = [d for d in _drops() if d.get("op") == "failure_context"]
    assert len(fc_drops) == 1
    assert fc_drops[0]["reason"] == "RuntimeError"


def test_failure_context_error_and_record_failure(
    armed: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(prevention, "_find_match", _boom)
    monkeypatch.setattr(prevention, "_record", _boom)
    assert prevention.failure_context(_fail()) is None
    assert _drops()[-1]["op"] == "failure_context"
    assert _drops()[-1]["reason"] == "RuntimeError"


def _cli_fail(tmp_path: Path, tool_use_id: str = "toolu_f1") -> dict[str, Any]:
    payload = _fail(_COMPOUND, tool_use_id)
    payload["cwd"] = str(tmp_path / "repo")
    return payload


def _expected_envelope() -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUseFailure",
            "additionalContext": prevention.build_failure_context(_cue()),
        }
    }


def test_cli_failure_context_delivery_and_repeat(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    _run(monkeypatch, _ss(tmp_path), [])
    out = _run(monkeypatch, _cli_fail(tmp_path), [])
    assert json.loads(out) == _expected_envelope()
    assert len(server.events()) == 1
    assert _fc_rows("failure_context")[0]["target_class"] == "git push"
    out2 = _run(monkeypatch, _cli_fail(tmp_path, "toolu_f2"), [])
    assert out2 == ""
    assert len(server.events()) == 2


def test_cli_failure_context_event_post_failure(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    _run(monkeypatch, _ss(tmp_path), [])
    server.event_exc = urllib.error.URLError("down")
    out = _run(monkeypatch, _cli_fail(tmp_path), [])
    assert json.loads(out) == _expected_envelope()
    last = _drops()[-1]
    assert last["reason"] == "urlerror"
    assert last["tool_use_id"] == "toolu_f1"
    assert "op" not in last


def test_cli_failure_context_off_by_default(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _cli_fail(tmp_path), []) == ""
    assert len(server.events()) == 1


def test_cli_failure_context_ignores_format(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    _run(monkeypatch, _ss(tmp_path), [])
    out = _run(monkeypatch, _cli_fail(tmp_path), ["--format=text"])
    assert out == json.dumps(_expected_envelope())


def test_orphan_sweep_gcs_failure_context_state(server: _Server) -> None:
    old = get_failure_context_state_path("old")
    old.write_text("{}")
    _age(old, 8 * 86400)
    young = get_failure_context_state_path(_SID)
    young.write_text("{}")
    _age(young, 86400)
    prevention.orphan_sweep(_SID)
    assert not old.exists()
    assert young.exists()


# ─── deny state, rate limits, re-arming, override ───────────────────────────


def _ago(hours: float) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _ups(sid: str = _SID) -> dict[str, Any]:
    return {"hook_event_name": "UserPromptSubmit", "session_id": sid, "prompt": "hi"}


def _decisions(tool_use_id: str) -> list[str]:
    return [r["decision"] for r in _rows() if r.get("tool_use_id") == tool_use_id]


def test_cache_carries_new_gate_settings(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(per_turn=3, per_hour=9, rearm_hours=48)
    _run(monkeypatch, _ss(tmp_path), [])
    assert _cache()["gate"] == {
        "enabled": True,
        "shadow": False,
        "max_denies_per_turn": 3,
        "max_denies_per_hour": 9,
        "rearm_hours": 48,
    }


@pytest.mark.parametrize("bad", [None, 0, 1001, "5", True])
def test_gate_settings_fall_back_to_defaults(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    body = _prevention_body()
    for key in ("max_denies_per_turn", "max_denies_per_hour", "rearm_hours"):
        if bad is None:
            del body["gate"][key]
        else:
            body["gate"][key] = bad
    server.prevention = body
    _run(monkeypatch, _ss(tmp_path), [])
    gate = _cache()["gate"]
    assert gate["max_denies_per_turn"] == prevention.DEFAULT_MAX_DENIES_PER_TURN
    assert gate["max_denies_per_hour"] == prevention.DEFAULT_MAX_DENIES_PER_HOUR
    assert gate["rearm_hours"] == prevention.DEFAULT_REARM_HOURS


def test_no_redeny_before_rearm_hours(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    _run(monkeypatch, _ups(), [])
    _mutate_cache(
        pending_retry=None,
        deny_state={"kb-00001": {"last_deny_ts": _ago(23.9), "overridden": False}},
    )
    assert _run(monkeypatch, _pre("git push", "t2"), []) == ""
    assert _decisions("t2") == ["skipped_already_denied"]
    assert _rows()[-1]["reason"] == "not_rearmed"


def test_redeny_after_rearm_hours(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    _run(monkeypatch, _stop(tmp_path), [])
    _mutate_cache(
        deny_state={"kb-00001": {"last_deny_ts": _ago(24.1), "overridden": False}},
        deny_timestamps=[_ago(24.1)],
    )
    assert _run(monkeypatch, _pre("git push", "t2"), [])
    assert _decisions("t2") == ["denied"]
    assert _cache()["deny_timestamps"] != [] and len(_cache()["deny_timestamps"]) == 1


def test_override_suppression_lifted_by_rearm_hours(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    assert _run(monkeypatch, _pre("git push", "t2"), []) == ""
    assert _cache()["deny_state"]["kb-00001"]["overridden"] is True
    _run(monkeypatch, _stop(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t3"), []) == ""
    assert _decisions("t3") == ["skipped_already_denied"]
    _mutate_cache(deny_state={"kb-00001": {"last_deny_ts": _ago(25), "overridden": True}})
    assert _run(monkeypatch, _pre("git push", "t4"), [])
    assert _cache()["deny_state"]["kb-00001"]["overridden"] is False


def test_changed_retry_is_not_an_override(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push github main", "t1"), [])
    _run(monkeypatch, _pre("git push origin main", "t2"), [])
    assert "overridden" not in [r["decision"] for r in _rows()]
    assert _cache()["deny_state"]["kb-00001"]["overridden"] is False


@pytest.mark.parametrize("source", ["compact", "resume", "clear"])
def test_session_start_rearms_on_memory_loss(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    server.prevention = _prevention_body(
        index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull")], per_turn=5
    )
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    assert _run(monkeypatch, _pre("git push", "t2"), []) == ""  # override kb-1
    assert _run(monkeypatch, _pre("git pull", "t3"), [])
    assert set(_cache()["deny_state"]) == {"kb-1", "kb-2"}
    _run(monkeypatch, {**_ss(tmp_path), "source": source}, [])
    assert _cache()["deny_state"] == {}
    rearmed = [r for r in _rows() if r["decision"] == "rearmed"]
    assert len(rearmed) == 1
    assert rearmed[0]["source"] == source
    assert rearmed[0]["tool"] == "SessionStart"
    assert rearmed[0]["cleared"] == 2
    _run(monkeypatch, _ups(), [])
    _mutate_cache(pending_retry=None)
    assert _run(monkeypatch, _pre("git push", "t4"), [])  # overridden lesson fires again


def test_session_start_startup_does_not_rearm(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    _run(monkeypatch, _ss(tmp_path), [])
    assert "kb-00001" in _cache()["deny_state"]
    assert "rearmed" not in [r["decision"] for r in _rows()]


def test_rearm_without_cache_still_logs(server: _Server, tmp_path: Path) -> None:
    server.prevention = OSError("down")
    prevention.session_start({**_ss(tmp_path), "source": "resume"})
    assert [r["decision"] for r in _rows()] == ["rearmed"]
    assert not get_prevention_cache_path(_SID).exists()


def test_per_turn_cap_resets_on_user_prompt_and_stop(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(
        index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull"), _cue("kb-3", "git fetch")]
    )
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    _mutate_cache(pending_retry=None)
    assert _run(monkeypatch, _pre("git pull", "t2"), []) == ""
    assert _rows()[-1]["decision"] == "skipped_cap"
    assert _rows()[-1]["reason"] == "per_turn"
    _run(monkeypatch, _ups(), [])
    assert _cache()["turn_denies"] == 0
    assert _run(monkeypatch, _pre("git pull", "t3"), [])
    _mutate_cache(pending_retry=None)
    assert _run(monkeypatch, _pre("git fetch", "t4"), []) == ""
    assert _rows()[-1]["reason"] == "per_turn"
    _run(monkeypatch, _stop(tmp_path), [])
    assert _cache()["turn_denies"] == 0
    assert _run(monkeypatch, _pre("git fetch", "t5"), [])


def test_per_hour_cap(server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server.prevention = _prevention_body(
        index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull")], per_hour=2
    )
    _run(monkeypatch, _ss(tmp_path), [])
    _mutate_cache(deny_timestamps=[_ago(0.5), _ago(0.9)])
    assert _run(monkeypatch, _pre("git push", "t1"), []) == ""
    assert _rows()[-1]["decision"] == "skipped_cap"
    assert _rows()[-1]["reason"] == "per_hour"
    assert "kb-1" not in _cache()["deny_state"]
    _mutate_cache(deny_timestamps=[_ago(0.5), _ago(1.1)])
    assert _run(monkeypatch, _pre("git push", "t2"), [])
    assert len(_cache()["deny_timestamps"]) == 2
    _run(monkeypatch, _ups(), [])
    _mutate_cache(pending_retry=None)
    assert _run(monkeypatch, _pre("git pull", "t3"), []) == ""
    assert _rows()[-1]["reason"] == "per_hour"


def test_deny_timestamps_bounded(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    _mutate_cache(deny_timestamps=[_ago(2)] * 5000)
    assert len(prevention._load_cache(_SID)["deny_timestamps"]) == 1000  # type: ignore[index]
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    assert len(_cache()["deny_timestamps"]) == 1


def test_shadow_mode_consumes_same_limits(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(
        shadow=True, index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull")], per_hour=1
    )
    _run(monkeypatch, _ss(tmp_path), [])
    _run(monkeypatch, _pre("git push", "t1"), [])
    _run(monkeypatch, _pre("git push", "t2"), [])
    _run(monkeypatch, _pre("git pull", "t3"), [])
    _run(monkeypatch, _ups(), [])
    _run(monkeypatch, _pre("git pull", "t4"), [])
    assert _decisions("t1") == ["would_deny"]
    assert _decisions("t2") == ["skipped_already_denied"]
    assert _decisions("t3") == ["skipped_cap"]
    assert _decisions("t4") == ["skipped_cap"]
    reasons = [r["reason"] for r in _rows() if r["decision"] == "skipped_cap"]
    assert reasons == ["per_turn", "per_hour"]
    _run(monkeypatch, {**_ss(tmp_path), "source": "compact"}, [])
    _mutate_cache(deny_timestamps=[])
    _run(monkeypatch, _pre("git push", "t5"), [])
    assert _decisions("t5") == ["would_deny"]


def test_legacy_cache_migrates_on_read(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    cache = _cache()
    for key in ("deny_state", "deny_timestamps", "turn_denies"):
        cache.pop(key)
    cache["denied_resolution_ids"] = ["kb-00001", 7]
    cache["deny_count"] = 2
    cache["gate"] = {"enabled": True, "shadow": False, "max_denies": 2}
    get_prevention_cache_path(_SID).write_text(json.dumps(cache))
    loaded = prevention._load_cache(_SID)
    assert loaded is not None
    state = loaded["deny_state"]
    assert list(state) == ["kb-00001"]
    assert state["kb-00001"]["overridden"] is False
    assert prevention._parse_ts(state["kb-00001"]["last_deny_ts"]) is not None
    assert "denied_resolution_ids" not in loaded
    assert "deny_count" not in loaded
    assert loaded["turn_denies"] == 0
    # legacy gate (no new settings): defaults apply; kb-00001 is not re-armed yet
    assert _run(monkeypatch, _pre("git push", "t1"), []) == ""
    assert _decisions("t1") == ["skipped_already_denied"]
    assert "denied_resolution_ids" not in _cache()


def test_legacy_cache_malformed_state_tolerated(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(monkeypatch, _ss(tmp_path), [])
    _mutate_cache(
        deny_state={"kb-00001": {"last_deny_ts": "garbage"}, "kb-x": 3},
        deny_timestamps="nope",
        turn_denies="x",
    )
    assert _run(monkeypatch, _pre("git push", "t1"), [])


def test_cli_user_prompt_submit_without_cache_is_quiet(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    prevention.new_turn({"session_id": ""})
    prevention.new_turn(_ups())
    assert not get_prevention_cache_path(_SID).exists()
