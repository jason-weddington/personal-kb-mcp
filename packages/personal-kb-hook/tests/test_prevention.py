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
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli, prevention
from personal_kb_hook.paths import (
    get_event_drop_log_path,
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

    def __call__(self, req: Any, timeout: float = 30.0) -> _Resp:
        url = req.full_url
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        self.calls.append({"url": url, "method": req.get_method(), "body": body})
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
) -> dict[str, Any]:
    return {
        "project": "personal-kb",
        "gate": {"enabled": enabled, "shadow": shadow, "max_denies": 2},
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
    assert cache["denied_resolution_ids"] == []
    assert cache["deny_count"] == 0
    assert cache["pending_retry"] is None
    out = _run(monkeypatch, _pre("git push origin main", "t1"), [])
    assert out
    before = _cache()
    server.prevention = _prevention_body(index=[_cue(), _cue("kb-00002", "git pull")])
    _run(monkeypatch, {**_ss(tmp_path), "source": "compact"}, [])
    after = _cache()
    for key in ("denied_resolution_ids", "deny_count", "pending_retry"):
        assert after[key] == before[key]
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
    assert [r["decision"] for r in later] == ["retry", "skipped_already_denied"]
    assert later[0]["retry_changed_command"] is False
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


def test_cap_after_two_denies(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.prevention = _prevention_body(
        index=[_cue("kb-1", "git push"), _cue("kb-2", "git pull"), _cue("kb-3", "git fetch")]
    )
    _run(monkeypatch, _ss(tmp_path), [])
    assert _run(monkeypatch, _pre("git push", "t1"), [])
    assert _run(monkeypatch, _pre("git pull", "t2"), [])
    assert _run(monkeypatch, _pre("git fetch", "t3"), []) == ""
    assert _rows()[-1]["decision"] == "skipped_cap"
    assert _cache()["deny_count"] == 2


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
    assert cache["deny_count"] == 1
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
    assert _cache()["denied_resolution_ids"] == ["kb-1"]
    assert _cache()["deny_count"] == 1


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
