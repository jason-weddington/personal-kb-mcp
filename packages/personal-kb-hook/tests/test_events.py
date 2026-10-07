"""PostToolUseFailure → POST /api/kb/event (record-only failure-cue feed).

Every test forces its env explicitly with monkeypatch; nothing relies on the
ambient environment. ``urllib.request.urlopen`` is always patched.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli, events, paths

if TYPE_CHECKING:
    from pathlib import Path

_URL = "https://kb.example.test"
_KEY = "test-key"


class _MockResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Recorder:
    def __init__(self, status: int = 200, exc: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.status = status
        self.exc = exc

    def __call__(self, req: Any, timeout: float = 30.0) -> _MockResponse:
        self.calls.append(
            {
                "url": req.full_url,
                "headers": dict(req.headers),
                "body": json.loads(req.data.decode("utf-8")),
                "method": req.method,
                "timeout": timeout,
            }
        )
        if self.exc is not None:
            raise self.exc
        return _MockResponse(self.status)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Remote URL + key, no build engine, drop log redirected to tmp_path."""
    monkeypatch.setenv("PERSONAL_KB_URL", _URL)
    monkeypatch.setenv("PERSONAL_KB_API_KEY", _KEY)
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    drop_log = tmp_path / "cache" / "event-drops.jsonl"
    monkeypatch.setattr(events, "get_event_drop_log_path", lambda: drop_log)
    return drop_log


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": "sess-1",
        "transcript_path": "/nonexistent/t.jsonl",
        "cwd": "/nonexistent/dir/personal_kb",
        "permission_mode": "default",
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "git push", "timeout": 5, "nested": {"x": 1}},
        "tool_use_id": "toolu_abc",
        "error": "Exit code 1\nerror: failed to push",
        "is_interrupt": False,
        "duration_ms": 42,
    }
    payload.update(overrides)
    return payload


def _run_cli(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> tuple[int, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    stdout_buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout_buf)
    rc = 0
    try:
        cli.main(["--format=claude-json"])
    except SystemExit as exc:  # pragma: no cover - main never exits non-zero
        rc = int(exc.code or 0)
    return rc, stdout_buf.getvalue()


def _drops(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_post_failure_end_to_end(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    monkeypatch.setattr(events, "_hook_version", lambda: "9.9.9")
    rc, out = _run_cli(monkeypatch, _payload())
    assert rc == 0
    assert out == ""
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["url"] == f"{_URL}/api/kb/event"
    assert call["method"] == "POST"
    assert call["timeout"] == 1.5
    assert call["headers"]["Authorization"] == f"Bearer {_KEY}"
    assert call["headers"]["Content-type"] == "application/json"
    body = call["body"]
    ts = body.pop("ts")
    assert isinstance(ts, str) and ts
    assert body == {
        "type": "post_tool",
        "event_id": "cc:sess-1:toolu_abc",
        "session_id": "sess-1",
        "harness": "claude-code",
        "mode": "interactive",
        "engine": None,
        "host": socket.gethostname(),
        "hook_version": "9.9.9",
        "cwd": "/nonexistent/dir/personal_kb",
        "project": None,
        "tool_name": "Bash",
        "tool_input": {"command": "git push", "timeout": 5},
        "tool_use_id": "toolu_abc",
        "error": "Exit code 1\nerror: failed to push",
        "is_error": True,
        "is_interrupt": False,
        "duration_ms": 42,
    }
    assert _drops(env) == []


def test_project_resolved_from_kb_project(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / ".kb_project").write_text("personal-kb\n")
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    _run_cli(monkeypatch, _payload(cwd=str(repo / "sub"), is_interrupt=True))
    body = recorder.calls[0]["body"]
    assert body["project"] == "personal-kb"
    assert body["is_interrupt"] is True


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (urllib.error.URLError("refused"), "urlerror"),
        (TimeoutError("slow"), "timeout"),
        (urllib.error.HTTPError(_URL, 404, "nf", {}, None), "http_404"),  # type: ignore[arg-type]
        (urllib.error.URLError(TimeoutError("slow")), "timeout"),
        (ValueError("weird"), "error"),
    ],
)
def test_post_failure_drops(
    env: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException, reason: str
) -> None:
    recorder = _Recorder(exc=exc)
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    rc, out = _run_cli(monkeypatch, _payload())
    assert rc == 0
    assert out == ""
    assert len(recorder.calls) == 1
    [drop] = _drops(env)
    assert drop["reason"] == reason
    assert drop["session_id"] == "sess-1"
    assert drop["tool_use_id"] == "toolu_abc"
    assert isinstance(drop["elapsed_ms"], int)
    assert isinstance(drop["ts"], str)


def test_non_2xx_response_dropped(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _Recorder(status=302))
    _run_cli(monkeypatch, _payload())
    assert [d["reason"] for d in _drops(env)] == ["http_302"]


@pytest.mark.parametrize("missing", ["error", "tool_use_id"])
def test_missing_fields_no_post(env: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    payload = _payload()
    del payload[missing]
    rc, out = _run_cli(monkeypatch, payload)
    assert rc == 0
    assert out == ""
    assert recorder.calls == []
    assert [d["reason"] for d in _drops(env)] == ["missing_fields"]


def test_mode_headless(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADLESS_BUILD_ENGINE", "claude-code")
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    _run_cli(monkeypatch, _payload())
    body = recorder.calls[0]["body"]
    assert body["mode"] == "headless"
    assert body["engine"] == "claude-code"


def test_mode_interactive(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    _run_cli(monkeypatch, _payload())
    body = recorder.calls[0]["body"]
    assert body["mode"] == "interactive"
    assert body["engine"] is None


def test_remote_url_without_key_no_post(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.test")
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    rc, out = _run_cli(monkeypatch, _payload())
    assert rc == 0
    assert out == ""
    assert recorder.calls == []
    assert [d["reason"] for d in _drops(env)] == ["no_url_key"]


def test_tool_input_shaping(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    _run_cli(monkeypatch, _payload(tool_input=["not", "a", "dict"]))
    _run_cli(
        monkeypatch,
        _payload(tool_input={"command": "x" * 5000, "flag": True, "ratio": 0.5}),
    )
    assert recorder.calls[0]["body"]["tool_input"] == {}
    shaped = recorder.calls[1]["body"]["tool_input"]
    assert shaped == {"command": "x" * 2000, "flag": True, "ratio": 0.5}


def test_error_and_duration_shaping(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    _run_cli(
        monkeypatch,
        _payload(error="e" * 5000, duration_ms=True, is_interrupt="yes", cwd=5),
    )
    body = recorder.calls[0]["body"]
    assert body["error"] == "e" * 4000
    assert body["duration_ms"] is None
    assert body["is_interrupt"] is False
    assert body["cwd"] is None


def test_hook_version_falls_back_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(name: str) -> str:
        raise RuntimeError(name)

    monkeypatch.setattr(events.importlib.metadata, "version", _boom)
    assert events._hook_version() is None


def test_drop_log_rotates_when_oversized(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("x" * 262145)
    payload = _payload()
    del payload["error"]
    _run_cli(monkeypatch, payload)
    assert [d["reason"] for d in _drops(env)] == ["missing_fields"]


def test_drop_log_failure_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(events, "get_event_drop_log_path", lambda: blocker / "sub" / "x.jsonl")
    events.post_failure({"session_id": "s"})  # must not raise


def test_drop_log_path_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.get_event_drop_log_path() == tmp_path / ".cache/personal_kb/event-drops.jsonl"


def test_post_tool_use_failure_is_supported() -> None:
    assert "PostToolUseFailure" in cli._SUPPORTED_EVENTS
