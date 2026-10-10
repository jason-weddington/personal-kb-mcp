"""Detached turn-digest sender."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import turn_sender
from personal_kb_hook.paths import get_event_drop_log_path, get_turn_digest_log_path

if TYPE_CHECKING:
    from pathlib import Path

SID = "s1"
SEND_KEYS = {
    "ts", "op", "session_id", "event_id", "http_status", "server_reason", "redactions",
    "drop_reason", "elapsed_ms",
}  # fmt: skip


class _Resp:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.status = status
        self._b = body

    def read(self) -> bytes:
        return self._b

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


@pytest.fixture(autouse=True)
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.test")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()


@pytest.fixture
def body(tmp_path: Path) -> Path:
    p = tmp_path / "kb-turn-x.json"
    p.write_bytes(json.dumps({"event_id": "s1:0"}).encode())
    return p


def fake(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[tuple[Any, float]]:
    seen: list[tuple[Any, float]] = []

    def urlopen(req: Any, timeout: float = 0) -> Any:
        seen.append((req, timeout))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def drops() -> list[dict[str, Any]]:
    p = get_event_drop_log_path()
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def rows() -> list[dict[str, Any]]:
    p = get_turn_digest_log_path(SID)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_recorded(body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = body.read_bytes()
    seen = fake(monkeypatch, _Resp(b'{"recorded": true, "reason": "recorded", "redactions": []}'))
    assert turn_sender.send_file(str(body), SID) is None
    req, timeout = seen[0]
    assert req.full_url == "https://kb.example.test/api/kb/turn"
    assert req.get_method() == "POST"
    assert req.data == data
    assert req.get_header("Authorization") == "Bearer test-key"
    assert req.get_header("Content-type") == "application/json"
    assert timeout == 10.0
    assert not body.exists()
    assert drops() == []
    (row,) = rows()
    assert set(row) == SEND_KEYS
    assert (row["op"], row["http_status"], row["server_reason"]) == ("send", 200, "recorded")
    assert (row["redactions"], row["drop_reason"], row["event_id"]) == (0, None, "s1:0")


@pytest.mark.parametrize("reason", ["write-failed", "redaction-unavailable", "duplicate-mismatch"])
def test_rejected(reason: str, body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, _Resp(json.dumps({"reason": reason}).encode()))
    assert turn_sender.send_file(str(body), SID) == f"rejected_{reason}"
    assert [d["op"] for d in drops()] == ["turn_digest"]
    assert rows()[0]["drop_reason"] == f"rejected_{reason}"


@pytest.mark.parametrize("raw", [b'{"reason": "capture-off"}', b'{"reason": "duplicate"}', b"nope"])
def test_benign(raw: bytes, body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, _Resp(raw))
    assert turn_sender.send_file(str(body), SID) is None
    assert drops() == []
    if b"capture-off" in raw:
        assert rows()[0]["server_reason"] == "capture-off"


def test_non_2xx_status(body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, _Resp(b"", 503))
    assert turn_sender.send_file(str(body), SID) == "http_503"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (urllib.error.HTTPError("u", 413, "m", None, io.BytesIO(b"")), "http_413"),
        (urllib.error.HTTPError("u", 422, "m", None, io.BytesIO(b"")), "http_422"),
        (urllib.error.URLError(TimeoutError()), "timeout"),
        (urllib.error.URLError("x"), "urlerror"),
    ],
)
def test_errors(exc: Exception, expected: str, body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, exc)
    assert turn_sender.send_file(str(body), SID) == expected
    assert rows()[0]["drop_reason"] == expected
    if expected == "http_413":
        assert rows()[0]["http_status"] == 413
    assert not body.exists()


def test_body_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = fake(monkeypatch, _Resp(b"{}"))
    assert turn_sender.send_file(str(tmp_path / "gone.json"), SID) == "body_unreadable"
    assert seen == []
    assert rows()[0]["event_id"] is None


def test_no_url_key(body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAL_KB_API_KEY")
    seen = fake(monkeypatch, _Resp(b"{}"))
    assert turn_sender.send_file(str(body), SID) == "no_url_key"
    assert seen == []
    assert not body.exists()


def test_main_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["turn_sender"])
    turn_sender.main()


def test_main_sends(body: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, _Resp(b"{}"))
    monkeypatch.setattr(sys, "argv", ["turn_sender", str(body), SID])
    turn_sender.main()
    assert not body.exists()


def test_runs_with_dash_m() -> None:
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "personal_kb_hook.turn_sender"], timeout=30, check=False
    )
    assert r.returncode == 0
