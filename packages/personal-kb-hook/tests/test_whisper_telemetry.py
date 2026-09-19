"""Tests for whisper-efficacy telemetry (GTD ccc05354).

Coverage map (stdlib-only — monkeypatch urllib.request.urlopen):

  (1) roster emit appends one jsonl row per MapKey in all_map_ids INCLUDING
      cross-project entries.
  (2) listener emit appends one jsonl row per cached winner with
      trigger_context containing excerpt_hash (sha256 of text) + operating
      + cwd_project.
  (3) PostToolUse consume marks the matching row consumed for BOTH a string
      entry_id and a list entry_id, and is a no-op (and zero stdout) for a
      non-kb_get tool_name.
  (4) A PostToolUse payload produces zero stdout.
  (5) Stop flush POSTs the rows to /api/kb/telemetry/whispers even when
      KB_LISTENER_ENABLED / PERSONAL_KB_LISTENER is unset/FALSE.
  (6) A 2-KB roster still POSTs all rows to exactly the personal endpoint
      once (v1 routing — no per-source_kb fan-out).
  (7) Orphan-sweep flushes + deletes a foreign-session log.
  (8) All paths silent-on-failure (no exception escapes; rc==0).

All assertions are stdlib-only. The hook package's ``dependencies = []``
invariant is preserved; the only imports are stdlib (json, hashlib, socket,
urllib, pathlib).
"""

from __future__ import annotations

import hashlib
import io
import json
import socket
import sys
import unittest.mock
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import cli, listener_worker, roster, telemetry
from personal_kb_hook.paths import get_whisper_log_path

# ─── shared fixtures / helpers ───────────────────────────────────────────────


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate the hook's filesystem touchpoints to ``tmp_path``.

    Pins ``HOME`` + ``XDG_CONFIG_HOME`` so no stray ``kbs.json`` leaks in
    and seeds the legacy env pair so telemetry POSTs have a target.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    # build_engine read defensively — unset => None.
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")
    cache_dir = tmp_path / ".cache" / "personal_kb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    return {"root": tmp_path, "cache_dir": cache_dir}


def _read_log(session_id: str) -> list[dict[str, Any]]:
    """Read all jsonl rows from a session's whisper-log file."""
    path = get_whisper_log_path(session_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


class _MockResponse:
    """Minimal context-manager HTTP response shim."""

    def __init__(self, body: bytes = b'{"upserted":0}', status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Recorder:
    """Records urlopen calls (URL, headers, body) and returns 200 by default."""

    def __init__(self, status: int = 200) -> None:
        self.calls: list[dict[str, Any]] = []
        self.status = status

    def __call__(self, req: Any, timeout: float = 30.0) -> _MockResponse:
        url = getattr(req, "full_url", None) or getattr(req, "_full_url", None)
        headers = dict(getattr(req, "headers", {}) or {})
        data = getattr(req, "data", b"") or b""
        body_decoded: Any
        try:
            body_decoded = json.loads(data.decode("utf-8")) if data else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            body_decoded = None
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "body": body_decoded,
                "method": getattr(req, "method", None),
            }
        )
        return _MockResponse(status=self.status)


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    args: list[str] | None = None,
) -> tuple[int, str]:
    """Run cli.main() with ``payload`` on stdin and capture stdout."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    stdout_buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout_buf)
    rc = 0
    try:
        cli.main(args or [])
    except SystemExit as exc:
        rc = int(exc.code or 0)
    return rc, stdout_buf.getvalue()


def _stub_http_index(
    monkeypatch: pytest.MonkeyPatch,
    projects: dict[str, list[tuple[str, dict[str, str]]]],
) -> None:
    """Replace ``http_index.load_index`` with a fixed dict-returning stub."""
    monkeypatch.setattr(cli.http_index, "load_index", lambda roster_arg: projects)


# ─── (1) roster emit (incl. cross-project) ──────────────────────────────────


def test_roster_emit_one_row_per_mapkey_including_cross_project(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Roster emit writes one jsonl row per MapKey (own + cross-project).

    Each row carries the map's ``pointers`` list — the kb-ids the map's
    body points at — so mark_consumed can chain-credit map → detail
    fetches (GTD 88441f9c).
    """
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])

    # Build a 2-project index: own (personal-kb) + cross (other-proj). The
    # index value shape is dict[project_ref, list[tuple[label, MapEntry]]].
    _stub_http_index(
        monkeypatch,
        {
            "personal-kb": [
                (
                    "personal",
                    {
                        "id": "kb-1",
                        "short_title": "auth",
                        "long_title": "A",
                        "pointers": ["kb-101", "kb-102"],
                    },
                ),
                (
                    "personal",
                    {
                        "id": "kb-2",
                        "short_title": "ingest",
                        "long_title": "I",
                        "pointers": [],
                    },
                ),
            ],
            "other-proj": [
                (
                    "team",
                    {
                        "id": "kb-9",
                        "short_title": "ops",
                        "long_title": "O",
                        "pointers": ["kb-999"],
                    },
                ),
            ],
        },
    )

    payload = {
        "hook_event_name": "SessionStart",
        "cwd": str(hook_env["root"]),
        "session_id": "sess-roster",
    }
    rc, _ = _run_cli(monkeypatch, payload, ["--format=text"])
    assert rc == 0

    rows = _read_log("sess-roster")
    # 2 own + 1 cross = 3 rows.
    assert len(rows) == 3

    by_id = {r["map_id"]: r for r in rows}
    assert set(by_id) == {"kb-1", "kb-2", "kb-9"}

    # Own-project rows: source_kb=personal, surface=roster, cwd_project=personal-kb.
    assert by_id["kb-1"]["source_kb"] == "personal"
    assert by_id["kb-1"]["surface"] == "roster"
    assert by_id["kb-1"]["cwd_project"] == "personal-kb"
    assert by_id["kb-1"]["consumed"] is False
    assert by_id["kb-1"]["consumed_ts"] is None
    assert by_id["kb-1"]["trigger_context"] == {
        "cwd_project": "personal-kb",
        "emit_reason": "first-emission",
    }
    assert by_id["kb-1"]["host"] == socket.gethostname()
    # build_engine unset => null.
    assert by_id["kb-1"]["build_engine"] is None
    # Pointers flow through from the index entry into the appended row.
    assert by_id["kb-1"]["pointers"] == ["kb-101", "kb-102"]
    assert by_id["kb-2"]["pointers"] == []

    # Cross-project row preserves its source_kb label AND its pointers.
    assert by_id["kb-9"]["source_kb"] == "team"
    assert by_id["kb-9"]["surface"] == "roster"
    assert by_id["kb-9"]["pointers"] == ["kb-999"]


def test_roster_emit_build_engine_read_defensively(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """HEADLESS_BUILD_ENGINE is read defensively per emit."""
    monkeypatch.setenv("HEADLESS_BUILD_ENGINE", "claude-code")
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    _stub_http_index(
        monkeypatch,
        {
            "personal-kb": [
                ("personal", {"id": "kb-1", "short_title": "auth", "long_title": "A"}),
            ]
        },
    )
    rc, _ = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "sess-bengine",
        },
        ["--format=text"],
    )
    assert rc == 0
    rows = _read_log("sess-bengine")
    assert rows[0]["build_engine"] == "claude-code"


def test_suppressed_second_run_appends_no_row_and_first_reason_is_first_emission(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A suppressed (non-emitting) re-run of the SAME scope+maps appends nothing.

    The first SessionStart emits (reason=first-emission) and writes one row.
    An immediate second SessionStart with the identical project/index is
    suppressed by should_emit()'s subset check and must not append another
    jsonl row at all -- the re-emission counter lives entirely server-side
    (emit_count), so a suppressed hook run must never even POST a row.
    """
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    _stub_http_index(
        monkeypatch,
        {"personal-kb": [("personal", {"id": "kb-1", "short_title": "auth", "long_title": "A"})]},
    )
    payload = {
        "hook_event_name": "SessionStart",
        "cwd": str(hook_env["root"]),
        "session_id": "sess-suppressed",
    }
    rc1, _ = _run_cli(monkeypatch, dict(payload), ["--format=text"])
    assert rc1 == 0
    rows = _read_log("sess-suppressed")
    assert len(rows) == 1
    assert rows[0]["trigger_context"]["emit_reason"] == "first-emission"

    # Second run: same scope, same maps -> should_emit() returns None.
    rc2, _ = _run_cli(monkeypatch, dict(payload), ["--format=text"])
    assert rc2 == 0
    rows_after = _read_log("sess-suppressed")
    assert rows_after == rows


def test_new_maps_emit_reason_recorded_on_re_emission(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A genuine second emission (new map key) records reason=new-maps.

    Post-delta-FYI (GTD 64f71a8a): the second emission is a DELTA — it
    telemeters ONLY the genuinely new map (kb-2), not kb-1, which was
    already surfaced and announced in the first-emission batch.
    """
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])

    _stub_http_index(
        monkeypatch,
        {"personal-kb": [("personal", {"id": "kb-1", "short_title": "auth", "long_title": "A"})]},
    )
    rc1, _ = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "sess-newmaps",
        },
        ["--format=text"],
    )
    assert rc1 == 0

    # Second call surfaces an additional map key -> not a subset -> re-emit.
    _stub_http_index(
        monkeypatch,
        {
            "personal-kb": [
                ("personal", {"id": "kb-1", "short_title": "auth", "long_title": "A"}),
                ("personal", {"id": "kb-2", "short_title": "ingest", "long_title": "I"}),
            ]
        },
    )
    rc2, out2 = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": "sess-newmaps",
        },
        ["--format=text"],
    )
    assert rc2 == 0
    # Delta FYI, not the full directory re-injected.
    assert out2 == "New map — personal/[kb-2] ingest"

    rows = _read_log("sess-newmaps")
    # First emission: 1 row (kb-1, first-emission). Second (delta): 1 row
    # (kb-2 only, new-maps) — kb-1 is NOT re-announced/re-telemetered.
    assert len(rows) == 2
    second_batch = rows[1:]
    assert {r["map_id"] for r in second_batch} == {"kb-2"}
    for row in second_batch:
        assert row["trigger_context"]["emit_reason"] == "new-maps"


# ─── (2) listener emit at pointer-CACHE time ────────────────────────────────


def test_listener_emit_one_row_per_winner_with_excerpt_hash(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path], tmp_path: Path
) -> None:
    """Listener emit writes one jsonl row per winner, trigger_context complete."""
    # Build a request_data with session_id + text + operating + cwd_project.
    text = "We need to know which machine runs traefik in the home lab."
    session_id = "sess-listener"
    request_data = {
        "session_id": session_id,
        "text": text,
        "cwd_project": "home-lab",
        "operating": ["mcp:personal-kb", "mcp:team-kb"],
        "source_label": "home-lab",
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(request_data), encoding="utf-8")
    cache_path = tmp_path / ".cache" / "personal_kb" / f"listener-{session_id}.json"

    # Stub roster.load_roster to return a single 'personal' KB so the worker
    # POSTs and arbitrates to a single winner.
    monkeypatch.setattr(
        listener_worker.roster,
        "load_roster",
        lambda: [roster.KbEntry(label="personal", url="https://kb.example.com", key="secret")],
    )

    # Stub urlopen to return a single non-null pointer.
    def fake_urlopen(req: Any, timeout: float = 30.0) -> _MockResponse:
        body = json.dumps(
            {
                "pointer": {
                    "id": "kb-77",
                    "short_title": "Home Lab",
                    "long_title": "Home lab network topology",
                }
            }
        ).encode("utf-8")
        return _MockResponse(body=body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "argv", ["listener_worker", str(req_path), str(cache_path)])

    listener_worker.main()

    rows = _read_log(session_id)
    assert len(rows) == 1
    row = rows[0]

    assert row["surface"] == "listener"
    assert row["map_id"] == "kb-77"
    assert row["source_kb"] == "personal"
    assert row["cwd_project"] == "home-lab"

    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert row["trigger_context"] == {
        "operating": ["mcp:personal-kb", "mcp:team-kb"],
        "cwd_project": "home-lab",
        "excerpt_hash": expected_hash,
    }
    assert row["consumed"] is False
    assert row["consumed_ts"] is None
    assert row["host"] == socket.gethostname()


# ─── (3) PostToolUse consume — str + list entry_id; non-kb_get no-op ────────


def test_post_tool_use_marks_consumed_for_string_entry_id(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Single string entry_id marks the matching row consumed=true."""
    session_id = "sess-consume-str"
    # Seed two pre-existing rows.
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-A",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-B",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )

    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": "mcp__personal-kb__kb_get",
            "tool_input": {"entry_id": "kb-A"},
        },
    )
    assert rc == 0
    assert out == ""

    rows = _read_log(session_id)
    by_id = {r["map_id"]: r for r in rows}
    assert by_id["kb-A"]["consumed"] is True
    assert by_id["kb-A"]["consumed_ts"] is not None
    assert by_id["kb-B"]["consumed"] is False


def test_post_tool_use_marks_consumed_for_list_entry_id(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """List entry_id marks every matching row consumed=true; non-matches unchanged."""
    session_id = "sess-consume-list"
    for mid in ("kb-1", "kb-2", "kb-3"):
        telemetry.append_row(
            session_id,
            {
                "session_id": session_id,
                "host": "h",
                "surface": "roster",
                "map_id": mid,
                "source_kb": "personal",
                "cwd_project": "p",
                "trigger_context": {},
                "emitted_ts": "2026-06-17T00:00:00+00:00",
                "consumed": False,
                "consumed_ts": None,
                "build_engine": None,
            },
        )

    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            # team_kb_get also counts — exercise both names indirectly.
            "tool_name": "mcp__team-kb__team_kb_get",
            "tool_input": {"entry_id": ["kb-1", "kb-3"]},
        },
    )
    assert rc == 0
    assert out == ""

    rows = _read_log(session_id)
    by_id = {r["map_id"]: r for r in rows}
    assert by_id["kb-1"]["consumed"] is True
    assert by_id["kb-2"]["consumed"] is False
    assert by_id["kb-3"]["consumed"] is True


def test_post_tool_use_marks_consumed_via_pointer(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A kb_get whose id is in a row's ``pointers`` chain-credits that row.

    The map row's ``map_id`` does NOT match the fetched entry_id, but the
    entry_id IS in the row's ``pointers`` list — so mark_consumed still
    marks the map row as consumed and records ``consumed_via='pointer'``
    inside ``trigger_context`` so the server can distinguish direct-vs-
    chain credit without a whisper_telemetry schema migration (GTD
    88441f9c).
    """
    session_id = "sess-consume-pointer"
    # Row 1: map kb-M with pointers [kb-D1, kb-D2].
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-M",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {"cwd_project": "p"},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
            "pointers": ["kb-D1", "kb-D2"],
        },
    )
    # Row 2: unrelated map kb-N (no pointers).
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-N",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {"cwd_project": "p"},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
            "pointers": [],
        },
    )

    # Fetch a DETAIL entry — kb-D1 — which is one of row 1's pointers.
    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": "mcp__personal-kb__kb_get",
            "tool_input": {"entry_id": "kb-D1"},
        },
    )
    assert rc == 0
    assert out == ""

    rows = _read_log(session_id)
    by_id = {r["map_id"]: r for r in rows}
    # Row 1 (kb-M) was credited via its pointer.
    assert by_id["kb-M"]["consumed"] is True
    assert by_id["kb-M"]["consumed_ts"] is not None
    assert by_id["kb-M"]["trigger_context"].get("consumed_via") == "pointer"
    # Row 2 (kb-N) untouched.
    assert by_id["kb-N"]["consumed"] is False
    assert "consumed_via" not in by_id["kb-N"].get("trigger_context", {})


def test_post_tool_use_direct_map_hit_records_consumed_via_map(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A direct map kb_get records ``consumed_via='map'`` in trigger_context.

    Direct map fetches are the pre-pointer ``mark_consumed`` behavior and
    must keep working; the new signal is that server-side observers can
    now distinguish "direct" from "chain" credit by inspecting
    ``trigger_context.consumed_via`` (GTD 88441f9c).
    """
    session_id = "sess-consume-map"
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-M",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {"cwd_project": "p"},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
            "pointers": ["kb-D1", "kb-D2"],
        },
    )
    rc, _ = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": "mcp__personal-kb__kb_get",
            "tool_input": {"entry_id": "kb-M"},
        },
    )
    assert rc == 0
    rows = _read_log(session_id)
    assert rows[0]["consumed"] is True
    assert rows[0]["trigger_context"].get("consumed_via") == "map"


def test_mark_consumed_stays_in_memory_no_network_calls(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """mark_consumed must never call http_index.load_index or urlopen.

    Fires on EVERY kb_get / team_kb_get — must be ultra-cheap (no
    http_index/load_index/network calls). Verified by asserting urlopen
    is NEVER invoked during the PostToolUse consume path.
    """
    session_id = "sess-noop-network"
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-M",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
            "pointers": ["kb-D1"],
        },
    )
    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    rc, _ = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": "mcp__personal-kb__kb_get",
            "tool_input": {"entry_id": "kb-D1"},
        },
    )
    assert rc == 0
    # No HTTP calls during PostToolUse consume.
    assert recorder.calls == []


def test_post_tool_use_non_kb_get_is_noop(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A non-kb_get tool_name is a no-op (no consume, no stdout)."""
    session_id = "sess-noop"
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-A",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )
    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": "mcp__personal-kb__kb_search",  # NOT kb_get
            "tool_input": {"entry_id": "kb-A"},
        },
    )
    assert rc == 0
    assert out == ""
    rows = _read_log(session_id)
    assert rows[0]["consumed"] is False  # untouched


# ─── (4) PostToolUse payload produces zero stdout ───────────────────────────


def test_post_tool_use_payload_produces_zero_stdout(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A PostToolUse payload — kb_get or otherwise — never writes to stdout."""
    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": "s-x",
            "tool_name": "mcp__personal-kb__kb_get",
            "tool_input": {"entry_id": "kb-1"},
        },
    )
    assert rc == 0
    assert out == ""


# ─── (5) Stop flush runs OUTSIDE the listener-enabled guard ─────────────────


def test_stop_flush_posts_even_when_listener_disabled(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Stop flushes the jsonl even with PERSONAL_KB_LISTENER unset/FALSE."""
    # Make sure the listener gate is OFF.
    monkeypatch.delenv("PERSONAL_KB_LISTENER", raising=False)

    session_id = "sess-flush-off"
    # Seed two roster rows.
    for mid in ("kb-1", "kb-2"):
        telemetry.append_row(
            session_id,
            {
                "session_id": session_id,
                "host": "h",
                "surface": "roster",
                "map_id": mid,
                "source_kb": "personal",
                "cwd_project": "p",
                "trigger_context": {},
                "emitted_ts": "2026-06-17T00:00:00+00:00",
                "consumed": False,
                "consumed_ts": None,
                "build_engine": None,
            },
        )

    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)

    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "session_id": session_id,
            "transcript_path": "",  # not required for the flush path
        },
    )
    assert rc == 0
    assert out == ""

    # Exactly one POST to the personal endpoint with both rows.
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["url"] == "https://kb.example.com/api/kb/telemetry/whispers"
    assert call["headers"].get("Authorization") == "Bearer secret"
    assert call["method"] == "POST"
    body = call["body"]
    assert isinstance(body, dict)
    assert "rows" in body
    assert len(body["rows"]) == 2
    sent_ids = {r["map_id"] for r in body["rows"]}
    assert sent_ids == {"kb-1", "kb-2"}

    # File LEFT IN PLACE on a 2xx (composite-PK upsert makes it idempotent).
    assert get_whisper_log_path(session_id).exists()


# ─── (6) 2-KB roster posts to exactly the personal endpoint ONCE ────────────


def test_two_kb_roster_posts_once_to_personal_endpoint(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """v1 routing: all rows POST to PERSONAL_KB_URL once, regardless of source_kb."""
    session_id = "sess-2kb"
    # Two rows, one per source_kb — but Stop still POSTs to the personal endpoint.
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-1",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-9",
            "source_kb": "team",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )

    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)

    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "session_id": session_id,
            "transcript_path": "",
        },
    )
    assert rc == 0
    assert out == ""

    # Exactly ONE POST to the personal endpoint with both rows.
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["url"] == "https://kb.example.com/api/kb/telemetry/whispers"
    body = recorder.calls[0]["body"]
    assert isinstance(body, dict)
    assert len(body["rows"]) == 2


# ─── (7) Orphan sweep flushes + deletes a foreign-session log ───────────────


def test_orphan_sweep_flushes_and_deletes_foreign_session_log(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """SessionStart orphan sweep POSTs + deletes any prior-session log."""
    # Seed a foreign-session log.
    foreign = "sess-foreign"
    telemetry.append_row(
        foreign,
        {
            "session_id": foreign,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-X",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )
    foreign_path = get_whisper_log_path(foreign)
    assert foreign_path.exists()

    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)

    # Minimal SessionStart that returns early after the orphan sweep — no
    # .kb_project anywhere on the walk, so the directory pipeline is silent.
    monkeypatch.chdir(hook_env["root"])
    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "sess-current",
        },
        ["--format=text"],
    )
    assert rc == 0
    # No directory output (no .kb_project), but the sweep still ran.
    assert out == ""

    # Sweep POSTed the foreign-session rows.
    sweep_calls = [c for c in recorder.calls if "/telemetry/whispers" in (c["url"] or "")]
    assert len(sweep_calls) == 1
    assert sweep_calls[0]["body"]["rows"][0]["map_id"] == "kb-X"

    # Foreign log deleted on the 2xx.
    assert not foreign_path.exists()


def test_orphan_sweep_does_not_delete_current_session_log(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """The sweep skips the current session's log even if it exists."""
    current = "sess-now"
    telemetry.append_row(
        current,
        {
            "session_id": current,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-K",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )
    current_path = get_whisper_log_path(current)
    assert current_path.exists()

    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)

    monkeypatch.chdir(hook_env["root"])
    rc, _ = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": current,
        },
        ["--format=text"],
    )
    assert rc == 0
    # No POSTs to the telemetry endpoint — the only log is the current one.
    sweep_calls = [c for c in recorder.calls if "/telemetry/whispers" in (c["url"] or "")]
    assert sweep_calls == []
    # Current log still on disk.
    assert current_path.exists()


# ─── (8) silent-on-failure ──────────────────────────────────────────────────


def test_flush_silent_when_personal_kb_url_unset(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Stop flush is silent when PERSONAL_KB_URL is unset (no POST attempted)."""
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)

    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "session_id": "sX",
            "transcript_path": "",
        },
    )
    assert rc == 0
    assert out == ""
    assert recorder.calls == []


def test_flush_silent_when_post_raises(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Stop flush swallows an HTTPError so it never propagates."""
    session_id = "sess-fail"
    telemetry.append_row(
        session_id,
        {
            "session_id": session_id,
            "host": "h",
            "surface": "roster",
            "map_id": "kb-1",
            "source_kb": "personal",
            "cwd_project": "p",
            "trigger_context": {},
            "emitted_ts": "2026-06-17T00:00:00+00:00",
            "consumed": False,
            "consumed_ts": None,
            "build_engine": None,
        },
    )

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.HTTPError(
            "https://kb.example.com/api/kb/telemetry/whispers",
            500,
            "Internal Server Error",
            unittest.mock.MagicMock(),
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "session_id": session_id,
            "transcript_path": "",
        },
    )
    assert rc == 0
    assert out == ""
    # The file is left in place on failure (re-flush is idempotent).
    assert get_whisper_log_path(session_id).exists()


def test_consume_silent_when_log_missing(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """PostToolUse against a session with no log file is a silent no-op."""
    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "session_id": "no-such-session",
            "tool_name": "mcp__personal-kb__kb_get",
            "tool_input": {"entry_id": "kb-1"},
        },
    )
    assert rc == 0
    assert out == ""


def test_append_row_silent_when_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """append_row swallows OSError so an emit failure never aborts a session."""

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "personal_kb_hook.telemetry.get_whisper_log_path",
        lambda _sid: Path("/nonexistent/whisper-log-x.jsonl"),
    )
    # Force an OSError inside the open() call.
    import builtins

    real_open = builtins.open

    def fake_open(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], (str, Path)):
            p = str(args[0])
            if "nonexistent" in p:
                explode()
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    # Should not raise.
    telemetry.append_row("any", {"session_id": "any", "map_id": "kb-1"})
