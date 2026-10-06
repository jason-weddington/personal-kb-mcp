"""Tests for the personal-kb-hook CLI entry point."""

import io
import json
import os
import subprocess
import unittest.mock
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import cli, listener
from personal_kb_hook.paths import get_listener_cache_path
from personal_kb_hook.render import BANNED_TOKENS, render_directory


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate the hook's filesystem touchpoints to ``tmp_path``.

    Pins ``HOME`` and ``XDG_CONFIG_HOME`` under ``tmp_path`` so
    :func:`personal_kb_hook.roster.load_roster` finds no ``kbs.json`` and
    falls back to its single 'personal' env-var entry — the legacy
    byte-identical fallback path P1 must preserve.
    """
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    # Unset means the local default (127.0.0.1:8765); point at a closed port so a
    # real daemon on the build machine can never leak into these tests.
    monkeypatch.setenv("PERSONAL_KB_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("HOME", str(tmp_path))
    # Pin XDG_CONFIG_HOME explicitly: any stray ~/.config/personal_kb/kbs.json
    # on the build machine MUST NOT leak into these tests.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    cache_root.mkdir()
    # The hook expands ~/.cache/... — make HOME point at tmp_path.
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True, exist_ok=True)
    return {
        "root": tmp_path,
        "db_path": db_path,
        # role unset -> "default" -> the CLI globs maps_index.*.jsonl
        "maps_index": db_path.parent / "maps_index.default.jsonl",
    }


def _write_index(path: Path, project_ref: str, maps: list[dict[str, str]]) -> None:
    """Write a single project record to the JSONL index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"project_ref": project_ref, "maps": maps}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _make_fake_urlopen(projects: list[dict[str, Any]]) -> Any:
    """Return a fake urlopen that serves ``projects`` as the HTTP index response."""

    def fake_urlopen(req: object, timeout: float = 3.0) -> object:
        body = json.dumps({"projects": projects}).encode("utf-8")
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
        return mock_resp

    return fake_urlopen


def _run(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    args: list[str] | None = None,
) -> tuple[int, str]:
    """Run cli.main() with ``payload`` on stdin and capture stdout."""
    if isinstance(payload, str):
        stdin_text = payload
    elif payload is None:
        stdin_text = ""
    else:
        stdin_text = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
    stdout_buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout_buf)
    rc = 0
    try:
        cli.main(args or [])
    except SystemExit as exc:
        rc = int(exc.code or 0)
    return rc, stdout_buf.getvalue()


# ---------------------------------------------------------------------------
# Stdin tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["", "   \n\n", "not-json", "[1, 2, 3]", '"a string"', "null", "42"],
)
def test_tolerant_stdin(
    monkeypatch: pytest.MonkeyPatch, raw: str, hook_env: dict[str, Path]
) -> None:
    """Any non-object stdin → exit 0 with no stdout."""
    rc, out = _run(monkeypatch, raw)
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# Event branching
# ---------------------------------------------------------------------------


def test_unhandled_event_silent(monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]) -> None:
    """An event we do not handle → exit 0 silent.

    PostToolUse is now a handled event (whisper-telemetry consume path); use
    a still-unhandled event name (``PreToolUse``) to keep this assertion
    valid.
    """
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"}],
    )
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "PreToolUse",
            "cwd": str(hook_env["root"]),
            "session_id": "s1",
        },
    )
    assert rc == 0
    assert out == ""


def test_missing_event_silent(monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]) -> None:
    """Missing hook_event_name → silent."""
    rc, out = _run(monkeypatch, {"cwd": str(hook_env["root"]), "session_id": "s1"})
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# Format=text vs claude-json
# ---------------------------------------------------------------------------


def _surface_fixture(hook_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Set up HTTP mock and .kb_project for a SessionStart test."""
    maps = [
        {"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"},
        {"id": "kb-2", "short_title": "ingest", "long_title": "Ingestion flow"},
    ]
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _make_fake_urlopen([{"project_ref": "personal-kb", "maps": maps}]),
    )
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    return {
        "hook_event_name": "SessionStart",
        "cwd": str(hook_env["root"]),
        "session_id": "session-format",
    }


def test_format_text_emits_directory_string(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    payload = _surface_fixture(hook_env, monkeypatch)
    rc, out = _run(monkeypatch, payload, ["--format=text"])
    assert rc == 0
    assert out.startswith("Maps for personal-kb — ")
    assert "[kb-1] auth: Authentication map" in out
    assert "; [kb-2] ingest: Ingestion flow" in out


def test_format_claude_json_envelope(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    # Use a different session_id so suppression scratch does not leak between tests
    payload = _surface_fixture(hook_env, monkeypatch)
    payload["session_id"] = "session-claude-json"
    rc, out = _run(monkeypatch, payload, ["--format=claude-json"])
    assert rc == 0
    obj = json.loads(out)
    assert obj["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    directory = obj["hookSpecificOutput"]["additionalContext"]
    assert directory.startswith("Maps for personal-kb — ")
    assert "[kb-1] auth: Authentication map" in directory


def test_suppressed_re_emit_text_empty(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    payload = _surface_fixture(hook_env, monkeypatch)
    payload["session_id"] = "session-suppress-text"
    rc1, out1 = _run(monkeypatch, payload, ["--format=text"])
    assert rc1 == 0
    assert out1
    # Second invocation, same session, same scope, same ids → suppressed.
    rc2, out2 = _run(monkeypatch, payload, ["--format=text"])
    assert rc2 == 0
    assert out2 == ""


def test_suppressed_re_emit_claude_json_empty(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    payload = _surface_fixture(hook_env, monkeypatch)
    payload["session_id"] = "session-suppress-json"
    rc1, out1 = _run(monkeypatch, payload, ["--format=claude-json"])
    assert rc1 == 0
    assert out1
    rc2, out2 = _run(monkeypatch, payload, ["--format=claude-json"])
    assert rc2 == 0
    assert out2 == ""


# ---------------------------------------------------------------------------
# Resolver / index miss
# ---------------------------------------------------------------------------


def test_no_kb_project_silent(monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]) -> None:
    """No .kb_project anywhere on the walk → silent."""
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"}],
    )
    nested = hook_env["root"] / "nest" / "deeper"
    nested.mkdir(parents=True)
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(nested),
            "session_id": "s2",
        },
    )
    assert rc == 0
    assert out == ""


def test_resolved_project_not_in_index_silent(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Resolved project_ref absent from the index → silent."""
    _write_index(hook_env["maps_index"], "some-other", [])
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s3",
        },
    )
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# Not-imperative / banned-tokens guard
# ---------------------------------------------------------------------------


def test_rendered_directory_not_imperative() -> None:
    """The rendered directory contains no banned imperative tokens."""
    directory = render_directory(
        "personal-kb",
        [
            {"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"},
            {"id": "kb-2", "short_title": "ingest", "long_title": "Ingestion flow"},
        ],
    )
    lowered = directory.lower()
    for tok in BANNED_TOKENS:
        assert tok not in lowered, f"banned imperative token {tok!r} found in: {directory}"


def test_render_long_title_empty_omits_colon() -> None:
    """An entry whose long_title is empty renders as ``[id] short_title`` only."""
    directory = render_directory(
        "p",
        [
            {"id": "kb-1", "short_title": "alpha", "long_title": ""},
            {"id": "kb-2", "short_title": "beta", "long_title": "second"},
        ],
    )
    assert directory == "Maps for p — [kb-1] alpha; [kb-2] beta: second"


def test_render_uses_em_dash_and_semicolon_join() -> None:
    """Exact shape: ``Maps for <ref> — [id] short: long; [id] short: long``."""
    directory = render_directory(
        "demo",
        [
            {"id": "kb-1", "short_title": "alpha", "long_title": "first"},
            {"id": "kb-2", "short_title": "beta", "long_title": "second"},
        ],
    )
    # U+2014 EM DASH explicitly:
    assert " — " in directory
    assert directory == "Maps for demo — [kb-1] alpha: first; [kb-2] beta: second"


def test_hook_package_is_stdlib_only_and_does_not_import_main_package() -> None:
    """The hook package must not import ``personal_kb`` or any third-party module.

    The split rule (AC1 of the standalone-package task): the standalone
    ``personal-kb-hook`` distribution is RUNTIME-stdlib-only. It must NOT
    depend on, or import from, the main ``personal_kb`` server package, and
    must not import any third-party top-level module.
    """
    import sys

    src_dir = Path(__file__).resolve().parent.parent / "src" / "personal_kb_hook"
    assert src_dir.is_dir(), src_dir
    py_files = sorted(src_dir.glob("*.py"))
    assert py_files, "no python files found in standalone hook package"

    # Whitelist of acceptable import roots: stdlib + the package's own modules.
    stdlib_roots = set(sys.stdlib_module_names)
    own_roots = {"personal_kb_hook"}

    import ast

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root in stdlib_roots or root in own_roots, (
                        f"{py_file.name}: forbidden import {alias.name!r} "
                        f"(must be stdlib or personal_kb_hook only)"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import — fine
                    continue
                module = node.module or ""
                root = module.split(".", 1)[0]
                assert root in stdlib_roots or root in own_roots, (
                    f"{py_file.name}: forbidden from-import {module!r} "
                    f"(must be stdlib or personal_kb_hook only)"
                )


# ---------------------------------------------------------------------------
# Internal exception safety
# ---------------------------------------------------------------------------


def test_main_swallows_internal_exception(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Forcing an internal raise still results in exit 0, no stdout."""

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("personal_kb_hook.http_index.load_index", boom)
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-boom",
        },
    )
    assert rc == 0
    assert out == ""


def test_module_main_callable() -> None:
    """``personal-kb-hook`` resolves to ``main`` and is importable."""
    assert callable(cli.main)
    # Smoke test: invoking with --help raises SystemExit but should not break.
    # We rely on the wrapper in main() to swallow SystemExit from --help.
    assert os.environ is not None


# ---------------------------------------------------------------------------
# End-to-end: HTTP env set but urlopen fails -> local directory emitted
# ---------------------------------------------------------------------------


def test_http_env_set_urlopen_fails_returns_empty(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """PERSONAL_KB_URL/API_KEY set but urlopen raises -> empty output (no local fallback)."""
    import urllib.error as _urllib_error

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")

    def raise_conn(*args: object, **kwargs: object) -> None:
        raise _urllib_error.URLError("simulated connection failure")

    monkeypatch.setattr(urllib.request, "urlopen", raise_conn)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-http-fail",
        },
    )
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# Stop-event: env gate
# ---------------------------------------------------------------------------


def _listener_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set all three listener env vars to active values."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "true")


def _make_transcript(path: Path, text: str = "A" * 300) -> None:
    """Write a minimal Claude Code transcript JSONL that extract_manifest can parse."""
    record = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


class _ProcHandle:
    """Mock Popen return value that fails if wait/communicate are called."""

    def wait(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("wait() must NOT be called on the Stop path")

    def communicate(self, *args: object, **kwargs: object) -> tuple[bytes, bytes]:
        raise AssertionError("communicate() must NOT be called on the Stop path")


@pytest.mark.parametrize(
    "missing_var",
    ["PERSONAL_KB_API_KEY", "PERSONAL_KB_LISTENER"],
)
def test_stop_env_gate_off_per_var_is_noop(
    missing_var: str,
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Stop event is a complete no-op when any of the three listener vars is absent."""
    _listener_env(monkeypatch)
    monkeypatch.delenv(missing_var, raising=False)

    popen_calls: list[Any] = []

    def mock_popen(*args: object, **kwargs: object) -> _ProcHandle:
        popen_calls.append((args, kwargs))
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(
        "personal_kb_hook.http_index.load_index",
        lambda: (_ for _ in ()).throw(AssertionError("load_index must not be called on Stop")),
    )

    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-gate-off",
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""
    assert popen_calls == []


def test_stop_gated_spawns_popen_no_wait(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Gated Stop event spawns Popen with correct argv and never calls wait."""
    _listener_env(monkeypatch)

    popen_calls: list[tuple[Any, Any]] = []

    def mock_popen(*args: object, **kwargs: object) -> _ProcHandle:
        popen_calls.append((args, kwargs))
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-gated-1",
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""
    assert len(popen_calls) == 1
    call_args = popen_calls[0][0][0]  # first positional arg (cmd list)
    assert call_args[1] == "-m"
    assert call_args[2] == "personal_kb_hook.listener_worker"
    # argv[3] = request_tmp_path, argv[4] = cache_path
    req_tmp_path = call_args[3]
    cache_path = call_args[4]
    assert req_tmp_path.endswith(".json")
    assert "listener-stop-gated-1.json" in cache_path


def test_stop_gated_at_most_one_popen_for_multi_kb_roster(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A Stop event whose load_roster() returns N>1 entries still spawns at most ONE Popen.

    AC-1 invariant: cli.py's Stop block spawns EXACTLY ONE worker via
    listener.spawn_worker — the worker (NOT the hook) loops the roster.
    There is NO code path that spawns one Popen per KB.
    """
    _listener_env(monkeypatch)

    popen_calls: list[tuple[Any, Any]] = []

    def mock_popen(*args: object, **kwargs: object) -> _ProcHandle:
        popen_calls.append((args, kwargs))
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    # Wire a 3-KB roster via kbs.json + key_file indirection.
    config_dir = hook_env["root"] / ".config" / "personal_kb"
    config_dir.mkdir(parents=True)
    for label in ("personal", "team", "ops"):
        kf = hook_env["root"] / f"{label}.key"
        kf.write_text(f"{label}-secret\n", encoding="utf-8")
    (config_dir / "kbs.json").write_text(
        json.dumps(
            [
                {
                    "label": "personal",
                    "url": "https://personal.kb/",
                    "key_file": str(hook_env["root"] / "personal.key"),
                },
                {
                    "label": "team",
                    "url": "https://team.kb/",
                    "key_file": str(hook_env["root"] / "team.key"),
                },
                {
                    "label": "ops",
                    "url": "https://ops.kb/",
                    "key_file": str(hook_env["root"] / "ops.key"),
                },
            ]
        ),
        encoding="utf-8",
    )

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-multi-kb-one-popen",
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""
    # AC-1: AT MOST ONE Popen regardless of roster cardinality.
    assert len(popen_calls) <= 1
    # In this fixture the Stop preconditions all pass so it's exactly one.
    assert len(popen_calls) == 1


def test_stop_request_tmp_body_contains_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """The request tmp file written before Popen has text/project_ref/operated fields."""
    _listener_env(monkeypatch)

    popen_calls: list[tuple[Any, Any]] = []

    def mock_popen(*args: object, **kwargs: object) -> _ProcHandle:
        popen_calls.append((args, kwargs))
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    transcript = hook_env["root"] / "transcript.jsonl"
    record = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "B" * 300},
                {"type": "tool_use", "name": "mcp__personal-kb__kb_search"},
            ]
        },
    }
    transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-body",
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""
    assert len(popen_calls) == 1

    req_tmp_path = popen_calls[0][0][0][3]
    with open(req_tmp_path, encoding="utf-8") as fh:
        body = json.load(fh)

    assert body["text"] == "B" * 300
    assert body["cwd_project"] == "personal-kb"
    assert "mcp:personal-kb" in body["operating"]

    # Cleanup tmp file
    os.unlink(req_tmp_path)


def test_stop_project_ref_null_when_no_kb_project(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """project_ref=null in request body when resolve_project returns None."""
    _listener_env(monkeypatch)

    popen_calls: list[tuple[Any, Any]] = []

    def mock_popen(*args: object, **kwargs: object) -> _ProcHandle:
        popen_calls.append((args, kwargs))
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    # No .kb_project file -> resolve_project returns None

    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    _rc, _out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-null-proj",
            "transcript_path": str(transcript),
        },
    )
    assert _rc == 0
    assert len(popen_calls) == 1

    req_tmp_path = popen_calls[0][0][0][3]
    with open(req_tmp_path, encoding="utf-8") as fh:
        body = json.load(fh)
    assert body["cwd_project"] is None

    os.unlink(req_tmp_path)


def test_stop_never_calls_load_index(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Stop event never calls http_index.load_index, even when gated and valid."""
    _listener_env(monkeypatch)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_index MUST NOT be called on a Stop event")

    monkeypatch.setattr("personal_kb_hook.http_index.load_index", boom)

    # Also mock Popen so no real subprocess is spawned
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _ProcHandle())

    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-no-li",
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""


def test_stop_missing_transcript_path_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Stop with missing/non-str transcript_path is a no-op (no Popen)."""
    _listener_env(monkeypatch)
    popen_calls: list[Any] = []

    def _record_popen(*a: Any, **kw: Any) -> _ProcHandle:
        popen_calls.append(a)
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", _record_popen)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-no-tp",
            # transcript_path intentionally omitted
        },
    )
    assert rc == 0
    assert out == ""
    assert popen_calls == []


def test_stop_empty_transcript_path_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Stop with empty-string transcript_path is a no-op."""
    _listener_env(monkeypatch)
    popen_calls: list[Any] = []

    def _record_popen(*a: Any, **kw: Any) -> _ProcHandle:
        popen_calls.append(a)
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", _record_popen)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "stop-empty-tp",
            "transcript_path": "",
        },
    )
    assert rc == 0
    assert out == ""
    assert popen_calls == []


def test_stop_missing_session_id_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Stop with missing session_id is a no-op (no Popen)."""
    _listener_env(monkeypatch)
    popen_calls: list[Any] = []

    def _record_popen(*a: Any, **kw: Any) -> _ProcHandle:
        popen_calls.append(a)
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", _record_popen)

    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            # session_id intentionally omitted
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""
    assert popen_calls == []


def test_stop_empty_session_id_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Stop with empty-string session_id is a no-op."""
    _listener_env(monkeypatch)
    popen_calls: list[Any] = []

    def _record_popen(*a: Any, **kw: Any) -> _ProcHandle:
        popen_calls.append(a)
        return _ProcHandle()

    monkeypatch.setattr(subprocess, "Popen", _record_popen)

    transcript = hook_env["root"] / "transcript.jsonl"
    _make_transcript(transcript)

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "Stop",
            "cwd": str(hook_env["root"]),
            "session_id": "",
            "transcript_path": str(transcript),
        },
    )
    assert rc == 0
    assert out == ""
    assert popen_calls == []


# ---------------------------------------------------------------------------
# UserPromptSubmit: whisper injection
# ---------------------------------------------------------------------------


def _write_listener_cache(path: Path, pending: Any, whispered_ids: list[Any]) -> None:
    """Write a listener cache file for test setup.

    ``pending`` may be either the LEGACY pre-P2 shape (a single dict or
    ``None``) or the P2 shape (a list of per-KB pointer dicts). The
    cli.py whisper block back-parses both. ``whispered_ids`` may be a list
    of bare-id strings (legacy) or ``[label, id]`` lists (P2).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pending": pending, "whispered_map_ids": whispered_ids}),
        encoding="utf-8",
    )


def _pending_map(
    entry_id: str = "kb-00099",
    short_title: str = "Authflow",
    long_title: str = "Authentication flow details",
    label: str = "personal",
) -> dict[str, str]:
    """Build one P2-shape per-KB pointer dict ({label, id, short_title, long_title})."""
    return {
        "label": label,
        "id": entry_id,
        "short_title": short_title,
        "long_title": long_title,
    }


def test_whisper_emitted_when_resolve_project_none(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Pending whisper is emitted even when resolve_project returns None."""
    _listener_env(monkeypatch)
    session_id = "whisper-no-proj"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(cache_path, _pending_map(), [])

    # No .kb_project -> resolve_project returns None
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    assert "Possibly relevant map" in out
    assert "kb-00099" in out


def test_whisper_emitted_when_empty_index(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Pending whisper is emitted even when the maps index has no maps."""
    _listener_env(monkeypatch)
    session_id = "whisper-empty-idx"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(cache_path, _pending_map(), [])

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    # Index has no maps for "personal-kb"
    _write_index(hook_env["maps_index"], "some-other-project", [])

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    assert "Possibly relevant map" in out
    assert "kb-00099" in out


def test_whisper_emitted_when_should_emit_false(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Pending whisper is emitted even when should_emit returns False (suppressed)."""
    _listener_env(monkeypatch)
    session_id = "whisper-suppressed"
    cache_path = get_listener_cache_path(session_id)

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-00001", "short_title": "auth", "long_title": "Auth flow"}],
    )
    # First invocation — no pending whisper, emits directory and marks suppression scratch
    _write_listener_cache(cache_path, None, [])
    _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    # Now write a new pending whisper to the cache (simulate worker writing it)
    _write_listener_cache(cache_path, _pending_map("kb-00099"), [])

    # Second invocation — suppression blocks the directory but whisper still emits
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    assert "Possibly relevant map" in out
    assert "kb-00099" in out


def test_whisper_injection_clears_pending_and_appends_id(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """After a whisper is emitted, pending=[] and whispered (label,id) is appended."""
    _listener_env(monkeypatch)
    session_id = "whisper-clear"
    cache_path = get_listener_cache_path(session_id)
    # Legacy seed: pending as a single dict, whispered_map_ids as bare-id strings.
    _write_listener_cache(cache_path, _pending_map("kb-00099"), ["kb-00001"])

    # Need a project + index so should_emit passes for a clean first emission,
    # but for this test we don't need the directory — we use no .kb_project
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    assert "Possibly relevant map" in out

    # Cache must be updated: pending cleared (P2 schema: empty list),
    # whispered (label,id) appended, legacy bare-id back-parsed.
    updated = json.loads(cache_path.read_text(encoding="utf-8"))
    assert updated["pending"] == []
    assert ["personal", "kb-00099"] in updated["whispered_map_ids"]
    # Pre-existing legacy bare-id back-parsed to ['personal', 'kb-00001'].
    assert ["personal", "kb-00001"] in updated["whispered_map_ids"]


def test_same_id_never_whispered_twice(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """An id already in whispered_map_ids is not whispered again."""
    _listener_env(monkeypatch)
    session_id = "whisper-dedup"
    cache_path = get_listener_cache_path(session_id)
    # pending=kb-00099 but it's already in whispered_map_ids
    _write_listener_cache(cache_path, _pending_map("kb-00099"), ["kb-00099"])

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    # whisper must NOT be emitted
    assert "Possibly relevant map" not in out


def test_directory_and_whisper_composed_text(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Directory + whisper are joined with a single newline (whisper last), text format."""
    _listener_env(monkeypatch)
    session_id = "whisper-compose-text"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(cache_path, _pending_map("kb-00099", "Authflow", "Auth details"), [])

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _make_fake_urlopen(
            [
                {
                    "project_ref": "personal-kb",
                    "maps": [
                        {"id": "kb-00001", "short_title": "ingest", "long_title": "Ingestion flow"}
                    ],
                }
            ]
        ),
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    lines = out.split("\n")
    assert any("Maps for personal-kb" in ln for ln in lines), f"Directory missing: {out!r}"
    assert any("Possibly relevant map" in ln for ln in lines), f"Whisper missing: {out!r}"
    # Whisper comes after directory
    dir_idx = next(i for i, ln in enumerate(lines) if "Maps for personal-kb" in ln)
    whi_idx = next(i for i, ln in enumerate(lines) if "Possibly relevant map" in ln)
    assert whi_idx > dir_idx


def test_directory_and_whisper_composed_claude_json(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Directory + whisper are in ONE additionalContext field (claude-json format)."""
    _listener_env(monkeypatch)
    session_id = "whisper-compose-json"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(cache_path, _pending_map("kb-00099", "Authflow", "Auth details"), [])

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _make_fake_urlopen(
            [
                {
                    "project_ref": "personal-kb",
                    "maps": [
                        {"id": "kb-00001", "short_title": "ingest", "long_title": "Ingestion flow"}
                    ],
                }
            ]
        ),
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=claude-json"],
    )
    assert rc == 0
    obj = json.loads(out)
    context = obj["hookSpecificOutput"]["additionalContext"]
    assert "Maps for personal-kb" in context
    assert "Possibly relevant map" in context
    # whisper appears after directory in the combined string
    assert context.index("Maps for personal-kb") < context.index("Possibly relevant map")


def test_whisper_only_claude_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Whisper-only output (no directory) wraps in one hookSpecificOutput envelope."""
    _listener_env(monkeypatch)
    session_id = "whisper-only-json"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(cache_path, _pending_map("kb-00099", "Authflow", "Auth details"), [])

    # No .kb_project -> directory pipeline returns early

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=claude-json"],
    )
    assert rc == 0
    obj = json.loads(out)
    assert obj["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    context = obj["hookSpecificOutput"]["additionalContext"]
    assert "Possibly relevant map" in context
    assert "kb-00099" in context
    assert "Maps for" not in context


def test_session_start_leaves_listener_cache_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """SessionStart does not read or mutate the listener cache."""
    _listener_env(monkeypatch)
    session_id = "session-start-cache"
    cache_path = get_listener_cache_path(session_id)
    cache_content = {"pending": _pending_map("kb-00099"), "whispered_map_ids": []}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    original_bytes = json.dumps(cache_content).encode("utf-8")
    cache_path.write_bytes(original_bytes)

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-00001", "short_title": "auth", "long_title": "Auth flow"}],
    )

    _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )

    # Cache must be byte-identical
    assert cache_path.read_bytes() == original_bytes


def test_extract_manifest_raising_still_emits_directory(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Listener failure (extract_manifest raises) leaves directory emission intact."""
    _listener_env(monkeypatch)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated listener failure")

    monkeypatch.setattr(listener, "extract_manifest", boom)
    # Also break read_listener_cache so the whisper check fails
    monkeypatch.setattr(listener, "read_listener_cache", boom)

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _make_fake_urlopen(
            [
                {
                    "project_ref": "personal-kb",
                    "maps": [{"id": "kb-00001", "short_title": "auth", "long_title": "Auth flow"}],
                }
            ]
        ),
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": "resilience-test",
        },
    )
    assert rc == 0
    assert "Maps for personal-kb" in out
    assert "auth" in out


# ===========================================================================
# P1 flagship tests: byte-identical legacy identity + one-KB-down never-raise
# ===========================================================================


def test_byte_identical_absent_roster_single_kb(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """ABSENT roster (no kbs.json) + legacy env vars → emitted directory is
    byte-identical to the pre-P1 single-KB output.

    Asserts the exact pre-change golden string for both Line 1
    (``Maps for {project} — ...``) and Line 2
    (``Maps in other domains — {proj}: ...``) with NO label prefix anywhere
    (only the single 'personal' label is present).
    """
    # No kbs.json exists under XDG_CONFIG_HOME (the fixture pins it under
    # tmp_path/.config which is never created). Legacy fallback fires.
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")
    service_response = [
        {
            "project_ref": "personal-kb",
            "maps": [
                {"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"},
            ],
        },
        {
            "project_ref": "agent-gtd",
            "maps": [
                {"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"},
            ],
        },
    ]
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-byte-identical-absent-roster",
        },
        ["--format=text"],
    )
    assert rc == 0
    # Exact pre-change golden string. The 'personal/' label prefix MUST NOT
    # appear anywhere — single-label mode is byte-identical to pre-P1.
    expected = (
        "Maps for personal-kb — [kb-1] auth: Authentication map\n"
        "Maps in other domains — agent-gtd: [gtd-1] tasks"
    )
    assert out == expected, f"output drifted: {out!r}"
    assert "personal/" not in out


def test_one_kb_down_error_isolation_emits_surviving(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A 2-label roster where ONE KB's maps-index fast-raises:
    the surviving KB's maps are emitted and NO exception propagates.
    """
    import urllib.error

    # Wire a 2-KB roster via kbs.json + key_file indirection.
    config_dir = hook_env["root"] / ".config" / "personal_kb"
    config_dir.mkdir(parents=True)
    key_file_p = hook_env["root"] / "personal.key"
    key_file_t = hook_env["root"] / "team.key"
    key_file_p.write_text("p-secret\n", encoding="utf-8")
    key_file_t.write_text("t-secret\n", encoding="utf-8")
    (config_dir / "kbs.json").write_text(
        json.dumps(
            [
                {
                    "label": "personal",
                    "url": "https://personal.kb/",
                    "key_file": str(key_file_p),
                },
                {
                    "label": "team",
                    "url": "https://team.kb/",
                    "key_file": str(key_file_t),
                },
            ]
        ),
        encoding="utf-8",
    )

    # personal KB returns one map; team KB raises immediately.
    def fake_urlopen(req: Any, timeout: float = 3.0) -> Any:
        full = req.full_url
        if "personal.kb" in full:
            body = json.dumps(
                {
                    "projects": [
                        {
                            "project_ref": "agent-gtd",
                            "maps": [
                                {
                                    "id": "gtd-1",
                                    "short_title": "tasks",
                                    "long_title": "Tasks",
                                }
                            ],
                        }
                    ]
                }
            ).encode("utf-8")
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = body
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
            return mock_resp
        raise urllib.error.URLError("team kb is down")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-one-kb-down-isolate",
        },
        ["--format=text"],
    )
    # Exit cleanly, no exception escaped.
    assert rc == 0
    # Surviving KB's data flowed through. Single-label-after-isolation:
    # only 'personal' contributed, so output stays in the byte-identical
    # single-KB form (no label prefix).
    assert out.startswith("Maps in other domains — ")
    assert "agent-gtd:" in out
    assert "[gtd-1] tasks" in out
    # Team's label must not appear (it contributed nothing).
    assert "team/" not in out


def test_one_kb_down_wall_deadline_elapsed_lt_3p5s(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """A 2-label roster where one KB's urlopen BLOCKS past the 3.0s wall
    deadline: the hook emits the surviving KB's maps and elapsed wall time
    is < 3.5s. Proves the single ``wait(timeout=3.0)`` deadline actually
    fires — not a 1.5s fast-fail per call (which would not exercise the cap).
    """
    import threading
    import time

    config_dir = hook_env["root"] / ".config" / "personal_kb"
    config_dir.mkdir(parents=True)
    key_file_p = hook_env["root"] / "personal.key"
    key_file_t = hook_env["root"] / "team.key"
    key_file_p.write_text("p-secret\n", encoding="utf-8")
    key_file_t.write_text("t-secret\n", encoding="utf-8")
    (config_dir / "kbs.json").write_text(
        json.dumps(
            [
                {
                    "label": "personal",
                    "url": "https://personal.kb/",
                    "key_file": str(key_file_p),
                },
                {
                    "label": "slow",
                    "url": "https://slow.kb/",
                    "key_file": str(key_file_t),
                },
            ]
        ),
        encoding="utf-8",
    )

    block_event = threading.Event()  # never set during the timed window

    def fake_urlopen(req: Any, timeout: float = 3.0) -> Any:
        full = req.full_url
        if "personal.kb" in full:
            body = json.dumps(
                {
                    "projects": [
                        {
                            "project_ref": "agent-gtd",
                            "maps": [
                                {
                                    "id": "gtd-1",
                                    "short_title": "tasks",
                                    "long_title": "Tasks",
                                }
                            ],
                        }
                    ]
                }
            ).encode("utf-8")
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = body
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
            return mock_resp
        # Slow KB blocks; release after a hard ceiling so we never wedge.
        block_event.wait(timeout=10.0)
        body = json.dumps({"projects": []}).encode("utf-8")
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
        return mock_resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    start = time.monotonic()
    try:
        rc, out = _run(
            monkeypatch,
            {
                "hook_event_name": "SessionStart",
                "cwd": str(hook_env["root"]),
                "session_id": "s-one-kb-down-wall",
            },
            ["--format=text"],
        )
    finally:
        block_event.set()
    elapsed = time.monotonic() - start

    # The single wait(3.0) deadline must have fired — wall time is bounded.
    assert elapsed < 3.5, f"hook took {elapsed:.2f}s (>= 3.5s wall budget)"
    assert rc == 0
    # Surviving KB's data flowed through.
    assert "agent-gtd:" in out
    assert "[gtd-1] tasks" in out
    # Slow KB contributed nothing.
    assert "slow/" not in out


# ===========================================================================
# P2 whisper-injection tests: byte-identical single-KB + multi-KB prefix +
# one-per-KB emission + legacy back-parse
# ===========================================================================


def _write_multi_kb_kbs_json(
    hook_env: dict[str, Path],
    labels: list[str],
) -> None:
    """Drop a kbs.json file with the given labels (URLs/keys unused in whisper path)."""
    config_dir = hook_env["root"] / ".config" / "personal_kb"
    config_dir.mkdir(parents=True)
    entries: list[dict[str, str]] = []
    for label in labels:
        kf = hook_env["root"] / f"{label}.key"
        kf.write_text(f"{label}-secret\n", encoding="utf-8")
        entries.append(
            {
                "label": label,
                "url": f"https://{label}.kb/",
                "key_file": str(kf),
            }
        )
    (config_dir / "kbs.json").write_text(json.dumps(entries), encoding="utf-8")


def test_whisper_single_kb_byte_identical_no_label_prefix(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """AC-10 single-KB identity: roster of exactly one entry yields a label-free whisper.

    With the legacy 'personal' fallback roster (no kbs.json) and a P2 list
    of pending pointers containing a single entry, the emitted whisper
    string equals the pre-P2 byte-identical form:
    ``Possibly relevant map — [<id>] <short_title>`` (empty long_title path,
    which is the production case since ListenerPointer has no long_title).
    """
    _listener_env(monkeypatch)
    session_id = "whisper-single-kb-identity"
    cache_path = get_listener_cache_path(session_id)
    # P2 schema: pending is a LIST of per-KB pointer dicts.
    _write_listener_cache(
        cache_path,
        [_pending_map("kb-00099", "Authflow", "", label="personal")],
        [],
    )

    # No .kb_project → directory pipeline returns early → whisper-only output.
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    # Byte-identical pre-P2 whisper string — NO 'personal/' label prefix.
    assert out == "Possibly relevant map — [kb-00099] Authflow"
    assert "personal/" not in out


def test_whisper_multi_kb_label_prefix(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """AC-8 multi-KB: with len(load_roster()) > 1, the whisper carries '<label>/' prefix."""
    _listener_env(monkeypatch)
    _write_multi_kb_kbs_json(hook_env, ["personal", "team"])

    session_id = "whisper-multi-kb-prefix"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [
            _pending_map("kb-team-1", "TeamAuth", "", label="team"),
        ],
        [],
    )

    # No .kb_project → directory pipeline returns early → whisper-only output.
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    # Multi-KB form: '<label>/' before the '[<id>]' group.
    assert out == "Possibly relevant map — team/[kb-team-1] TeamAuth"


def test_whisper_one_per_kb_emission_two_labels_same_turn(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """AC-9 one-per-KB emission: two KBs may whisper in one turn on different titles.

    Ordering is deterministic: tie-break winner ('personal' when source_label
    is not a roster label) goes first; remaining labels sorted ascending.
    """
    _listener_env(monkeypatch)
    _write_multi_kb_kbs_json(hook_env, ["personal", "team"])

    session_id = "whisper-one-per-kb"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [
            # Note: order in cache file is reversed from emit order to prove
            # cli.py sorts by tie-break-winner-first, NOT by file order.
            _pending_map("kb-team-1", "TeamMap", "", label="team"),
            _pending_map("kb-personal-1", "PersonalMap", "", label="personal"),
        ],
        [],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    # Both whispers present, one per label, 'personal' first.
    lines = out.split("\n")
    assert len(lines) == 2, f"expected 2 whisper lines, got: {out!r}"
    assert lines[0] == "Possibly relevant map — personal/[kb-personal-1] PersonalMap"
    assert lines[1] == "Possibly relevant map — team/[kb-team-1] TeamMap"


def test_whisper_legacy_back_parse_single_dict_pending(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A pre-P2 cache (pending as single dict, bare-id whispered_map_ids) is read tolerantly.

    The legacy single-dict pending is wrapped with label='personal', and
    bare-id whispered_map_ids are back-parsed to ['personal', <id>].
    """
    _listener_env(monkeypatch)
    session_id = "whisper-legacy-backparse"
    cache_path = get_listener_cache_path(session_id)
    # LEGACY pre-P2 shape: pending is a single dict (no label key), and
    # whispered_map_ids is a list of bare-id strings.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_cache = {
        "pending": {
            "id": "kb-legacy-1",
            "short_title": "LegacyMap",
            "long_title": "Legacy details",
        },
        "whispered_map_ids": ["kb-prev-1", "kb-prev-2"],
    }
    cache_path.write_text(json.dumps(legacy_cache), encoding="utf-8")

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    # Single-KB roster (legacy env fallback) → no label prefix; populated
    # long_title path.
    assert out == "Possibly relevant map — [kb-legacy-1] LegacyMap: Legacy details"

    # Post-emission cache state: pending=[] (P2 schema), bare-id back-parse
    # of pre-existing whispered_map_ids preserved, new emission appended.
    updated = json.loads(cache_path.read_text(encoding="utf-8"))
    assert updated["pending"] == []
    assert ["personal", "kb-prev-1"] in updated["whispered_map_ids"]
    assert ["personal", "kb-prev-2"] in updated["whispered_map_ids"]
    assert ["personal", "kb-legacy-1"] in updated["whispered_map_ids"]


def test_whisper_legacy_back_parse_skips_already_whispered(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A legacy bare-id whispered_map_ids matching the pending id suppresses re-whisper."""
    _listener_env(monkeypatch)
    session_id = "whisper-legacy-skip"
    cache_path = get_listener_cache_path(session_id)
    legacy_cache = {
        "pending": {
            "id": "kb-already",
            "short_title": "AlreadySeen",
            "long_title": "",
        },
        # Bare-id back-parses to ['personal', 'kb-already'] — matches the
        # legacy-back-parsed pending pointer, so re-whisper is suppressed.
        "whispered_map_ids": ["kb-already"],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(legacy_cache), encoding="utf-8")

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    assert "Possibly relevant map" not in out


def test_whisper_label_id_dedup_blocks_re_emission(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A pending (label, id) already in whispered_map_ids is not whispered again."""
    _listener_env(monkeypatch)
    session_id = "whisper-pair-dedup"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [_pending_map("kb-seen", "Seen", "", label="team")],
        [["team", "kb-seen"]],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    assert "Possibly relevant map" not in out


def test_whisper_same_id_different_labels_dont_collide(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """Same bare id under DIFFERENT labels are treated as distinct (label,id) keys."""
    _listener_env(monkeypatch)
    _write_multi_kb_kbs_json(hook_env, ["personal", "team"])

    session_id = "whisper-id-collide"
    cache_path = get_listener_cache_path(session_id)
    # Already whispered: ('personal', 'kb-77').
    # Pending: ('team', 'kb-77') — should still whisper (different label).
    _write_listener_cache(
        cache_path,
        [_pending_map("kb-77", "Shared", "", label="team")],
        [["personal", "kb-77"]],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    assert out == "Possibly relevant map — team/[kb-77] Shared"


def test_whisper_two_pointers_same_kb_one_line(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """GTD 66ea1fe4: a single KB's 2 pending pointers render on ONE line,
    comma-separated, with the pluralized header — not two separate lines."""
    _listener_env(monkeypatch)
    session_id = "whisper-plural-one-kb"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [
            _pending_map("kb-00001", "MapA", "", label="personal"),
            _pending_map("kb-00002", "MapB", "", label="personal"),
        ],
        [],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    assert out == "Possibly relevant maps — [kb-00001] MapA, [kb-00002] MapB"

    # Post-emission: pending cleared, BOTH ids recorded as whispered.
    updated = json.loads(cache_path.read_text(encoding="utf-8"))
    assert updated["pending"] == []
    assert ["personal", "kb-00001"] in updated["whispered_map_ids"]
    assert ["personal", "kb-00002"] in updated["whispered_map_ids"]


def test_whisper_two_pointers_one_already_whispered_only_other_emits(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """When one of a KB's 2 pending pointers was already whispered this
    session, only the NEW one is emitted — as a singular-header line."""
    _listener_env(monkeypatch)
    session_id = "whisper-plural-partial-suppress"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [
            _pending_map("kb-00001", "MapA", "", label="personal"),
            _pending_map("kb-00002", "MapB", "", label="personal"),
        ],
        [["personal", "kb-00001"]],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    assert out == "Possibly relevant map — [kb-00002] MapB"
    assert "kb-00001" not in out


def test_whisper_three_pending_same_kb_capped_at_two(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A cache file with 3 pending pointers for one KB (should never happen
    given the worker's own cap, but defended anyway) drains only 2."""
    _listener_env(monkeypatch)
    session_id = "whisper-plural-overcap"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [
            _pending_map("kb-00001", "MapA", "", label="personal"),
            _pending_map("kb-00002", "MapB", "", label="personal"),
            _pending_map("kb-00003", "MapC", "", label="personal"),
        ],
        [],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        args=["--format=text"],
    )
    assert rc == 0
    assert out == "Possibly relevant maps — [kb-00001] MapA, [kb-00002] MapB"
    assert "kb-00003" not in out


# ---------------------------------------------------------------------------
# Whisper-debug PROMPT-path lines (UserPromptSubmit)
# ---------------------------------------------------------------------------


def test_whisper_debug_prompt_inject_written_on_fresh_whisper(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A fresh pending whisper writes ``PROMPT inject <id> "<short_title>"``."""
    _listener_env(monkeypatch)
    session_id = "whisper-debug-inject"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [_pending_map("kb-00099", "Authflow", "Auth details")],
        [],
    )

    rc, _out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0

    debug_log = hook_env["root"] / ".cache" / "personal_kb" / f"whisper-debug-{session_id}.log"
    assert debug_log.exists(), f"whisper-debug log missing at {debug_log}"
    content = debug_log.read_text(encoding="utf-8")
    assert 'PROMPT inject kb-00099 "Authflow"' in content
    # No suppress line: nothing was already whispered.
    assert "PROMPT suppress" not in content


def test_whisper_debug_prompt_suppress_written_when_already_whispered(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """A candidate whose (label, id) is already in ``whispered_map_ids`` writes
    ``PROMPT suppress <id> (already whispered this session)`` and NO inject line.
    """
    _listener_env(monkeypatch)
    session_id = "whisper-debug-suppress"
    cache_path = get_listener_cache_path(session_id)
    # The pending whisper has the SAME (label, id) as an entry in whispered_map_ids
    # -> the pre_set filter drops it -> a suppress debug line is written.
    _write_listener_cache(
        cache_path,
        [_pending_map("kb-00099", "Authflow", "")],
        [["personal", "kb-00099"]],
    )

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0
    # Whisper itself is NOT emitted (the pre_set drop empties the candidate list).
    assert "Possibly relevant map" not in out

    debug_log = hook_env["root"] / ".cache" / "personal_kb" / f"whisper-debug-{session_id}.log"
    assert debug_log.exists()
    content = debug_log.read_text(encoding="utf-8")
    assert "PROMPT suppress kb-00099 (already whispered this session)" in content
    # No inject line: the candidate was dropped before the commit point.
    assert "PROMPT inject" not in content


def test_whisper_debug_prompt_inject_empty_short_title_renders_empty_quotes(
    monkeypatch: pytest.MonkeyPatch,
    hook_env: dict[str, Path],
) -> None:
    """An empty short_title renders verbatim inside double quotes: ``PROMPT inject kb-X ""``.

    Pins the AC behaviour: do NOT fall back to the id when short_title is
    empty — the empty-string rendering is itself a debugging signal.
    """
    _listener_env(monkeypatch)
    session_id = "whisper-debug-empty-title"
    cache_path = get_listener_cache_path(session_id)
    _write_listener_cache(
        cache_path,
        [_pending_map("kb-empty-title", short_title="", long_title="")],
        [],
    )

    rc, _out = _run(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
    )
    assert rc == 0

    debug_log = hook_env["root"] / ".cache" / "personal_kb" / f"whisper-debug-{session_id}.log"
    content = debug_log.read_text(encoding="utf-8")
    assert 'PROMPT inject kb-empty-title ""' in content


# ---------------------------------------------------------------------------
# read_prompt_text helper (GTD 022837d4) — not wired into any behavior yet.
# ---------------------------------------------------------------------------


def test_read_prompt_text_returns_prompt_value() -> None:
    assert cli.read_prompt_text({"prompt": "hello there"}) == "hello there"


def test_read_prompt_text_returns_user_input_value_when_prompt_absent() -> None:
    assert cli.read_prompt_text({"user_input": "docs spelling"}) == "docs spelling"


def test_read_prompt_text_prefers_prompt_when_both_present() -> None:
    assert cli.read_prompt_text({"prompt": "real one", "user_input": "docs one"}) == "real one"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"user_input": ""},
        {"prompt": "", "user_input": ""},
        {"prompt": 123},
        {"user_input": 123},
        {"prompt": None, "user_input": None},
        {"other_key": "value"},
    ],
)
def test_read_prompt_text_returns_none_for_missing_empty_or_non_string(
    payload: dict[str, Any],
) -> None:
    assert cli.read_prompt_text(payload) is None


def test_defaults_target_local_daemon_and_stay_silent_when_nothing_listens(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """No URL/key in env: hook targets the local default, never spawns, exits silent."""
    import subprocess
    import urllib.request

    from personal_kb_hook.defaults import LOCAL_KB_API_KEY, LOCAL_KB_URL, resolve_url_key

    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    assert resolve_url_key() == (LOCAL_KB_URL, LOCAL_KB_API_KEY)
    assert (LOCAL_KB_URL, LOCAL_KB_API_KEY) == ("http://127.0.0.1:8765", "local-no-auth")

    seen: list[str] = []

    def refuse(req: Any, *a: object, **k: object) -> Any:
        seen.append(getattr(req, "full_url", str(req)))
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    spawned: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(hook_env["root"]), "session_id": "d1"},
    )
    assert rc == 0
    assert out == ""
    assert spawned == []
    assert all(u.startswith(LOCAL_KB_URL) for u in seen)
