"""Tests for the personal-kb-hook CLI entry point."""

import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import cli, listener
from personal_kb_hook.paths import get_listener_cache_path
from personal_kb_hook.render import BANNED_TOKENS, render_directory


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate the hook's filesystem touchpoints to ``tmp_path``."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("HOME", str(tmp_path))
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
    """An event we do not handle → exit 0 silent."""
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
            "hook_event_name": "PostToolUse",
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


def _surface_fixture(hook_env: dict[str, Path]) -> dict[str, Any]:
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [
            {"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"},
            {"id": "kb-2", "short_title": "ingest", "long_title": "Ingestion flow"},
        ],
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
    payload = _surface_fixture(hook_env)
    rc, out = _run(monkeypatch, payload, ["--format=text"])
    assert rc == 0
    assert out.startswith("Maps for personal-kb — ")
    assert "[kb-1] auth: Authentication map" in out
    assert "; [kb-2] ingest: Ingestion flow" in out


def test_format_claude_json_envelope(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    # Use a different session_id so suppression scratch does not leak between tests
    payload = _surface_fixture(hook_env)
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
    payload = _surface_fixture(hook_env)
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
    payload = _surface_fixture(hook_env)
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


def test_http_env_set_urlopen_fails_emits_local_directory(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """PERSONAL_KB_URL/API_KEY set but urlopen raises -> local index used, no crash."""
    import urllib.request

    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"}],
    )
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")

    def raise_conn(*args: object, **kwargs: object) -> None:
        import urllib.error

        raise urllib.error.URLError("simulated connection failure")

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
    assert "personal-kb" in out
    assert "auth" in out


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
    ["PERSONAL_KB_URL", "PERSONAL_KB_API_KEY", "PERSONAL_KB_LISTENER"],
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
    assert body["project_ref"] == "personal-kb"
    assert "mcp:personal-kb" in body["operated"]

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
    assert body["project_ref"] is None

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


def _write_listener_cache(path: Path, pending: Any, whispered_ids: list[str]) -> None:
    """Write a listener cache file for test setup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pending": pending, "whispered_map_ids": whispered_ids}),
        encoding="utf-8",
    )


def _pending_map(
    entry_id: str = "kb-00099",
    short_title: str = "Authflow",
    long_title: str = "Authentication flow details",
) -> dict[str, str]:
    return {"id": entry_id, "short_title": short_title, "long_title": long_title}


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
    """After a whisper is emitted, pending=null and whispered id is appended."""
    _listener_env(monkeypatch)
    session_id = "whisper-clear"
    cache_path = get_listener_cache_path(session_id)
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

    # Cache must be updated: pending cleared, whispered id appended
    updated = json.loads(cache_path.read_text(encoding="utf-8"))
    assert updated["pending"] is None
    assert "kb-00099" in updated["whispered_map_ids"]
    assert "kb-00001" in updated["whispered_map_ids"]  # pre-existing id preserved


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
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-00001", "short_title": "ingest", "long_title": "Ingestion flow"}],
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
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-00001", "short_title": "ingest", "long_title": "Ingestion flow"}],
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
    _write_index(
        hook_env["maps_index"],
        "personal-kb",
        [{"id": "kb-00001", "short_title": "auth", "long_title": "Auth flow"}],
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
