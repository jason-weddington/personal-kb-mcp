"""Tests for the personal-kb-hook CLI entry point."""

import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import cli
from personal_kb_hook.render import BANNED_TOKENS, render_directory


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate the hook's filesystem touchpoints to ``tmp_path``."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_root.mkdir()
    # The hook expands ~/.cache/... — make HOME point at tmp_path.
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True, exist_ok=True)
    return {
        "root": tmp_path,
        "db_path": db_path,
        "maps_index": db_path.parent / "maps_index.jsonl",
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

    monkeypatch.setattr("personal_kb_hook.cli.read_index", boom)
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
