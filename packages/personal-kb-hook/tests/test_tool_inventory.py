"""Tests for the SessionStart tool inventory."""

import io
import json
import unittest.mock
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import cli, tool_inventory
from personal_kb_hook.render import BANNED_TOKENS
from personal_kb_hook.tool_inventory import build_inventory, describe


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.setenv("PERSONAL_KB_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KB_TOOL_INVENTORY", "1")
    return {"root": tmp_path}


def _tool(d: Path, name: str, content: str = "#!/bin/sh\n", mode: int = 0o755) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(content, encoding="utf-8")
    p.chmod(mode)
    return p


def _log_rows(root: Path) -> list[dict[str, Any]]:
    p = root / ".cache" / "personal_kb" / "tool-inventory.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()]


def _run(monkeypatch: pytest.MonkeyPatch, payload: Any) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    cli.main(["--format=claude-json"])
    return buf.getvalue()


def _fake_urlopen(projects: list[dict[str, Any]]) -> Any:
    def fake(req: object, timeout: float = 3.0) -> object:
        resp = unittest.mock.MagicMock()
        resp.read.return_value = json.dumps({"projects": projects}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = unittest.mock.MagicMock(return_value=False)
        return resp

    return fake


# --------------------------------------------------------------------------
# describe()
# --------------------------------------------------------------------------

_LONG = "# " + "a" * 300


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        (
            "add_remote.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n\n# add_remote.sh — Create a bare repo on "
            "the git server and add it as a remote.\n#\n# Usage: add_remote.sh [remote-name]\n",
            "Create a bare repo on the git server and add it as a remote.",
        ),
        (
            "collect.sh",
            "#!/usr/bin/env bash\n#\n# Collect Adobe Camera Raw / Lightroom .dcp camera profiles "
            "into a tarball.\n",
            "Collect Adobe Camera Raw / Lightroom .dcp camera profiles into a tarball.",
        ),
        (
            "bounce_video.sh",
            "#!/bin/bash\n\n# Usage: bounce_video.sh --bounce-forward <frame> --bounce-backward "
            "<frame>\n",
            "Usage: bounce_video.sh --bounce-forward <frame> --bounce-backward <frame>",
        ),
        (
            "key.py",
            '#!/usr/bin/env python3\n"""Create the gritmile-deploy IAM access key and install '
            "it as a local AWS profile.\n",
            "Create the gritmile-deploy IAM access key and install it as a local AWS profile.",
        ),
        ("e.sh", '#!/usr/bin/env bash\nset -euo pipefail\n\nREMOTE="x"\n# later comment\n', None),
        ("f.sh", "#!/bin/bash\n\nollama list | tail -n +2\n", None),
        ("j.py", '#!/usr/bin/env python3\n"""One-line doc."""\n', "One-line doc."),
        ("k.sh", "#\n" * 31 + "# late\n", None),
        ("multi.py", '"""\nSecond line doc.\n"""\n', "Second line doc."),
    ],
)
def test_describe_golden(tmp_path: Path, name: str, content: str, expected: str | None) -> None:
    assert describe(_tool(tmp_path, name, content)) == expected


def test_describe_big_binary_long(tmp_path: Path) -> None:
    assert describe(_tool(tmp_path, "big", "# x\n" + "y" * 70000)) is None
    p = tmp_path / "bin"
    p.write_bytes(b"#!/bin/sh\n\x00# hi\n")
    assert describe(p) is None
    out = describe(_tool(tmp_path, "long", _LONG + "\n"))
    assert out is not None
    assert len(out) == 120
    assert out.endswith("…")
    assert describe(tmp_path / "missing") is None


def test_templates_have_no_banned_tokens() -> None:
    templates = [
        tool_inventory.HEADER_PREFIX,
        ":",
        tool_inventory.OVERFLOW_FMT.format(n=0),
        tool_inventory.TRUNCATED_LINE,
    ]
    for t in templates:
        assert not any(b in t.lower() for b in BANNED_TOKENS)


# --------------------------------------------------------------------------
# build_inventory()
# --------------------------------------------------------------------------


def test_kill_switch(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _tool(env["root"] / "t", "a.sh")
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "t"))
    for v in ("0", "false", "No", " OFF "):
        monkeypatch.setenv("KB_TOOL_INVENTORY", v)
        assert build_inventory() is None
    assert _log_rows(env["root"])[-1]["outcome"] == "disabled"
    monkeypatch.setenv("KB_TOOL_INVENTORY", "")
    assert build_inventory() is not None


def test_dedup_and_relative(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    r = env["root"]
    _tool(r / "d1", "dup", "# one\n")
    _tool(r / "d1", "nx", "# nx1\n", mode=0o644)
    _tool(r / "d2", "dup", "# two\n")
    _tool(r / "d2", "nx", "# nx2\n")
    monkeypatch.setenv("KB_TOOL_DIRS", f"relative/dir:{r}/d1::{r}/d2")
    out = build_inventory()
    assert out is not None
    assert "dup — one" in out
    assert "dup — two" not in out
    assert "nx — nx2" in out
    assert "relative" not in out
    assert out.splitlines()[0] == "Personal tools in ~/d1:"


def test_filter(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    d = env["root"] / "t"
    _tool(d, "add.sh")
    _tool(d, "x.workflow.js", mode=0o644)
    _tool(d, "notes.md")
    _tool(d, ".hidden")
    _tool(d, "foo~")
    _tool(d / "sub", "inner")
    (d / "link").symlink_to(d / "add.sh")
    (d / "dead").symlink_to(d / "missing")
    monkeypatch.setenv("KB_TOOL_DIRS", str(d))
    out = build_inventory()
    assert out is not None
    assert out.splitlines()[1:] == ["add.sh", "link"]


def test_display_dir_home_prefix(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/x/h")
    assert tool_inventory._display_dir("/x/h2") == "/x/h2"
    assert tool_inventory._display_dir("/x/h/scripts") == "~/scripts"
    assert tool_inventory._display_dir("/x/h") == "~"


def test_max_tools_overflow(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    d = env["root"] / "t"
    for i in range(50):
        _tool(d, f"n{i:02d}")
    monkeypatch.setenv("KB_TOOL_DIRS", str(d))
    out = build_inventory()
    assert out is not None
    lines = out.splitlines()
    assert len(lines) == 1 + 40 + 1
    assert lines[-1] == "(+10 more not shown)"
    row = _log_rows(env["root"])[-1]
    assert row["rendered_count"] == 40
    assert len(row["tools"]) == 40
    assert row["overflow_count"] == 10
    assert len(row["skipped_names"]) == 10


def test_max_chars(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    d = env["root"] / "t"
    for i in range(40):
        _tool(d, f"t{i:02d}", "# " + "d" * 130 + "\n")
    monkeypatch.setenv("KB_TOOL_DIRS", str(d))
    out = build_inventory()
    assert out is not None
    assert len(out) <= 2000
    lines = out.splitlines()
    rendered = len(lines) - 2
    assert rendered >= 1
    assert lines[-1] == f"(+{40 - rendered} more not shown)"


def test_time_budget(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    d = env["root"] / "t"
    for i in range(10):
        _tool(d, f"t{i:02d}")
    monkeypatch.setenv("KB_TOOL_DIRS", str(d))
    calls = {"n": 0}

    def clock() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] <= 4 else 1.0

    monkeypatch.setattr(tool_inventory, "_clock", clock)
    out = build_inventory()
    assert out is not None
    lines = out.splitlines()
    assert lines[1:-1] == ["t00", "t01", "t02"]
    assert lines[-1] == tool_inventory.TRUNCATED_LINE


def test_budget_empty(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _tool(env["root"] / "t", "a")
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "t"))
    vals = iter([0.0, 5.0])
    monkeypatch.setattr(tool_inventory, "_clock", lambda: next(vals))
    assert build_inventory() is None
    assert _log_rows(env["root"])[-1]["outcome"] == "budget_empty"


def test_outcomes(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "nope"))
    assert build_inventory(session_id="s", source="startup") is None
    row = _log_rows(env["root"])[-1]
    assert row["outcome"] == "no_dirs"
    assert row["session_id"] == "s"
    assert row["source"] == "startup"
    (env["root"] / "empty").mkdir()
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "empty"))
    assert build_inventory() is None
    assert _log_rows(env["root"])[-1]["outcome"] == "no_tools"
    _tool(env["root"] / "empty", "a")
    assert build_inventory() is not None
    assert _log_rows(env["root"])[-1]["outcome"] == "emitted"


def test_fail_open_logs_error(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _tool(env["root"] / "t", "a")
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "t"))

    def boom(path: Path) -> str:
        raise RuntimeError("x")

    monkeypatch.setattr(tool_inventory, "describe", boom)
    assert build_inventory() is None
    row = _log_rows(env["root"])[-1]
    assert row["outcome"] == "error"
    assert row["error_type"] == "RuntimeError"


def test_log_rotation(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    log = env["root"] / ".cache" / "personal_kb" / "tool-inventory.jsonl"
    log.write_text("x" * int(1.1 * 1024 * 1024))
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "nope"))
    build_inventory()
    assert log.with_name("tool-inventory.jsonl.1").exists()
    assert len(log.read_text().splitlines()) == 1


# --------------------------------------------------------------------------
# CLI integration
# --------------------------------------------------------------------------


@pytest.fixture
def tools(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> str:
    _tool(env["root"] / "scripts", "hello.sh", "#!/bin/sh\n# Say hello\n")
    monkeypatch.setenv("KB_TOOL_DIRS", str(env["root"] / "scripts"))
    inv = build_inventory()
    assert inv is not None
    return inv


def _ctx(out: str) -> str:
    obj = json.loads(out)
    assert obj["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    return str(obj["hookSpecificOutput"]["additionalContext"])


def _project(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    proj = env["root"] / "proj"
    proj.mkdir()
    (proj / ".kb_project").write_text("demo\n")
    maps = [
        {
            "id": "kb-00001",
            "title": "Map one",
            "short_title": "Map one",
            "description": "d",
            "scope": "demo",
        }
    ]
    monkeypatch.setattr(
        urllib.request, "urlopen", _fake_urlopen([{"project_ref": "demo", "maps": maps}])
    )
    return proj


def test_cli_no_project(env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _run(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(env["root"]), "session_id": "s1"},
    )
    assert _ctx(out) == tools


def test_cli_disabled_empty(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_TOOL_INVENTORY", "false")
    out = _run(monkeypatch, {"hook_event_name": "SessionStart", "cwd": str(env["root"])})
    assert out == ""


def test_cli_with_maps_and_suppressed(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = _project(env, monkeypatch)
    payload = {
        "hook_event_name": "SessionStart",
        "cwd": str(proj),
        "session_id": "sess-c",
        "source": "startup",
    }
    monkeypatch.setenv("KB_TOOL_INVENTORY", "0")
    directory = _ctx(_run(monkeypatch, {**payload, "session_id": "other"}))
    monkeypatch.setenv("KB_TOOL_INVENTORY", "1")
    assert _ctx(_run(monkeypatch, payload)) == directory + "\n\n" + tools
    assert _ctx(_run(monkeypatch, payload)) == tools


def test_cli_headless_engine(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADLESS_BUILD_ENGINE", "claude-code-sonnet")
    out = _run(monkeypatch, {"hook_event_name": "SessionStart", "cwd": str(env["root"])})
    assert _ctx(out) == tools


def test_cli_user_prompt_submit(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(
        monkeypatch,
        {"hook_event_name": "UserPromptSubmit", "cwd": str(env["root"]), "prompt": "hi"},
    )
    assert tool_inventory.HEADER_PREFIX not in out


def test_cli_compose_directory_raises(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = _project(env, monkeypatch)

    def boom(*a: object, **k: object) -> str:
        raise RuntimeError("x")

    monkeypatch.setattr(cli, "compose_directory", boom)
    out = _run(
        monkeypatch, {"hook_event_name": "SessionStart", "cwd": str(proj), "session_id": "s"}
    )
    assert _ctx(out) == tools


def test_cli_mark_emitted_raises(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = _project(env, monkeypatch)
    payload = {"hook_event_name": "SessionStart", "cwd": str(proj), "session_id": "s-g"}
    monkeypatch.setenv("KB_TOOL_INVENTORY", "0")
    directory = _ctx(_run(monkeypatch, {**payload, "session_id": "other"}))
    monkeypatch.setenv("KB_TOOL_INVENTORY", "1")

    def boom(**k: object) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(cli, "mark_emitted", boom)
    out = _run(monkeypatch, payload)
    assert _ctx(out) == directory + "\n\n" + tools


def test_cli_no_cwd(env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _run(monkeypatch, {"hook_event_name": "SessionStart"})
    assert _ctx(out) == tools


@pytest.mark.parametrize("source", ["resume", "clear", "compact"])
def test_cli_sources(
    env: dict[str, Path], tools: str, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    out = _run(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(env["root"]), "source": source},
    )
    assert _ctx(out) == tools
