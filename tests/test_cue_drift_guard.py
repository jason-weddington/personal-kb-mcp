"""Drift guard: the hook's vendored ``cues_lite`` matches ``kb_core.cues``.

The standalone hook must not import ``kb_core``, so the PreToolUse soft gate
carries stdlib copies of ``extract_target`` and ``target_class``. If either
side changes, this test fails until both agree again.
"""

from __future__ import annotations

import pytest
from kb_core import cues
from personal_kb_hook import cues_lite

_EXTRACT_VECTORS: list[tuple[str, object]] = [
    ("Bash", {"command": "ls -la"}),
    ("Read", {"file_path": "/a/b.py"}),
    ("Edit", {"file_path": "/a/b.py"}),
    ("Write", {"file_path": "/a/b.py"}),
    ("MultiEdit", {"file_path": "/a/b.py"}),
    ("NotebookEdit", {"notebook_path": "/a/n.ipynb"}),
    ("Glob", {"pattern": "**/*.py"}),
    ("Grep", {"pattern": "foo"}),
    ("Bash", {"command": 5}),
    ("Bash", {}),
    ("mcp__x__y", {"command": "x"}),
    ("Bash", None),
    ("Read", "not-a-dict"),
]

_BASH_VECTORS = [
    "cd /x && git push origin main",
    "cd /x || git fetch origin",
    "FOO=1 uv run pytest -q",
    "ls -la /tmp",
    "sudo -n -u dispatch git -C /x status",
    "/usr/bin/python3 -m pytest",
    "",
    "cd /x",
    "FOO=1",
    "git",
    "echo hi | grep h",
]

_FILE_VECTORS = [
    ("Read", "/a/b/c.PY"),
    ("Read", "/a/Makefile"),
    ("Read", ""),
    ("NotebookEdit", "/a/n.ipynb"),
    ("mcp__agent-gtd__get_item", ""),
    ("Grep", "foo"),
]

_PINNED = [
    ("git -C /x push", "git"),
    ("npm run build && npm test", "npm run"),
    ("docker compose up -d", "docker compose"),
    ("echo hi | grep h", "echo"),
    ("", ""),
    ("   ", ""),
]


@pytest.mark.parametrize(("tool", "tool_input"), _EXTRACT_VECTORS)
def test_extract_target_agrees(tool: str, tool_input: object) -> None:
    assert cues_lite.extract_target(tool, tool_input) == cues.extract_target(tool, tool_input)


@pytest.mark.parametrize("command", _BASH_VECTORS + [c for c, _ in _PINNED])
def test_bash_target_class_agrees(command: str) -> None:
    assert cues_lite.target_class("Bash", command) == cues.target_class("Bash", command)


@pytest.mark.parametrize(("tool", "target"), _FILE_VECTORS)
def test_other_target_class_agrees(tool: str, target: str) -> None:
    assert cues_lite.target_class(tool, target) == cues.target_class(tool, target)


@pytest.mark.parametrize(("command", "expected"), _PINNED)
def test_pinned_vectors(command: str, expected: str) -> None:
    assert cues.target_class("Bash", command) == expected
    assert cues_lite.target_class("Bash", command) == expected


_SEGMENT_VECTORS = [
    "git show HEAD -- README.md | tail -8; git push origin main",
    "cd /app\nmake smoke 2>&1 | tail -5",
    "FOO=1 sudo systemctl restart caddy; echo ok",
    "git commit -qam x && git push github main",
    "cd /x",
    "",
    "FOO=1",
    "cd /x && npm run build -- --mode prod || echo failed",
]


@pytest.mark.parametrize("command", _SEGMENT_VECTORS)
def test_bash_segments_agrees(command: str) -> None:
    assert cues_lite.bash_segments(command) == cues.bash_segments(command)
