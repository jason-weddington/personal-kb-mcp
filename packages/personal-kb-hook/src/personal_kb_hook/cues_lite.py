"""Vendored lexical cue functions (stdlib-only copy of ``kb_core.cues``).

:func:`extract_target` and :func:`target_class` are VERBATIM copies of the
rules in ``kb_core.cues`` — the hook is zero-dependency and must never import
``kb_core``. The root test ``tests/test_cue_drift_guard.py`` asserts both
implementations agree; if you change one side, change the other (and bump
``kb_core.cues.CUE_NORMALIZER_VERSION``).

:func:`bash_args_after_class` is hook-only: it powers the soft gate's
``args_prefix`` narrowing.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit"})
_FILE_CLASS_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"})

_TWO_WORD_PROGRAMS = frozenset(
    {
        "git",
        "uv",
        "npm",
        "pnpm",
        "yarn",
        "docker",
        "kubectl",
        "aws",
        "gh",
        "cargo",
        "sam",
        "cdk",
        "terraform",
        "agent-gtd",
        "systemctl",
        "pip",
        "python",
        "python3",
        "pytest",
        "make",
    }
)

_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\|")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")


def extract_target(tool: str, tool_input: object) -> str:
    """Return the raw target a tool was pointed at, or ``''``."""
    if not isinstance(tool_input, dict):
        return ""
    if tool == "Bash":
        key = "command"
    elif tool in _FILE_TOOLS:
        key = "file_path"
    elif tool == "NotebookEdit":
        key = "notebook_path"
    elif tool in {"Glob", "Grep"}:
        key = "pattern"
    else:
        return ""
    value = tool_input.get(key)
    return value if isinstance(value, str) else ""


def _bash_tokens(command: str) -> list[str]:
    """Tokens of the first non-``cd`` segment, env assigns and sudo stripped."""
    segment: str | None = None
    for candidate in _SEGMENT_SPLIT.split(command):
        parts = candidate.split()
        if parts and parts[0] != "cd":
            segment = candidate
            break
    if segment is None:
        return []
    tokens = segment.split()
    while tokens and _ENV_ASSIGN.match(tokens[0]):
        tokens.pop(0)
    if tokens and tokens[0] == "sudo":
        tokens.pop(0)
        while tokens and tokens[0].startswith("-"):
            flag = tokens.pop(0)
            if flag in {"-u", "-g"} and tokens:
                tokens.pop(0)
    return tokens


def _is_two_word(tokens: list[str]) -> bool:
    word = PurePosixPath(tokens[0]).name
    return word in _TWO_WORD_PROGRAMS and len(tokens) > 1 and not tokens[1].startswith("-")


def _bash_class(command: str) -> str:
    """Classify a shell command by its program (and subcommand for known CLIs)."""
    tokens = _bash_tokens(command)
    if not tokens:
        return ""
    word = PurePosixPath(tokens[0]).name
    if _is_two_word(tokens):
        return f"{word} {tokens[1]}"
    return word


def target_class(tool: str, target: str) -> str:
    """Coarse class of a target: program for Bash, ``ext:<suffix>`` for file tools."""
    if tool == "Bash":
        return _bash_class(target)
    if tool in _FILE_CLASS_TOOLS:
        if not target:
            return ""
        suffix = PurePosixPath(target).suffix.lower().lstrip(".")
        return f"ext:{suffix}" if suffix else "ext:none"
    return ""


def bash_args_after_class(command: str) -> list[str]:
    """Tokens after those that produced the Bash ``target_class``, flags dropped."""
    tokens = _bash_tokens(command)
    if not tokens:
        return []
    rest = tokens[2:] if _is_two_word(tokens) else tokens[1:]
    return [t for t in rest if not t.startswith("-")]


_NEWLINE_SPLIT = re.compile(r"&&|\|\||;|\||\n")


def bash_segments(command: str) -> list[tuple[str, list[str]]]:
    """``(target_class, args_after_class)`` for every segment of ``command``.

    Hook-only (powers the soft gate). Splits on ``&&``, ``||``, ``;``, ``|`` and
    newlines; ``cd`` segments and segments that classify to ``''`` are skipped.
    Each segment is run through the same helpers as the single-segment path.
    """
    out: list[tuple[str, list[str]]] = []
    for segment in _NEWLINE_SPLIT.split(command):
        cls = _bash_class(segment)
        if not cls:
            continue
        # A segment starting with ``cd`` is skipped by _bash_tokens only when it
        # is not the last candidate; check explicitly.
        parts = segment.split()
        if parts and parts[0] == "cd":
            continue
        out.append((cls, bash_args_after_class(segment)))
    return out
