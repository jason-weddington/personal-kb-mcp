"""Failure-cue normalizer: turn a raw tool failure into a stable cue key.

A *failure cue* is the identity of "the same mistake": the tool that failed,
a coarse class of what it was pointed at, a normalized error message, and
the project. Two failures that differ only in incidental detail (pids,
uuids, timestamps, paths, line numbers) collapse to the same ``cue_key``,
which lets us measure how often an agent repeats a mistake across sessions.

This module is the single source of truth for normalization. The kb-service
``POST /api/kb/event`` route (live ingest) and the offline transcript backfill
script both call :func:`build_cue`; neither carries its own rules.

Stdlib-only, no I/O, no environment reads.

**Versioning rule:** any change to a rule in :func:`extract_target`,
:func:`target_class`, :func:`normalize_error`, :func:`host_class_from_cwd`,
:func:`resolve_cue_project` or :func:`cue_key` (including soft values such as
the two-word program set, the keyword regex, the 200-char cap or the
4000-char truncation) MUST increment :data:`CUE_NORMALIZER_VERSION`, because
the version is part of the key and keys from different rule sets must never
be compared as equal.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

__all__ = [
    "CUE_NORMALIZER_VERSION",
    "FailureCue",
    "bash_segments",
    "build_cue",
    "cue_key",
    "extract_target",
    "host_class_from_cwd",
    "normalize_error",
    "resolve_cue_project",
    "target_class",
]

CUE_NORMALIZER_VERSION: int = 1

_ERROR_TRUNCATE = 4000
_NORMALIZED_CAP = 200

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
_NEWLINE_SPLIT = re.compile(r"&&|\|\||;|\||\n")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TRUNC_MARKER = re.compile(r"\.\.\. \[\d+ characters truncated\] \.\.\.")
_EXIT_LINE = re.compile(r"^Exit code (\d+)$")
_KEYWORD = re.compile(
    r"error|fail|denied|not found|no such|cannot|can't|refused|invalid|traceback"
    r"|exception|fatal|timed out|forbidden|unauthorized|conflict",
    re.I,
)

_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
        "<uuid>",
    ),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",
            re.I,
        ),
        "<ts>",
    ),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}\b", re.I), "<ts>"),
    (re.compile(r"\bkb-\d+\b", re.I), "<kb>"),
    (re.compile(r"\btoolu_\w+", re.I), "<id>"),
    (re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,}\b", re.I), "<hex>"),
    (
        re.compile(r"(?<![\w.])~?/[^\s:'\"`(),\[\]]*/[^\s:'\"`(),\[\]]*", re.I),
        "<path>",
    ),
    (re.compile(r"\bline \d+\b", re.I), "line <n>"),
    (re.compile(r":\d+(?=[:\s,)]|$)", re.I), ":<n>"),
    (re.compile(r"\b\d{2,}\b", re.I), "<n>"),
)
_WHITESPACE = re.compile(r"\s+")

_WINDOWS_CWD = re.compile(r"^[A-Za-z]:\\")
_RUN_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass(frozen=True)
class FailureCue:
    """A normalized tool failure plus its stable ``cue_key``."""

    tool: str
    target: str
    target_class: str
    normalized_error: str
    error_rule: str
    project: str
    project_source: str
    host_class: str
    normalizer_version: int
    cue_key: str


def extract_target(tool: str, tool_input: object) -> str:
    """Return the raw target a tool was pointed at, or ``''``.

    Bash → ``command``; file tools → ``file_path``; NotebookEdit →
    ``notebook_path``; Glob/Grep → ``pattern``. Anything else (including MCP
    tools), a missing key, a non-str value or a non-dict input yields ``''``.
    """
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


def bash_args_after_class(command: str) -> list[str]:
    """Tokens after those that produced the Bash ``target_class``, flags dropped."""
    tokens = _bash_tokens(command)
    if not tokens:
        return []
    rest = tokens[2:] if _is_two_word(tokens) else tokens[1:]
    return [t for t in rest if not t.startswith("-")]


def bash_segments(command: str) -> list[tuple[str, list[str]]]:
    """``(target_class, args_after_class)`` for every segment of ``command``.

    Splits on ``&&``, ``||``, ``;``, ``|`` and newlines; ``cd`` segments and
    segments that classify to ``''`` are skipped. Mirrors
    ``personal_kb_hook.cues_lite.bash_segments`` (drift-guarded).
    """
    out: list[tuple[str, list[str]]] = []
    for segment in _NEWLINE_SPLIT.split(command):
        cls = _bash_class(segment)
        if not cls:
            continue
        parts = segment.split()
        if parts and parts[0] == "cd":
            continue
        out.append((cls, bash_args_after_class(segment)))
    return out


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


def _normalize_line(line: str) -> str:
    for pattern, replacement in _SUBSTITUTIONS:
        line = pattern.sub(replacement, line)
    return _WHITESPACE.sub(" ", line.lower())


def normalize_error(text: str) -> tuple[str, str]:
    """Normalize raw error text; return ``(normalized_error, error_rule)``.

    ``error_rule`` is one of ``'keyword'``, ``'last_line'``, ``'exit_only'``
    or ``'empty'``.
    """
    text = _ANSI.sub("", text)
    text = _TRUNC_MARKER.sub("<trunc>", text)
    lines = [stripped for raw in text.split("\n") if (stripped := raw.strip())]
    if not lines:
        return "", "empty"

    exit_line: str | None = None
    match = _EXIT_LINE.fullmatch(lines[0])
    if match:
        lines.pop(0)
        exit_line = f"exit {match.group(1)}"

    kept = [line for line in lines if _KEYWORD.search(line)][:2]
    if kept:
        rule = "keyword"
    elif lines:
        kept = [lines[-1]]
        rule = "last_line"
    else:
        rule = "exit_only"

    parts = ([exit_line] if exit_line else []) + [_normalize_line(line) for line in kept]
    return " | ".join(parts)[:_NORMALIZED_CAP], rule


def host_class_from_cwd(cwd: str | None) -> str:
    """Return ``'darwin'``, ``'linux'``, ``'windows'`` or ``'unknown'`` from a cwd."""
    if not cwd:
        return "unknown"
    if cwd.startswith("/Users/"):
        return "darwin"
    if cwd.startswith("/"):
        return "linux"
    if _WINDOWS_CWD.match(cwd):
        return "windows"
    return "unknown"


def resolve_cue_project(kb_project: str | None, cwd: str | None) -> tuple[str, str]:
    """Return ``(project, project_source)`` for a failure.

    Prefers an explicit ``.kb_project`` value; otherwise derives a project
    from the cwd basename (stripping a dispatch run-id suffix, lowercasing,
    ``_`` → ``-``); otherwise ``('', 'none')``.
    """
    if isinstance(kb_project, str) and kb_project.strip():
        return kb_project.strip(), "kb_project"
    if cwd:
        if _WINDOWS_CWD.match(cwd):
            base = PureWindowsPath(cwd.rstrip("\\/")).name
        else:
            base = PurePosixPath(cwd.rstrip("/")).name
        base = _RUN_ID_SUFFIX.sub("", base).lower().replace("_", "-")
        if base:
            return base, "cwd_basename"
    return "", "none"


def cue_key(
    tool: str,
    target_class: str,
    normalized_error: str,
    project: str,
    version: int = CUE_NORMALIZER_VERSION,
) -> str:
    """Stable 16-hex key over (version, tool, target_class, normalized_error, project)."""
    joined = "\x1f".join([str(version), tool, target_class, normalized_error, project])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def build_cue(
    tool: str,
    tool_input: object,
    error_text: str,
    kb_project: str | None,
    cwd: str | None,
) -> FailureCue:
    """Compose the normalizer functions into a :class:`FailureCue`.

    ``error_text`` is truncated to 4000 chars first so live (hook-truncated)
    and backfill (full) inputs normalize identically.
    """
    error_text = error_text[:_ERROR_TRUNCATE]
    if not isinstance(tool_input, dict):
        tool_input = {}
    target = extract_target(tool, tool_input)
    tclass = target_class(tool, target)
    normalized, rule = normalize_error(error_text)
    project, project_source = resolve_cue_project(kb_project, cwd)
    return FailureCue(
        tool=tool,
        target=target,
        target_class=tclass,
        normalized_error=normalized,
        error_rule=rule,
        project=project,
        project_source=project_source,
        host_class=host_class_from_cwd(cwd),
        normalizer_version=CUE_NORMALIZER_VERSION,
        cue_key=cue_key(tool, tclass, normalized, project),
    )
