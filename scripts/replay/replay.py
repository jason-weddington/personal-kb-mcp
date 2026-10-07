#!/usr/bin/env python3
"""Replay experiment harness: known mistakes with KB off vs session-start slice vs soft gate.

Five subcommands form a pipeline, each writing into a required ``--out DIR``
that must lie OUTSIDE this repository (real KB content never lands here):

* ``export``   -- read qualifying ``supersedes`` edges into ``pairs.json``.
* ``generate`` -- an LLM selects real corrections and writes tempting tasks.
* ``run``      -- replay every task under each arm in a sandboxed scratch dir.
* ``judge``    -- an arm-blind LLM judge scores each run.
* ``report``   -- paired McNemar counts and the pinned decision rule.

Stdlib only. ``claude`` and ``psql`` are invoked as subprocesses through the
``run_claude`` / ``run_psql`` seams; ``kb_core`` and ``personal_kb`` are never
imported. See ``scripts/replay/README.md``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = Path(__file__).resolve().parent / "hooks"
PRETOOL_GATE = HOOKS_DIR / "pretool_gate.py"
SESSION_SLICE = HOOKS_DIR / "session_slice.py"

# --------------------------------------------------------------------------- #
# Pinned constants
# --------------------------------------------------------------------------- #

EXPORT_SQL = (
    "select e.source as new_id, e.target as old_id, o.project_ref as project_ref,"
    " o.entry_type as old_type, o.short_title as old_short_title,"
    " o.long_title as old_long_title, o.knowledge_details as old_details,"
    " n.entry_type as new_type, n.short_title as new_short_title,"
    " n.long_title as new_long_title, n.knowledge_details as new_details,"
    " e.properties as edge_properties, n.is_active as new_is_active"
    " from graph_edges e join knowledge_entries o on o.id = e.target"
    " join knowledge_entries n on n.id = e.source"
    " where e.edge_type = 'supersedes' order by e.source, e.target"
)

PAIR_KEYS = (
    "new_id",
    "old_id",
    "project_ref",
    "old_type",
    "old_short_title",
    "old_long_title",
    "old_details",
    "new_type",
    "new_short_title",
    "new_long_title",
    "new_details",
)

GEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "is_correction",
        "already_in_steering",
        "wrong_belief",
        "correction",
        "task_prompt",
        "files",
        "gate",
        "success_criterion",
    ],
    "properties": {
        "is_correction": {"type": "boolean"},
        "already_in_steering": {"type": "boolean"},
        "wrong_belief": {"type": "string"},
        "correction": {"type": "string"},
        "task_prompt": {"type": "string"},
        "files": {"type": "object", "additionalProperties": {"type": "string"}},
        "gate": {
            "type": ["object", "null"],
            "required": ["tool", "target_regex"],
            "properties": {
                "tool": {"enum": ["Bash", "Read", "Edit", "Write"]},
                "target_regex": {"type": "string"},
            },
        },
        "success_criterion": {"type": "string"},
    },
}

CONTROLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["controls"],
    "properties": {
        "controls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task_prompt", "files", "success_criterion", "project_ref"],
                "properties": {
                    "task_prompt": {"type": "string"},
                    "files": {"type": "object", "additionalProperties": {"type": "string"}},
                    "success_criterion": {"type": "string"},
                    "project_ref": {"type": "string"},
                },
            },
        }
    },
}

JUDGE_SCHEMA_CORRECTION: dict[str, Any] = {
    "type": "object",
    "required": ["repeated", "evidence"],
    "properties": {"repeated": {"type": "boolean"}, "evidence": {"type": "string"}},
}

JUDGE_SCHEMA_CONTROL: dict[str, Any] = {
    "type": "object",
    "required": ["passed", "evidence"],
    "properties": {"passed": {"type": "boolean"}, "evidence": {"type": "string"}},
}

ARMS = ("kb_off", "slice", "soft_gate")
ARM_HOOKS: dict[str, set[str]] = {
    "kb_off": {"PreToolUse"},
    "soft_gate": {"PreToolUse"},
    "slice": {"PreToolUse", "SessionStart"},
}

SLICE_HEADER = "Known corrections from the knowledge base (each replaces an earlier wrong belief):"
SLICE_MAX = 20
SANDBOX_BASH = (
    "This environment does not execute shell commands. "
    "The command was recorded; continue without its output."
)
SANDBOX_PATH = "Path outside the task directory."
SANDBOX_TOOL = "Tool not available."
GATE_PREFIX = "KB correction: "

CHILD_ENV_STRIP = (
    "ANTHROPIC_API_KEY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "HEADLESS_BUILD_ENGINE",
    "PERSONAL_KB_URL",
    "PERSONAL_KB_API_KEY",
)

INVALID_REASONS = (
    "mcp_servers_present",
    "hook_leak",
    "slice_not_delivered",
    "hook_error",
    "unhooked_tool_call",
    "gate_in_wrong_arm",
    "slice_in_wrong_arm",
)
ERROR_REASONS = ("timeout", "nonzero_exit", "no_result_event", "bad_files")

GEN_RULES = (
    "R1 is_correction is true ONLY when the old entry asserts something the new entry shows"
    " to be wrong; a session-log continuation, a design evolution or a topic replacement is"
    " false.",
    "R2 task_prompt must not state the wrong belief or the correction, and must be completable"
    " in an empty scratch directory with no network.",
    "R3 Shell commands are recorded but never executed and return no output; the task must be"
    " completable using only Read, Write, Edit, Glob and Grep, and success_criterion must be"
    " checkable from the agent's file edits and final answer.",
    "R4 already_in_steering is true when the correction is already stated or clearly implied"
    " in the STEERING text.",
)

JUDGE_DIGEST_MAX = 12000
JUDGE_DIGEST_HALF = 6000
CALL_TIMEOUT = 300
REFUSAL = "refusing to write replay data inside the repo"
CANARY_PROMPT = "Run the shell command: touch CANARY"
_MCP_EMPTY = '{"mcpServers":{}}'


def GEN_ARGV(model: str, schema: dict[str, Any]) -> list[str]:  # noqa: N802 - pinned name
    """Return the pinned argv for a generate call.

    Args:
        model: Claude model alias.
        schema: JSON schema for ``structured_output``.

    Returns:
        The argv list.
    """
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--mcp-config",
        _MCP_EMPTY,
        "--tools",
        "",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--max-budget-usd",
        "1.00",
        "--json-schema",
        json.dumps(schema),
    ]


def RUN_ARGV(model: str) -> list[str]:  # noqa: N802 - pinned name
    """Return the pinned argv for an agent run.

    Args:
        model: Claude model alias.

    Returns:
        The argv list.
    """
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--mcp-config",
        _MCP_EMPTY,
        "--tools",
        "Bash,Read,Write,Edit,Glob,Grep",
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-hook-events",
        "--no-session-persistence",
        "--max-turns",
        "15",
        "--max-budget-usd",
        "0.50",
    ]


def JUDGE_ARGV(model: str, schema: dict[str, Any]) -> list[str]:  # noqa: N802 - pinned name
    """Return the pinned argv for a judge call.

    Args:
        model: Claude model alias.
        schema: JSON schema for ``structured_output``.

    Returns:
        The argv list.
    """
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--mcp-config",
        _MCP_EMPTY,
        "--tools",
        "",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--max-budget-usd",
        "0.50",
        "--json-schema",
        json.dumps(schema),
    ]


# --------------------------------------------------------------------------- #
# Subprocess seams
# --------------------------------------------------------------------------- #


def run_claude(
    argv: list[str], *, stdin: str, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Invoke the ``claude`` CLI. The ONE seam every claude call goes through.

    Args:
        argv: Full argv, starting with ``claude``.
        stdin: Text fed on stdin (the prompt).
        cwd: Working directory.
        env: Child environment.
        timeout: Seconds before ``subprocess.TimeoutExpired``.

    Returns:
        The completed process with text stdout/stderr.
    """
    return subprocess.run(  # noqa: S603 - argv is built from pinned literals
        argv,
        input=stdin,
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def run_psql(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke ``psql``. The seam the ``export --dsn`` path goes through.

    Args:
        argv: Full argv, starting with ``psql``.

    Returns:
        The completed process with text stdout/stderr.
    """
    return subprocess.run(  # noqa: S603 - argv is built from a pinned literal
        argv, capture_output=True, text=True, check=False
    )


def _git_sha() -> str | None:
    """Return the repo HEAD sha, or None."""
    try:
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    """Return the current UTC timestamp at second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def child_env(use_api_key: bool = False) -> dict[str, str]:
    """Return the child environment with the isolation keys stripped.

    Args:
        use_api_key: Keep ``ANTHROPIC_API_KEY`` (API billing) when True.

    Returns:
        A copy of ``os.environ`` minus ``CHILD_ENV_STRIP``.
    """
    env = dict(os.environ)
    for key in CHILD_ENV_STRIP:
        if use_api_key and key == "ANTHROPIC_API_KEY":
            continue
        env.pop(key, None)
    return env


def redact_dsn(dsn: str) -> str:
    """Replace any password component of a DSN with ``***``.

    Args:
        dsn: A libpq URI or key/value connection string.

    Returns:
        The DSN with its password masked.
    """
    out = re.sub(r"(://[^:/@?#]*:)[^@/?#]*@", r"\1***@", dsn)
    return re.sub(r"(password\s*=\s*)('[^']*'|\S+)", r"\1***", out, flags=re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank or unparseable lines."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _write_json(path: Path, data: Any) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def merge_manifest(out: Path, fields: dict[str, Any]) -> None:
    """Merge *fields* into ``out/manifest.json``, keeping existing keys.

    Args:
        out: Output directory.
        fields: Keys to set.
    """
    path = out / "manifest.json"
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except ValueError:
            manifest = {}
    manifest.update(fields)
    _write_json(path, manifest)


def _claude_version(out: Path) -> str | None:
    """Return ``claude --version`` output via the seam, or None."""
    try:
        proc = run_claude(["claude", "--version"], stdin="", cwd=out, env=child_env(), timeout=30)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _record_phase(out: Path, phase: str, args: argparse.Namespace) -> None:
    """Merge the per-phase manifest fields."""
    recorded = {k: v for k, v in vars(args).items() if k != "func"}
    if isinstance(recorded.get("dsn"), str):
        recorded["dsn"] = redact_dsn(recorded["dsn"])
    if isinstance(recorded.get("out"), Path):
        recorded["out"] = str(recorded["out"])
    merge_manifest(
        out,
        {
            f"{phase}_started_ts": _now(),
            f"{phase}_args": json.loads(json.dumps(recorded, default=str)),
            "claude_version": _claude_version(out),
            "git_sha": _git_sha(),
        },
    )


def ledger_total(out: Path) -> float:
    """Return the summed ``cost_usd`` of the shared cost ledger.

    Args:
        out: Output directory.

    Returns:
        Total spend in USD.
    """
    total = 0.0
    for row in _read_jsonl(out / "cost-ledger.jsonl"):
        try:
            total += float(row.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _ledger_append(
    out: Path,
    *,
    phase: str,
    model: str,
    cost_usd: float,
    num_turns: Any,
    duration_ms: int,
    api_key_source: Any = None,
    task_id: str | None = None,
    arm: str | None = None,
    rep: int | None = None,
) -> None:
    """Append one call to the shared cost ledger."""
    _append_jsonl(
        out / "cost-ledger.jsonl",
        {
            "ts": _now(),
            "phase": phase,
            "task_id": task_id,
            "arm": arm,
            "rep": rep,
            "model": model,
            "cost_usd": cost_usd,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "api_key_source": api_key_source,
        },
    )


def _float_or_zero(value: Any) -> float:
    """Coerce a cost to float, 0.0 when absent or malformed."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def bad_file_key(key: str) -> bool:
    """Return True for an unsafe task-file key.

    Args:
        key: Relative path from a task's ``files`` mapping.

    Returns:
        True when absolute, containing ``..``, empty, or under ``.claude*``.
    """
    if not isinstance(key, str) or not key.strip():
        return True
    path = PurePosixPath(key)
    if path.is_absolute() or key.startswith("\\") or ".." in path.parts:
        return True
    return bool(path.parts) and path.parts[0].startswith(".claude")


def _files_ok(files: Any) -> bool:
    """Return True when *files* is a str->str mapping with safe keys."""
    if not isinstance(files, dict):
        return False
    return all(not bad_file_key(k) and isinstance(v, str) for k, v in files.items())


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def _is_llm_edge(properties: object) -> bool:
    """Mirror of kb_core.supersession._is_llm_edge (re-implemented, not imported)."""
    if not properties:
        return False
    try:
        parsed = json.loads(properties) if isinstance(properties, str) else properties
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("source") == "llm"


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the KB's qualifying-superseder rule to exported edge rows.

    Args:
        rows: Rows shaped by ``EXPORT_SQL``.

    Returns:
        ``(pairs, dropped)``: pairs restricted to ``PAIR_KEYS`` and drop counts.
    """
    dropped = {"llm_edge": 0, "inactive_superseder": 0, "mental_map": 0}
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if _is_llm_edge(row.get("edge_properties")):
            dropped["llm_edge"] += 1
            continue
        try:
            active = int(row.get("new_is_active"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            active = 0
        if active != 1:
            dropped["inactive_superseder"] += 1
            continue
        if row.get("new_type") == "mental_map":
            dropped["mental_map"] += 1
            continue
        pairs.append({key: row.get(key) for key in PAIR_KEYS})
    return pairs, dropped


def cmd_export(args: argparse.Namespace) -> int:
    """Export qualifying supersedes pairs to ``pairs.json``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    out: Path = args.out
    if bool(args.dsn) == bool(args.sqlite):
        print("export: give exactly one of --dsn or --sqlite", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]]
    if args.dsn:
        argv = [
            "psql",
            "-X",
            "-At",
            "-d",
            args.dsn,
            "-c",
            f"select coalesce(json_agg(t), '[]') from ({EXPORT_SQL}) t",  # noqa: S608 - pinned constant SQL
        ]
        try:
            proc = run_psql(argv)
        except FileNotFoundError as exc:
            print(f"export: psql not found: {exc}", file=sys.stderr)
            return 1
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return 1
        try:
            loaded = json.loads(proc.stdout.strip() or "[]")
        except ValueError as exc:
            print(f"export: unparseable psql output: {exc}", file=sys.stderr)
            return 1
        rows = [r for r in loaded if isinstance(r, dict)]
        source, target = "dsn", redact_dsn(args.dsn)
    else:
        conn = sqlite3.connect(args.sqlite)
        try:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(EXPORT_SQL)]
        finally:
            conn.close()
        source, target = "sqlite", str(args.sqlite)
    pairs, dropped = filter_rows(rows)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "pairs.json", pairs)
    _record_phase(out, "export", args)
    merge_manifest(
        out,
        {
            "export_source": source,
            "export_target": target,
            "pairs_exported": len(pairs),
            "export_dropped": dropped,
            "exported_ts": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    print(f"exported {len(pairs)} pairs (dropped {dropped})", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def default_steering_paths() -> list[Path]:
    """Return the default steering files (user CLAUDE.md plus rules), symlinks resolved.

    Returns:
        Candidate paths; missing files are skipped at read time.
    """
    home = Path("~/.claude").expanduser()
    paths = [home / "CLAUDE.md"]
    paths += [Path(p) for p in sorted(glob.glob(str(home / "rules" / "*.md")))]
    return [p.resolve() for p in paths]


def read_steering(paths: list[Path]) -> str:
    """Read and concatenate steering files once, skipping missing ones.

    Args:
        paths: Steering file paths.

    Returns:
        The files joined with blank lines.
    """
    texts: list[str] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved.is_file():
            texts.append(resolved.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(texts)


def build_gen_prompt(pair: dict[str, Any], steering: str) -> str:
    """Build the per-pair generate prompt.

    Args:
        pair: One ``pairs.json`` object.
        steering: Concatenated STEERING text.

    Returns:
        The prompt text.
    """
    sections = [
        "You are building a replay experiment. Below are an OLD knowledge-base entry and the"
        " NEW entry that superseded it. Decide whether the new entry CORRECTS the old one and,"
        " if so, write a realistic task in which an agent holding the old (wrong) belief would"
        " be tempted to act on it.",
        "Rules:",
        *GEN_RULES,
        "gate is non-null only when acting on the wrong belief shows up as a specific tool call;"
        " then gate.tool is that tool and gate.target_regex is a Python regex matched against"
        " the call's command (Bash) or file_path (Read/Edit/Write).",
        "files maps relative paths to the contents of files to create in the scratch directory.",
    ]
    for key in (
        "project_ref",
        "old_short_title",
        "old_long_title",
        "old_details",
        "new_short_title",
        "new_long_title",
        "new_details",
    ):
        sections.append(f"## {key}\n{pair.get(key) or ''}")
    sections.append(f"## STEERING\n{steering}")
    return "\n\n".join(sections)


def build_controls_prompt(kept: list[dict[str, Any]], n: int) -> str:
    """Build the single controls-generation prompt.

    Args:
        kept: Kept correction tasks.
        n: Number of controls requested.

    Returns:
        The prompt text.
    """
    refs = sorted({str(t.get("project_ref") or "") for t in kept})
    beliefs = "\n".join(f"- {t.get('wrong_belief')}" for t in kept) or "- (none)"
    return "\n\n".join(
        [
            f"Write {n} control tasks for a replay experiment: realistic tasks in the same"
            " projects to which NONE of the stored corrections below applies.",
            GEN_RULES[2],
            "files maps relative paths to the contents of files to create in the scratch"
            " directory; each control needs a project_ref.",
            "## project_refs\n" + ("\n".join(refs) or "(none)"),
            "## wrong_beliefs\n" + beliefs,
        ]
    )


def _call_structured(
    out: Path, argv: list[str], prompt: str, model: str, phase: str
) -> tuple[dict[str, Any] | None, float, int]:
    """Run a structured-output claude call in a fresh empty tempdir.

    Returns:
        ``(structured_output | None, cost_usd, duration_ms)``. ``None`` means a
        call error; the ledger is written whenever the call ran.
    """
    tmp = tempfile.mkdtemp(prefix="kb-replay-gen-")
    start = time.monotonic()
    try:
        proc = run_claude(argv, stdin=prompt, cwd=Path(tmp), env=child_env(), timeout=CALL_TIMEOUT)
    except Exception:
        return None, 0.0, int((time.monotonic() - start) * 1000)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    duration_ms = int((time.monotonic() - start) * 1000)
    parsed: Any = None
    try:
        parsed = json.loads(proc.stdout)
    except (TypeError, ValueError):
        parsed = None
    data = parsed if isinstance(parsed, dict) else {}
    cost = _float_or_zero(data.get("total_cost_usd"))
    _ledger_append(
        out,
        phase=phase,
        model=model,
        cost_usd=cost,
        num_turns=data.get("num_turns"),
        duration_ms=duration_ms,
    )
    structured = data.get("structured_output")
    if proc.returncode != 0 or not isinstance(structured, dict):
        return None, cost, duration_ms
    return structured, cost, duration_ms


def _gen_log_line(
    pair: dict[str, Any] | None,
    *,
    decision: str,
    drop_reason: str | None,
    model: str,
    task_id: str | None = None,
    verdict: dict[str, Any] | None = None,
    leak_check_hit: bool = False,
    cost: float = 0.0,
    duration_ms: int = 0,
) -> dict[str, Any]:
    """Build one generate-log line."""
    verdict = verdict or {}
    return {
        "old_id": pair.get("old_id") if pair else None,
        "new_id": pair.get("new_id") if pair else None,
        "decision": decision,
        "drop_reason": drop_reason,
        "task_id": task_id,
        "is_correction": verdict.get("is_correction"),
        "already_in_steering": verdict.get("already_in_steering"),
        "gate": verdict.get("gate"),
        "leak_check_hit": leak_check_hit,
        "model": model,
        "total_cost_usd": cost,
        "duration_ms": duration_ms,
        "structured_output": verdict or None,
    }


def _classify_verdict(pair: dict[str, Any], verdict: dict[str, Any]) -> tuple[str | None, bool]:
    """Return ``(drop_reason | None, leak_check_hit)`` for a generate verdict."""
    if verdict.get("is_correction") is not True:
        return "not_correction", False
    if verdict.get("already_in_steering") is True:
        return "in_steering", False
    gate = verdict.get("gate")
    if gate is not None:
        if not isinstance(gate, dict) or not isinstance(gate.get("target_regex"), str):
            return "bad_regex", False
        try:
            re.compile(gate["target_regex"])
        except re.error:
            return "bad_regex", False
    title = str(pair.get("new_short_title") or "").strip().lower()
    task_prompt = str(verdict.get("task_prompt") or "")
    if title and title in task_prompt.lower():
        return "leak", True
    if not _files_ok(verdict.get("files")):
        return "bad_files", False
    return None, False


def cmd_generate(args: argparse.Namespace) -> int:
    """Select corrections and write tempting tasks via an LLM.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    _record_phase(out, "generate", args)
    pairs_path = out / "pairs.json"
    if not pairs_path.exists():
        print(f"generate: {pairs_path} not found; run export first", file=sys.stderr)
        return 1
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs = sorted(pairs, key=lambda p: (str(p.get("new_id")), str(p.get("old_id"))))
    steering_paths = [Path(p) for p in args.steering] if args.steering else default_steering_paths()
    steering = read_steering(steering_paths)

    log_path = out / "generate-log.jsonl"
    tasks_path = out / "tasks.jsonl"
    log_path.write_text("", encoding="utf-8")
    model = args.model
    kept: list[dict[str, Any]] = []
    budget_hit = False

    for pair in pairs:
        if len(kept) >= args.limit:
            break
        if budget_hit or ledger_total(out) >= args.budget_usd:
            budget_hit = True
            _append_jsonl(
                log_path, _gen_log_line(pair, decision="dropped", drop_reason="budget", model=model)
            )
            continue
        verdict, cost, duration_ms = _call_structured(
            out, GEN_ARGV(model, GEN_SCHEMA), build_gen_prompt(pair, steering), model, "generate"
        )
        if verdict is None:
            _append_jsonl(
                log_path,
                _gen_log_line(
                    pair,
                    decision="dropped",
                    drop_reason="call_error",
                    model=model,
                    cost=cost,
                    duration_ms=duration_ms,
                ),
            )
            continue
        reason, leak_hit = _classify_verdict(pair, verdict)
        task_id = None if reason else f"c{len(kept) + 1:02d}"
        _append_jsonl(
            log_path,
            _gen_log_line(
                pair,
                decision="dropped" if reason else "kept",
                drop_reason=reason,
                model=model,
                task_id=task_id,
                verdict=verdict,
                leak_check_hit=leak_hit,
                cost=cost,
                duration_ms=duration_ms,
            ),
        )
        if reason is None:
            kept.append(
                {
                    "task_id": task_id,
                    "kind": "correction",
                    "project_ref": pair.get("project_ref"),
                    "old_id": pair.get("old_id"),
                    "new_id": pair.get("new_id"),
                    "wrong_belief": verdict.get("wrong_belief"),
                    "correction": verdict.get("correction"),
                    "task_prompt": verdict.get("task_prompt"),
                    "files": verdict.get("files"),
                    "gate": verdict.get("gate"),
                    "success_criterion": verdict.get("success_criterion"),
                }
            )

    controls: list[dict[str, Any]] = []
    if args.controls > 0:
        controls = _generate_controls(out, args, kept, log_path, budget_hit)

    with tasks_path.open("w", encoding="utf-8") as fh:
        for task in kept + controls:
            fh.write(json.dumps(task) + "\n")
    print(f"kept {len(kept)} correction tasks and {len(controls)} controls", file=sys.stderr)
    return 0


def _generate_controls(
    out: Path,
    args: argparse.Namespace,
    kept: list[dict[str, Any]],
    log_path: Path,
    budget_hit: bool,
) -> list[dict[str, Any]]:
    """Make the single controls call and log it."""
    model = args.model
    base = {"kind": "control", "got": None}
    if budget_hit or ledger_total(out) >= args.budget_usd:
        line = _gen_log_line(None, decision="dropped", drop_reason="budget", model=model)
        _append_jsonl(log_path, {**line, **base})
        return []
    verdict, cost, duration_ms = _call_structured(
        out,
        GEN_ARGV(model, CONTROLS_SCHEMA),
        build_controls_prompt(kept, args.controls),
        model,
        "generate",
    )
    raw = verdict.get("controls") if verdict else None
    if not isinstance(raw, list):
        line = _gen_log_line(
            None,
            decision="dropped",
            drop_reason="call_error",
            model=model,
            cost=cost,
            duration_ms=duration_ms,
        )
        _append_jsonl(log_path, {**line, **base})
        return []
    got = len(raw)
    controls: list[dict[str, Any]] = []
    for item in raw[: args.controls]:
        if not isinstance(item, dict) or not _files_ok(item.get("files")):
            line = _gen_log_line(None, decision="dropped", drop_reason="bad_files", model=model)
            _append_jsonl(log_path, {**line, "kind": "control", "got": got})
            continue
        controls.append(
            {
                "task_id": f"k{len(controls) + 1:02d}",
                "kind": "control",
                "project_ref": item.get("project_ref"),
                "old_id": None,
                "new_id": None,
                "wrong_belief": None,
                "correction": None,
                "task_prompt": item.get("task_prompt"),
                "files": item.get("files"),
                "gate": None,
                "success_criterion": item.get("success_criterion"),
            }
        )
    short = got < args.controls
    line = _gen_log_line(
        None,
        decision="kept" if controls else "dropped",
        drop_reason="short" if short else None,
        model=model,
        cost=cost,
        duration_ms=duration_ms,
    )
    line["structured_output"] = verdict
    _append_jsonl(log_path, {**line, "kind": "control", "got": got})
    return controls


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def build_slice_text(tasks: list[dict[str, Any]]) -> str:
    """Build the SAME slice text used for every task in the slice arm.

    Args:
        tasks: All tasks (controls are ignored).

    Returns:
        The header plus up to ``SLICE_MAX`` corrections sorted by task_id.
    """
    corrections = sorted(
        (t for t in tasks if t.get("kind") == "correction"), key=lambda t: str(t["task_id"])
    )[:SLICE_MAX]
    return (
        SLICE_HEADER
        + "\n"
        + "\n".join(f"- {t['correction']} (replaces: {t['wrong_belief']})" for t in corrections)
    )


def build_hook_config(
    *,
    arm: str,
    task: dict[str, Any],
    rep: int,
    scratch: Path,
    run_dir: Path,
    slice_text: str,
) -> dict[str, Any]:
    """Build the HOOK_CONFIG dict for one run.

    Args:
        arm: Arm name.
        task: The task object.
        rep: Repetition number.
        scratch: Scratch directory.
        run_dir: Run directory (holds the hook log).
        slice_text: The slice text (used only in the slice arm).

    Returns:
        The hook config.
    """
    task_gate = task.get("gate")
    gate = None
    if arm == "soft_gate" and isinstance(task_gate, dict):
        gate = {
            "tool": task_gate.get("tool"),
            "target_regex": task_gate.get("target_regex"),
            "correction": task.get("correction"),
        }
    return {
        "arm": arm,
        "task_id": task["task_id"],
        "rep": rep,
        "scratch_dir": str(Path(scratch).resolve()),
        "hook_log": str((run_dir / "hook-log.jsonl").resolve()),
        "gate": gate,
        "slice_text": slice_text if arm == "slice" else None,
    }


def build_settings(arm: str, config_path: Path) -> dict[str, Any]:
    """Build the scratch ``.claude/settings.json`` dict for an arm.

    Args:
        arm: Arm name.
        config_path: Absolute hook-config path.

    Returns:
        The settings dict.
    """
    cfg = shlex.quote(str(config_path))
    hooks: dict[str, Any] = {
        "PreToolUse": [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"python3 {shlex.quote(str(PRETOOL_GATE))} {cfg}",
                        "timeout": 5,
                    }
                ],
            }
        ]
    }
    if arm == "slice":
        hooks["SessionStart"] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"python3 {shlex.quote(str(SESSION_SLICE))} {cfg}",
                        "timeout": 5,
                    }
                ]
            }
        ]
    return {"hooks": hooks}


def write_task_files(scratch: Path, files: Any) -> bool:
    """Write task files into *scratch*, rejecting any path escaping it.

    Args:
        scratch: Scratch directory.
        files: Mapping of relative path to content.

    Returns:
        False (nothing further written) when any path is unsafe.
    """
    if not isinstance(files, dict):
        return False
    root = scratch.resolve()
    targets: list[tuple[Path, str]] = []
    for key, content in files.items():
        if not isinstance(key, str) or not isinstance(content, str) or bad_file_key(key):
            return False
        target = (root / key).resolve()
        if not target.is_relative_to(root) or target == root:
            return False
        targets.append((target, content))
    for target, content in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return True


def prepare_run(
    *,
    arm: str,
    task: dict[str, Any],
    rep: int,
    scratch: Path,
    run_dir: Path,
    slice_text: str,
) -> Path:
    """Write hook-config.json and scratch/.claude/settings.json.

    Args:
        arm: Arm name.
        task: Task object.
        rep: Repetition number.
        scratch: Scratch directory.
        run_dir: Run directory.
        slice_text: Slice text.

    Returns:
        The hook-config path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = (run_dir / "hook-config.json").resolve()
    config = build_hook_config(
        arm=arm, task=task, rep=rep, scratch=scratch, run_dir=run_dir, slice_text=slice_text
    )
    _write_json(config_path, config)
    _write_json(scratch / ".claude" / "settings.json", build_settings(arm, config_path))
    return config_path


def parse_stream(text: str) -> list[dict[str, Any]]:
    """Parse stream-json stdout into event dicts.

    Args:
        text: Raw stdout.

    Returns:
        Parsed events, skipping non-JSON lines.
    """
    events: list[dict[str, Any]] = []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the content blocks of an assistant/user event."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict)]


def tool_use_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every tool_use block from assistant messages, in stream order.

    Args:
        events: Parsed stream events.

    Returns:
        The tool_use blocks.
    """
    blocks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        blocks += [b for b in _content_blocks(event) if b.get("type") == "tool_use"]
    return blocks


def stream_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract init, hook events and the result event from a stream.

    Args:
        events: Parsed stream events.

    Returns:
        Dict with ``init``, ``hooks`` (hook_event names) and ``result``.
    """
    init: dict[str, Any] | None = None
    hooks: list[Any] = []
    result: dict[str, Any] | None = None
    for event in events:
        if event.get("type") == "system":
            if event.get("subtype") == "init" and init is None:
                init = event
            elif event.get("subtype") == "hook_started":
                hooks.append(event.get("hook_event"))
        elif event.get("type") == "result" and result is None:
            result = event
    return {"init": init or {}, "hooks": hooks, "result": result}


def invalid_reason(
    arm: str, events: list[dict[str, Any]], hook_log: list[dict[str, Any]]
) -> str | None:
    """Return the first INVALID_REASONS hit for a run, or None.

    Args:
        arm: Arm name.
        events: Parsed stream events.
        hook_log: Parsed hook-log lines.

    Returns:
        The reason, or None when the run is isolated correctly.
    """
    summary = stream_summary(events)
    if summary["init"].get("mcp_servers"):
        return "mcp_servers_present"
    hooks = summary["hooks"]
    allowed = ARM_HOOKS[arm]
    if any(h not in allowed for h in hooks):
        return "hook_leak"
    if arm == "slice" and sum(1 for h in hooks if h == "SessionStart") != 1:
        return "hook_leak"
    decisions = [line.get("decision") for line in hook_log]
    n_delivered = decisions.count("slice_delivered")
    if arm == "slice" and n_delivered != 1:
        return "slice_not_delivered"
    if "hook_error" in decisions:
        return "hook_error"
    hooked = {line.get("tool_use_id") for line in hook_log if line.get("tool_use_id")}
    if any(b.get("id") not in hooked for b in tool_use_blocks(events)):
        return "unhooked_tool_call"
    if arm != "soft_gate" and "gate_deny" in decisions:
        return "gate_in_wrong_arm"
    if arm != "slice" and n_delivered:
        return "slice_in_wrong_arm"
    return None


def _empty_result(task: dict[str, Any], arm: str, rep: int) -> dict[str, Any]:
    """Return a result.json skeleton."""
    return {
        "task_id": task["task_id"],
        "arm": arm,
        "rep": rep,
        "status": None,
        "invalid_reason": None,
        "error_reason": None,
        "gate_applicable": task.get("gate") is not None,
        "gate_fired": False,
        "api_key_source": None,
        "claude_version": None,
        "result_subtype": None,
        "num_turns": None,
        "total_cost_usd": None,
        "duration_s": None,
        "n_tool_calls": 0,
        "n_sandbox_deny": 0,
        "n_gate_deny": 0,
        "n_hook_log_lines": 0,
        "final_text_chars": 0,
    }


def _reset_run_dir(run_dir: Path) -> None:
    """Remove per-run artefacts left by a previous attempt."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hook-log.jsonl", "stream.jsonl", "result.json"):
        (run_dir / name).unlink(missing_ok=True)


def execute_run(
    out: Path,
    *,
    task: dict[str, Any],
    arm: str,
    rep: int,
    model: str,
    env: dict[str, str],
    slice_text: str,
) -> dict[str, Any]:
    """Execute one (task, arm, rep) run and write its result.json.

    Args:
        out: Output directory.
        task: Task object.
        arm: Arm name.
        rep: Repetition number.
        model: Claude model alias.
        env: Child environment.
        slice_text: Slice text.

    Returns:
        The result dict.
    """
    run_dir = out / "runs" / str(task["task_id"]) / arm / str(rep)
    _reset_run_dir(run_dir)
    result = _empty_result(task, arm, rep)
    scratch = Path(tempfile.mkdtemp(prefix="kb-replay-"))
    start = time.monotonic()
    proc: subprocess.CompletedProcess[str] | None = None
    try:
        if not write_task_files(scratch, task.get("files")):
            result.update(status="error", error_reason="bad_files")
            _write_json(run_dir / "result.json", result)
            return result
        prepare_run(
            arm=arm, task=task, rep=rep, scratch=scratch, run_dir=run_dir, slice_text=slice_text
        )
        try:
            proc = run_claude(
                RUN_ARGV(model),
                stdin=str(task.get("task_prompt") or ""),
                cwd=scratch,
                env=env,
                timeout=CALL_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            result["error_reason"] = "timeout"
        except OSError:
            result["error_reason"] = "nonzero_exit"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    result["duration_s"] = round(time.monotonic() - start, 3)
    stdout = proc.stdout if proc is not None else ""
    (run_dir / "stream.jsonl").write_text(stdout or "", encoding="utf-8")
    events = parse_stream(stdout)
    summary = stream_summary(events)
    init, res = summary["init"], summary["result"] or {}
    hook_log = _read_jsonl(run_dir / "hook-log.jsonl")
    decisions = [line.get("decision") for line in hook_log]
    _ledger_append(
        out,
        phase="run",
        task_id=str(task["task_id"]),
        arm=arm,
        rep=rep,
        model=model,
        cost_usd=_float_or_zero(res.get("total_cost_usd")),
        num_turns=res.get("num_turns"),
        duration_ms=int(result["duration_s"] * 1000),
        api_key_source=init.get("apiKeySource"),
    )
    final_text = res.get("result")
    result.update(
        gate_fired="gate_deny" in decisions,
        api_key_source=init.get("apiKeySource"),
        claude_version=init.get("claude_code_version"),
        result_subtype=res.get("subtype"),
        num_turns=res.get("num_turns"),
        total_cost_usd=res.get("total_cost_usd"),
        n_tool_calls=len(tool_use_blocks(events)),
        n_sandbox_deny=decisions.count("sandbox_deny"),
        n_gate_deny=decisions.count("gate_deny"),
        n_hook_log_lines=len(hook_log),
        final_text_chars=len(final_text) if isinstance(final_text, str) else 0,
    )
    if result["error_reason"] is None:
        reason = invalid_reason(arm, events, hook_log)
        if reason is not None:
            result.update(status="invalid", invalid_reason=reason)
        elif proc is not None and proc.returncode != 0:
            result["error_reason"] = "nonzero_exit"
        elif summary["result"] is None:
            result["error_reason"] = "no_result_event"
    if result["status"] is None:
        result["status"] = "error" if result["error_reason"] else "ok"
    _write_json(run_dir / "result.json", result)
    return result


def run_canary(out: Path, *, model: str, env: dict[str, str]) -> bool:
    """Prove the sandbox denies shell commands before any task runs.

    Args:
        out: Output directory.
        model: Claude model alias.
        env: Child environment.

    Returns:
        True when the canary's touch was denied and CANARY does not exist.
    """
    run_dir = out / "runs" / "_canary"
    _reset_run_dir(run_dir)
    scratch = Path(tempfile.mkdtemp(prefix="kb-replay-"))
    canary_task = {"task_id": "_canary", "gate": None, "correction": None}
    start = time.monotonic()
    proc: subprocess.CompletedProcess[str] | None = None
    try:
        prepare_run(
            arm="kb_off", task=canary_task, rep=0, scratch=scratch, run_dir=run_dir, slice_text=""
        )
        try:
            proc = run_claude(
                RUN_ARGV(model), stdin=CANARY_PROMPT, cwd=scratch, env=env, timeout=CALL_TIMEOUT
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        touched = (scratch / "CANARY").exists()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    stdout = proc.stdout if proc is not None else ""
    (run_dir / "stream.jsonl").write_text(stdout or "", encoding="utf-8")
    summary = stream_summary(parse_stream(stdout))
    res = summary["result"] or {}
    _ledger_append(
        out,
        phase="canary",
        model=model,
        cost_usd=_float_or_zero(res.get("total_cost_usd")),
        num_turns=res.get("num_turns"),
        duration_ms=int((time.monotonic() - start) * 1000),
        api_key_source=summary["init"].get("apiKeySource"),
    )
    hook_log = _read_jsonl(run_dir / "hook-log.jsonl")
    denied = any(line.get("decision") == "sandbox_deny" for line in hook_log)
    return denied and not touched


def _parse_arms(raw: str) -> list[str] | None:
    """Parse ``--arms``; None when any value is not in ARMS."""
    chosen = [a.strip() for a in raw.split(",") if a.strip()]
    if not chosen or any(a not in ARMS for a in chosen):
        return None
    return [a for a in ARMS if a in chosen]


def load_tasks(out: Path) -> list[dict[str, Any]]:
    """Load ``tasks.jsonl``.

    Args:
        out: Output directory.

    Returns:
        Tasks in file order.
    """
    return [t for t in _read_jsonl(out / "tasks.jsonl") if t.get("task_id")]


def _load_result(run_dir: Path) -> dict[str, Any] | None:
    """Load result.json, or None."""
    path = run_dir / "result.json"
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def cmd_run(args: argparse.Namespace) -> int:
    """Replay every task under each selected arm.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    out: Path = args.out
    arms = _parse_arms(args.arms)
    if arms is None:
        print(f"run: --arms values must be in {ARMS}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    _record_phase(out, "run", args)
    tasks = load_tasks(out)
    slice_text = build_slice_text(tasks)
    env = child_env(args.use_api_key)

    pending: list[tuple[int, dict[str, Any], str]] = []
    for rep in range(1, args.reps + 1):
        for task in tasks:
            for arm in arms:
                prior = _load_result(out / "runs" / str(task["task_id"]) / arm / str(rep))
                if prior and prior.get("status") in ("ok", "invalid"):
                    continue
                pending.append((rep, task, arm))
    if not pending:
        print("run: nothing to do", file=sys.stderr)
        return 0

    budget_hit = ledger_total(out) >= args.budget_usd
    if not budget_hit and not run_canary(out, model=args.model, env=env):
        print("sandbox canary failed", file=sys.stderr)
        return 3

    for rep, task, arm in pending:
        if budget_hit or ledger_total(out) >= args.budget_usd:
            budget_hit = True
            run_dir = out / "runs" / str(task["task_id"]) / arm / str(rep)
            run_dir.mkdir(parents=True, exist_ok=True)
            skipped = _empty_result(task, arm, rep)
            skipped["status"] = "skipped_budget"
            _write_json(run_dir / "result.json", skipped)
            continue
        result = execute_run(
            out, task=task, arm=arm, rep=rep, model=args.model, env=env, slice_text=slice_text
        )
        print(
            f"{task['task_id']} {arm} rep{rep}: {result['status']}"
            f" {result['invalid_reason'] or result['error_reason'] or ''}",
            file=sys.stderr,
        )
    return 0


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #

_REDACT = (SLICE_HEADER, "Known corrections from the knowledge base", "KB correction:")
BLINDING_BLACKLIST = ("kb_off", "soft_gate", SLICE_HEADER, "KB correction:", SANDBOX_BASH)


def redact(text: str) -> str:
    """Case-insensitively replace arm-revealing phrases with ``[redacted]``.

    Args:
        text: Text to redact.

    Returns:
        The redacted text.
    """
    for phrase in _REDACT:
        text = re.sub(re.escape(phrase), "[redacted]", text, flags=re.IGNORECASE)
    return text


def build_digest(
    events: list[dict[str, Any]], hook_log: list[dict[str, Any]]
) -> tuple[str, int, bool, int]:
    """Build the judge DIGEST of a run.

    Args:
        events: Parsed stream events.
        hook_log: Parsed hook-log lines.

    Returns:
        ``(digest, untruncated_chars, truncated, excluded_gate_calls)``.
    """
    denied = {
        line.get("tool_use_id")
        for line in hook_log
        if line.get("decision") == "gate_deny" and line.get("tool_use_id")
    }
    parts: list[str] = []
    excluded = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in _content_blocks(event):
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif kind == "tool_use":
                if block.get("id") in denied:
                    excluded += 1
                    continue
                parts.append(f"[{block.get('name')}] {json.dumps(block.get('input'))}")
    result = stream_summary(events)["result"] or {}
    if isinstance(result.get("result"), str):
        parts.append(result["result"])
    digest = redact("\n".join(parts))
    full_chars = len(digest)
    truncated = full_chars > JUDGE_DIGEST_MAX
    if truncated:
        digest = digest[:JUDGE_DIGEST_HALF] + "\n...\n" + digest[-JUDGE_DIGEST_HALF:]
    return digest, full_chars, truncated, excluded


def build_judge_prompt(task: dict[str, Any], digest: str) -> str:
    """Build the arm-blind judge prompt.

    Args:
        task: Task object.
        digest: Run digest.

    Returns:
        The prompt text.
    """
    if task.get("kind") == "correction":
        return "\n\n".join(
            [
                "You are judging a transcript of an AI coding agent. A previously held belief"
                " is known to be WRONG. Decide whether the agent's effective actions or final"
                " answer acted on the wrong belief (repeated=true) or not (repeated=false)."
                " Tool calls that were blocked before running are not shown and do not count."
                " Quote the deciding evidence.",
                f"## wrong_belief\n{task.get('wrong_belief')}",
                f"## correction\n{task.get('correction')}",
                f"## task_prompt\n{task.get('task_prompt')}",
                f"## transcript digest\n{digest}",
            ]
        )
    return "\n\n".join(
        [
            "You are judging a transcript of an AI coding agent. Decide whether the agent met"
            " the success criterion (passed=true) or not (passed=false). Quote the deciding"
            " evidence.",
            f"## success_criterion\n{task.get('success_criterion')}",
            f"## task_prompt\n{task.get('task_prompt')}",
            f"## transcript digest\n{digest}",
        ]
    )


def blinding_violation(prompt: str) -> bool:
    """Return True when the judge prompt could reveal the arm.

    Args:
        prompt: The exact judge stdin.

    Returns:
        True on any case-insensitive blacklist hit.
    """
    lowered = prompt.lower()
    return any(term.lower() in lowered for term in BLINDING_BLACKLIST)


def _iter_results(out: Path, tasks: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    """Return every result.json for known tasks, in tasks/rep/arm order."""
    found: list[dict[str, Any]] = []
    for task in tasks:
        base = out / "runs" / str(task["task_id"])
        for arm in arms:
            arm_dir = base / arm
            if not arm_dir.is_dir():
                continue
            reps = sorted(
                (p for p in arm_dir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
            )
            for rep_dir in reps:
                result = _load_result(rep_dir)
                if result is not None:
                    found.append(result)
    return found


def cmd_judge(args: argparse.Namespace) -> int:
    """Judge every ok run, blind to the arm.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    out: Path = args.out
    arms = _parse_arms(args.arms)
    if arms is None:
        print(f"judge: --arms values must be in {ARMS}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    _record_phase(out, "judge", args)
    tasks = load_tasks(out)
    by_id = {str(t["task_id"]): t for t in tasks}
    judgments_path = out / "judgments.jsonl"
    done = {
        (j.get("task_id"), j.get("arm"), j.get("rep"))
        for j in _read_jsonl(judgments_path)
        if j.get("judge_error") is None
    }
    model = args.model
    budget_hit = False
    for result in _iter_results(out, tasks, arms):
        if result.get("status") != "ok":
            continue
        key = (result["task_id"], result["arm"], result["rep"])
        if key in done:
            continue
        task = by_id[str(result["task_id"])]
        kind = task.get("kind")
        verdict_key = "repeated" if kind == "correction" else "passed"
        run_dir = out / "runs" / str(result["task_id"]) / str(result["arm"]) / str(result["rep"])
        stream = (
            (run_dir / "stream.jsonl").read_text(encoding="utf-8")
            if (run_dir / "stream.jsonl").exists()
            else ""
        )
        digest, full_chars, truncated, excluded = build_digest(
            parse_stream(stream), _read_jsonl(run_dir / "hook-log.jsonl")
        )
        prompt = build_judge_prompt(task, digest)
        (run_dir / "judge_prompt.txt").write_text(prompt, encoding="utf-8")
        line: dict[str, Any] = {
            "task_id": result["task_id"],
            "arm": result["arm"],
            "rep": result["rep"],
            "kind": kind,
            verdict_key: None,
            "evidence": None,
            "judge_error": None,
            "judge_model": model,
            "digest_chars": full_chars,
            "digest_truncated": truncated,
            "excluded_gate_calls": excluded,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        if blinding_violation(prompt):
            line["judge_error"] = "blinding_violation"
            _append_jsonl(judgments_path, line)
            continue
        if budget_hit or ledger_total(out) >= args.budget_usd:
            budget_hit = True
            line["judge_error"] = "budget"
            _append_jsonl(judgments_path, line)
            continue
        schema = JUDGE_SCHEMA_CORRECTION if kind == "correction" else JUDGE_SCHEMA_CONTROL
        line.update(_judge_call(out, run_dir, JUDGE_ARGV(model, schema), prompt, model, result))
        verdict = line.pop("_verdict", None)
        if verdict is None or not isinstance(verdict.get(verdict_key), bool):
            line["judge_error"] = "call_error"
        else:
            line[verdict_key] = verdict[verdict_key]
            line["evidence"] = verdict.get("evidence")
        _append_jsonl(judgments_path, line)
    return 0


def _judge_call(
    out: Path,
    run_dir: Path,
    argv: list[str],
    prompt: str,
    model: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Make one judge call; return cost/duration and the parsed verdict."""
    tmp = tempfile.mkdtemp(prefix="kb-replay-judge-")
    start = time.monotonic()
    proc: subprocess.CompletedProcess[str] | None = None
    try:
        proc = run_claude(argv, stdin=prompt, cwd=Path(tmp), env=child_env(), timeout=CALL_TIMEOUT)
    except Exception:
        proc = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    duration_ms = int((time.monotonic() - start) * 1000)
    if proc is None:
        return {"_verdict": None, "duration_ms": duration_ms}
    (run_dir / "judge_raw.json").write_text(proc.stdout or "", encoding="utf-8")
    try:
        parsed = json.loads(proc.stdout)
    except (TypeError, ValueError):
        parsed = None
    data = parsed if isinstance(parsed, dict) else {}
    cost = _float_or_zero(data.get("total_cost_usd"))
    _ledger_append(
        out,
        phase="judge",
        task_id=str(result["task_id"]),
        arm=str(result["arm"]),
        rep=result["rep"],
        model=model,
        cost_usd=cost,
        num_turns=data.get("num_turns"),
        duration_ms=duration_ms,
    )
    structured = data.get("structured_output")
    ok = proc.returncode == 0 and isinstance(structured, dict)
    return {
        "_verdict": structured if ok else None,
        "total_cost_usd": cost,
        "duration_ms": duration_ms,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts.

    Args:
        b: Units where kb_off repeated and the arm did not.
        c: Units where kb_off did not repeat and the arm did.

    Returns:
        The p-value, capped at 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def _rate(num: int, den: int) -> float | None:
    """Return ``round(num / den, 4)`` or None for an empty denominator."""
    return round(num / den, 4) if den else None


def _arm_row(
    label: str,
    arm: str,
    judged: dict[tuple[str, str, int], dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    restrict_gate: bool,
) -> dict[str, Any]:
    """Compute one arms-table row."""

    def corr_units(a: str) -> dict[tuple[str, int], bool]:
        units: dict[tuple[str, int], bool] = {}
        for (tid, arm_name, rep), j in judged.items():
            task = tasks.get(tid, {})
            if arm_name != a or task.get("kind") != "correction":
                continue
            if restrict_gate and task.get("gate") is None:
                continue
            units[(tid, rep)] = bool(j.get("repeated"))
        return units

    def controls(a: str) -> tuple[int, int]:
        rows = [
            j
            for (tid, arm_name, _), j in judged.items()
            if arm_name == a and tasks.get(tid, {}).get("kind") == "control"
        ]
        return len(rows), sum(1 for j in rows if j.get("passed") is True)

    ctrl_n, ctrl_pass = controls(arm)
    row: dict[str, Any] = {
        "arm": label,
        "n_paired": 0,
        "repeats": 0,
        "repeat_rate": None,
        "delta_pts": None,
        "b": None,
        "c": None,
        "p_value": None,
        "control_n": ctrl_n,
        "control_pass_rate": _rate(ctrl_pass, ctrl_n),
        "control_drop_pts": None,
        "verdict": "baseline",
    }
    base_units = corr_units("kb_off")
    if arm == "kb_off":
        row["n_paired"] = len(base_units)
        row["repeats"] = sum(base_units.values())
        row["repeat_rate"] = _rate(row["repeats"], row["n_paired"])
        return row
    arm_units = corr_units(arm)
    paired = sorted(set(base_units) & set(arm_units))
    n = len(paired)
    base_rep = sum(1 for u in paired if base_units[u])
    arm_rep = sum(1 for u in paired if arm_units[u])
    b = sum(1 for u in paired if base_units[u] and not arm_units[u])
    c = sum(1 for u in paired if not base_units[u] and arm_units[u])
    row.update(n_paired=n, repeats=arm_rep, repeat_rate=_rate(arm_rep, n), b=b, c=c)
    if n:
        row["delta_pts"] = round((base_rep / n - arm_rep / n) * 100, 1)
        row["p_value"] = round(mcnemar_exact(b, c), 6)
    base_ctrl_n, base_ctrl_pass = controls("kb_off")
    if base_ctrl_n and ctrl_n:
        row["control_drop_pts"] = round(
            (base_ctrl_pass / base_ctrl_n - ctrl_pass / ctrl_n) * 100, 1
        )
    if n == 0 or base_ctrl_n == 0 or ctrl_n == 0:
        row["verdict"] = "insufficient"
    elif row["control_drop_pts"] is not None and row["control_drop_pts"] > 5.0:
        row["verdict"] = "stop-control"
    elif row["delta_pts"] >= 25.0:
        row["verdict"] = "meets"
    else:
        row["verdict"] = "below"
    return row


def build_report(out: Path) -> dict[str, Any]:
    """Compute report.json from the run artefacts.

    Args:
        out: Output directory.

    Returns:
        The report dict.
    """
    task_list = load_tasks(out)
    tasks = {str(t["task_id"]): t for t in task_list}
    results = _iter_results(out, task_list, list(ARMS))
    status_of = {(r["task_id"], r["arm"], r["rep"]): r for r in results}
    judgment_lines = _read_jsonl(out / "judgments.jsonl")
    judged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for j in judgment_lines:
        key = (j.get("task_id"), j.get("arm"), j.get("rep"))
        res = status_of.get(key)  # type: ignore[arg-type]
        if j.get("judge_error") is None and res is not None and res.get("status") == "ok":
            judged[key] = j  # type: ignore[index]

    rows = [
        _arm_row("kb_off", "kb_off", judged, tasks, False),
        _arm_row("slice", "slice", judged, tasks, False),
        _arm_row("soft_gate", "soft_gate", judged, tasks, False),
        _arm_row("soft_gate_applicable", "soft_gate", judged, tasks, True),
    ]

    applicable = [
        r
        for r in results
        if r["arm"] == "soft_gate" and r.get("gate_applicable") and r.get("status") == "ok"
    ]
    tool_match_no_regex = 0
    repeated_without_fire: list[str] = []
    for r in applicable:
        run_dir = out / "runs" / str(r["task_id"]) / "soft_gate" / str(r["rep"])
        log = _read_jsonl(run_dir / "hook-log.jsonl")
        if not r.get("gate_fired") and any(
            line.get("gate_tool_match") and not line.get("gate_regex_match") for line in log
        ):
            tool_match_no_regex += 1
        j = judged.get((r["task_id"], "soft_gate", r["rep"]))
        if not r.get("gate_fired") and j is not None and j.get("repeated") is True:
            repeated_without_fire.append(str(r["task_id"]))

    run_shape: dict[str, Any] = {}
    for arm in ARMS:
        shaped = [r for r in results if r["arm"] == arm and r.get("result_subtype") is not None]
        subtypes: dict[str, int] = {}
        for r in shaped:
            subtypes[str(r["result_subtype"])] = subtypes.get(str(r["result_subtype"]), 0) + 1
        run_shape[arm] = {
            "mean_n_sandbox_deny": round(
                sum(int(r.get("n_sandbox_deny") or 0) for r in shaped) / len(shaped), 4
            )
            if shaped
            else None,
            "by_result_subtype": subtypes,
            "zero_tool_call_runs": sum(1 for r in shaped if not r.get("n_tool_calls")),
            "n_runs": len(shaped),
        }

    per_task: list[dict[str, Any]] = []
    for r in results:
        task = tasks.get(str(r["task_id"]), {})
        kind = task.get("kind")
        verdict_key = "repeated" if kind == "correction" else "passed"
        j = judged.get((r["task_id"], r["arm"], r["rep"]))
        per_task.append(
            {
                "task_id": r["task_id"],
                "kind": kind,
                "arm": r["arm"],
                "rep": r["rep"],
                verdict_key: j.get(verdict_key) if j else None,
                "gate_fired": r.get("gate_fired"),
                "result_subtype": r.get("result_subtype"),
                "judge_evidence": str(j.get("evidence") or "")[:200] if j else None,
            }
        )

    gen_log = _read_jsonl(out / "generate-log.jsonl")
    drops: dict[str, int] = {}
    for line in gen_log:
        if line.get("decision") == "dropped" and line.get("drop_reason"):
            drops[line["drop_reason"]] = drops.get(line["drop_reason"], 0) + 1
    statuses = [r.get("status") for r in results]
    tasks_correction = sum(1 for t in task_list if t.get("kind") == "correction")
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    return {
        "pilot": tasks_correction < 10,
        "manifest": manifest,
        "arms": rows,
        "soft_gate": {
            "applicable_runs": len(applicable),
            "gate_fired_runs": sum(1 for r in applicable if r.get("gate_fired")),
            "tool_match_no_regex_runs": tool_match_no_regex,
            "repeated_without_fire": len(repeated_without_fire),
            "repeated_without_fire_tasks": sorted(set(repeated_without_fire)),
        },
        "run_shape": run_shape,
        "per_task": per_task,
        "counts": {
            "pairs_considered": sum(1 for line in gen_log if line.get("kind") != "control"),
            "tasks_correction": tasks_correction,
            "tasks_control": sum(1 for t in task_list if t.get("kind") == "control"),
            "drops_by_reason": drops,
            "runs_ok": statuses.count("ok"),
            "runs_invalid": statuses.count("invalid"),
            "runs_error": statuses.count("error"),
            "runs_skipped_budget": statuses.count("skipped_budget"),
            "judge_errors": sum(1 for j in judgment_lines if j.get("judge_error") is not None),
            "blinding_violations": sum(
                1 for j in judgment_lines if j.get("judge_error") == "blinding_violation"
            ),
        },
        "cost_usd_total": round(ledger_total(out), 6),
    }


def _cell(value: Any) -> str:
    """Render one markdown table cell."""
    return "" if value is None else str(value)


def render_report_md(report: dict[str, Any]) -> str:
    """Render report.md from the report dict.

    Args:
        report: Output of ``build_report``.

    Returns:
        Markdown text.
    """
    lines: list[str] = []
    if report["pilot"]:
        lines.append("PILOT - underpowered; verdicts are indicative only")
        lines.append("")
    columns = list(report["arms"][0].keys())
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in report["arms"]:
        lines.append("| " + " | ".join(_cell(row[c]) for c in columns) + " |")
    lines.append("")

    cells: dict[tuple[str, str], list[str]] = {}
    for p in report["per_task"]:
        value = p.get("repeated", p.get("passed"))
        label = "repeated" if p.get("kind") == "correction" else "passed"
        mark = "-" if value is None else f"{label}={value}"
        cells.setdefault((str(p["task_id"]), str(p["arm"])), []).append(mark)
    task_ids = sorted({tid for tid, _ in cells})
    lines.append("| task | " + " | ".join(ARMS) + " |")
    lines.append("|---|" + "---|" * len(ARMS))
    for tid in task_ids:
        lines.append(
            f"| {tid} | " + " | ".join(", ".join(cells.get((tid, a), [])) for a in ARMS) + " |"
        )
    lines.append("")

    soft = report["soft_gate"]
    if soft["repeated_without_fire"] > 0:
        lines.append("possible regex misses: " + ", ".join(soft["repeated_without_fire_tasks"]))
    violations = report["counts"]["blinding_violations"]
    if violations > 0:
        lines.append(f"WARN blinding violations: {violations}")
    shape = report["run_shape"]
    base = shape.get("kb_off", {})
    for arm in ARMS[1:]:
        other = shape.get(arm, {})
        if not base.get("n_runs") or not other.get("n_runs"):
            continue
        subtypes = set(base["by_result_subtype"]) | set(other["by_result_subtype"])
        for subtype in sorted(subtypes - {"success"}):
            base_rate = base["by_result_subtype"].get(subtype, 0) / base["n_runs"]
            other_rate = other["by_result_subtype"].get(subtype, 0) / other["n_runs"]
            if abs(base_rate - other_rate) * 100 > 20.0:
                lines.append(f"WARN run-shape skew: {arm} {subtype}")
    return "\n".join(lines).rstrip() + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    """Write report.json and report.md.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    _record_phase(out, "report", args)
    report = build_report(out)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_report_md(report), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def inside_repo(out: str | Path) -> bool:
    """Return True when *out* is REPO_ROOT or lies under it.

    Args:
        out: The ``--out`` value.

    Returns:
        True when writing there must be refused.
    """
    resolved = Path(out).expanduser().resolve()
    return resolved == REPO_ROOT or resolved.is_relative_to(REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog="replay", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="export qualifying supersedes pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--dsn")
    p.add_argument("--sqlite")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("generate", help="LLM-select corrections and write tasks")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--controls", type=int, default=2)
    p.add_argument("--model", default="opus")
    p.add_argument("--budget-usd", type=float, default=20.0)
    p.add_argument("--steering", nargs="+", action="extend", default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("run", help="replay tasks under each arm")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--budget-usd", type=float, default=20.0)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--use-api-key", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("judge", help="arm-blind LLM judging of ok runs")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--budget-usd", type=float, default=20.0)
    p.add_argument("--arms", default=",".join(ARMS))
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="McNemar table and the pinned decision rule")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    if inside_repo(args.out):
        print(REFUSAL, file=sys.stderr)
        return 2
    args.out = Path(args.out).expanduser().resolve()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
