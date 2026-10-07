#!/usr/bin/env python3
"""PreToolUse hook for the replay experiment harness.

Invoked by Claude Code as ``python3 pretool_gate.py <hook-config.json>`` with
the hook payload JSON on stdin. Applies, in order:

1. The deny-once soft gate (``soft_gate`` arm only): when the configured tool
   matches and ``target_regex`` matches the call's target, deny with the stored
   correction. "Once" is derived from the hook log, never from in-process state.
2. The sandbox: Bash is always denied (recorded, never executed), file tools
   are confined to the scratch directory, and every other tool is denied.

Every invocation appends exactly one instrumented line to the run's hook log
BEFORE printing anything. The hook FAILS CLOSED: any exception prints a deny.

Stdlib only. Never imports ``replay.py`` or ``kb_core``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SANDBOX_BASH = (
    "This environment does not execute shell commands. "
    "The command was recorded; continue without its output."
)
SANDBOX_PATH = "Path outside the task directory."
SANDBOX_TOOL = "Tool not available."
GATE_PREFIX = "KB correction: "

# Field map copied from the failure-cue owner contract's ``extract_target``.
_TARGET_FIELDS: dict[str, str] = {
    "Bash": "command",
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
    "Glob": "pattern",
    "Grep": "pattern",
}

_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Glob", "Grep"})
_PATH_KEYS = ("file_path", "notebook_path", "path")


def extract_target(tool_name: object, tool_input: object) -> str:
    """Return the matchable target string of a tool call.

    Args:
        tool_name: The tool's name from the hook payload.
        tool_input: The tool's input object from the hook payload.

    Returns:
        The mapped field's value, or ``''`` for an unmapped tool, a non-dict
        input, or a missing or non-str value.
    """
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        return ""
    key = _TARGET_FIELDS.get(tool_name)
    if key is None:
        return ""
    value = tool_input.get(key)
    return value if isinstance(value, str) else ""


def _deny(reason: str) -> str:
    """Render the PreToolUse deny envelope."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def _read_log(path: Path) -> list[dict[str, Any]]:
    """Read the hook log, skipping blank or unparseable lines."""
    if not path.exists():
        return []
    lines: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            lines.append(parsed)
    return lines


def _append_log(path: Path, line: dict[str, Any]) -> None:
    """Append one JSON line to the hook log."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def _sandbox(
    tool_name: str, tool_input: dict[str, Any], payload: dict[str, Any], scratch: Path
) -> str | None:
    """Return a sandbox deny reason, or None when the call is allowed."""
    if tool_name == "Bash":
        return SANDBOX_BASH
    if tool_name in _FILE_TOOLS:
        raw = None
        for key in _PATH_KEYS:
            if key in tool_input:
                raw = tool_input[key]
                break
        if raw is None:
            return None if tool_name in ("Glob", "Grep") else SANDBOX_PATH
        if not isinstance(raw, str):
            return SANDBOX_PATH
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(payload["cwd"]) / raw
        resolved = candidate.resolve()
        return None if resolved.is_relative_to(scratch) else SANDBOX_PATH
    return SANDBOX_TOOL


def main(argv: list[str]) -> int:
    """Run the hook.

    Args:
        argv: Process argv; ``argv[1]`` is the hook-config path.

    Returns:
        Always 0 (decisions are expressed on stdout).
    """
    tool_name: object = None
    tool_use_id: object = None
    config: dict[str, Any] = {}
    try:
        raw_stdin = sys.stdin.read()
        payload_error: Exception | None = None
        payload: Any = None
        try:
            payload = json.loads(raw_stdin)
        except ValueError as exc:
            payload_error = exc
        if isinstance(payload, dict):
            tool_name = payload.get("tool_name")
            tool_use_id = payload.get("tool_use_id")
        loaded = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("hook config is not an object")
        config = loaded
        if payload_error is not None:
            raise payload_error
        if not isinstance(payload, dict):
            raise ValueError("hook payload is not an object")
        hook_log = Path(config["hook_log"])
        scratch = Path(config["scratch_dir"]).resolve()
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        target = extract_target(tool_name, tool_input)

        gate = config.get("gate")
        gate_tool = gate.get("tool") if isinstance(gate, dict) else None
        gate_regex = gate.get("target_regex") if isinstance(gate, dict) else None
        gate_tool_match = gate is not None and tool_name == gate_tool
        gate_regex_match = bool(
            gate_tool_match and isinstance(gate_regex, str) and re.search(gate_regex, target)
        )

        reason: str | None
        if (
            isinstance(gate, dict)
            and gate_tool_match
            and gate_regex_match
            and not any(line.get("decision") == "gate_deny" for line in _read_log(hook_log))
        ):
            reason = GATE_PREFIX + str(gate["correction"])
            decision = "gate_deny"
        else:
            reason = _sandbox(str(tool_name), tool_input, payload, scratch)
            decision = "allow" if reason is None else "sandbox_deny"

        _append_log(
            hook_log,
            {
                "ts": datetime.now(UTC).isoformat(),
                "task_id": config.get("task_id"),
                "arm": config.get("arm"),
                "rep": config.get("rep"),
                "event": "PreToolUse",
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "target": target[:300],
                "gate_tool": gate_tool,
                "gate_regex": gate_regex,
                "gate_tool_match": gate_tool_match,
                "gate_regex_match": gate_regex_match,
                "decision": decision,
                "error_type": None,
            },
        )
        if reason is not None:
            print(_deny(reason))
        return 0
    except Exception as exc:
        try:
            log_path = config.get("hook_log")
            if log_path:
                _append_log(
                    Path(log_path),
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "task_id": config.get("task_id"),
                        "arm": config.get("arm"),
                        "rep": config.get("rep"),
                        "event": "PreToolUse",
                        "tool_name": tool_name if isinstance(tool_name, str) else None,
                        "tool_use_id": tool_use_id,
                        "target": "",
                        "gate_tool": None,
                        "gate_regex": None,
                        "gate_tool_match": False,
                        "gate_regex_match": False,
                        "decision": "hook_error",
                        "error_type": type(exc).__name__,
                    },
                )
        except Exception:  # noqa: S110 - best-effort logging on the fail-closed path
            pass
        print(_deny(SANDBOX_BASH if tool_name == "Bash" else SANDBOX_TOOL))
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
