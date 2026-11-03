"""Antigravity (agy) adapter: agy payload in, Claude Code payload out (and back).

The ``agy`` CLI (Google's Gemini agent harness) has its own hook contract in
two places, and this module is the whole of it:

* **Input.** agy hands the hook ONE camelCase JSON object on stdin
  (``conversationId``, ``workspacePaths``, ``transcriptPath``, ``lastUserInput``
  and, on tool events, ``toolCall`` / ``stepIdx`` / ``error``). It carries NO
  event name, so the event arrives on the command line (``--event``) and
  :func:`claude_event_for` maps it to the Claude Code event name the pipeline
  already branches on. :func:`to_claude_payload` then renames the common
  fields, translates agy's tool and argument names to the canonical Claude
  Code ones (:data:`TOOL_NAME_MAP` plus the ``call_mcp_tool`` ->
  ``mcp__<ServerName>__<ToolName>`` rule) and synthesizes the ``tool_use_id``
  agy does not supply.
* **Output.** :func:`render_reply` reads the pipeline's captured stdout (a
  Claude Code ``hookSpecificOutput`` envelope, or nothing) and renders agy's
  reply shape: an ``injectSteps`` entry for SessionStart / PreInvocation, an
  always-``allow`` decision for PreToolUse, and ``{}`` everywhere else.

agy's PostToolUse and Stop replies cannot carry context, so the failure
context and the turn digest stay Claude-Code-only; the gate text that Claude
Code would have denied with is delivered as the PreToolUse ``reason`` instead.

Stdlib only (``hashlib``, ``json``) — the hook package has no dependencies.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: The event names agy's hooks configuration can name on the command line.
AGY_EVENTS: frozenset[str] = frozenset(
    {"SessionStart", "PreInvocation", "PreToolUse", "PostToolUse", "Stop"}
)

#: agy event -> the Claude Code event name the pipeline branches on. ``Stop``
#: maps to ``None``: its whole Claude Code body (digest, listener, gate-log
#: flush, cache refresh) is out of scope for agy, so it never enters the
#: pipeline at all.
EVENT_MAP: dict[str, str | None] = {
    "SessionStart": "SessionStart",
    "PreInvocation": "UserPromptSubmit",
    "PreToolUse": "PreToolUse",
    "PostToolUse": "PostToolUse",
    "Stop": None,
}

#: agy tool name -> canonical (Claude Code) tool name. ``call_mcp_tool`` is
#: absent on purpose: its canonical name is computed from the call's
#: ``ServerName`` and ``ToolName`` arguments.
TOOL_NAME_MAP: dict[str, str] = {
    "run_command": "Bash",
    "read_file": "Read",
    "view_file": "Read",
    "write_to_file": "Write",
}

#: agy tool name -> (canonical argument name, agy aliases in priority order).
_ARG_RENAMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "run_command": ("command", ("CommandLine",)),
    "read_file": ("file_path", ("Path",)),
    "view_file": ("file_path", ("Path",)),
    "write_to_file": ("file_path", ("TargetFile", "Path")),
}

_MCP_TOOL = "call_mcp_tool"
_TOOL_EVENTS = frozenset({"PreToolUse", "PostToolUse", "PostToolUseFailure"})


def claude_event_for(event: str | None, payload: dict[str, Any]) -> str | None:
    """Map an agy event name to the Claude Code event name, or ``None``.

    ``payload`` is agy's stdin object; only its TOP-LEVEL ``error`` key is
    consulted, and only for ``PostToolUse`` (a non-empty string means the tool
    failed, so the pipeline takes the ``PostToolUseFailure`` path). An unknown
    or absent event, and ``Stop``, return ``None`` — the caller writes ``{}``
    without entering the pipeline.
    """
    if event not in AGY_EVENTS:
        return None
    if event == "PostToolUse":
        error = payload.get("error")
        if isinstance(error, str) and error:
            return "PostToolUseFailure"
        return "PostToolUse"
    return EVENT_MAP[event]


def _mcp_tool_name(args: dict[str, Any]) -> str | None:
    """Return ``mcp__<ServerName>__<ToolName>`` when both arguments are usable."""
    server = args.get("ServerName")
    tool = args.get("ToolName")
    if isinstance(server, str) and server and isinstance(tool, str) and tool:
        return f"mcp__{server}__{tool}"
    return None


def _rename_args(tool_input: dict[str, Any], target: str, aliases: tuple[str, ...]) -> None:
    """Move the first present alias of ``aliases`` to ``target``, dropping all.

    In place on a dict the caller already copied. The FIRST alias in the
    tuple wins (``TargetFile`` over ``Path``) and every alias is removed, so
    no leftover native key reaches the canonical tool input.
    """
    value: Any = None
    found = False
    for alias in aliases:
        if alias in tool_input:
            candidate = tool_input.pop(alias)
            if not found:
                value = candidate
                found = True
    if found:
        tool_input[target] = value


def _map_tool_call(tool_call: object) -> tuple[str, dict[str, Any]]:
    """Translate agy's ``toolCall`` into ``(canonical name, canonical input)``.

    The name is ``""`` when agy supplied no usable name, so the caller can
    omit ``tool_name`` entirely. The input is ALWAYS a fresh dict — a shallow
    copy of ``args`` (or ``{}`` when ``args`` is missing / not a dict) — and
    the adapter never mutates the payload.
    """
    if not isinstance(tool_call, dict):
        return "", {}
    raw_name = tool_call.get("name")
    name = raw_name if isinstance(raw_name, str) else ""
    raw_args = tool_call.get("args")
    args = raw_args if isinstance(raw_args, dict) else {}
    if name == _MCP_TOOL:
        mcp_name = _mcp_tool_name(args)
        raw_arguments = args.get("Arguments")
        tool_input = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}
        return (mcp_name if mcp_name is not None else name), tool_input
    mapped = TOOL_NAME_MAP.get(name)
    tool_input = dict(args)
    if mapped is None:
        return name, tool_input
    rename = _ARG_RENAMES.get(name)
    if rename is not None:
        _rename_args(tool_input, rename[0], rename[1])
    return mapped, tool_input


def _tool_use_id(step_idx: object, tool_name: str, tool_input: dict[str, Any]) -> str:
    """Synthesize the ``tool_use_id`` agy does not send.

    ``agy-<stepIdx>`` when agy supplied an integer step index (the common
    case), else ``agy-`` plus 12 hex digits of a sha256 over the canonical
    name and the mapped input. The fallback must be derived from the call:
    the server dedupes failure events on ``event_id``
    (``cc:<session>:<tool_use_id>``), so a constant id would silently drop
    every failure after the first in a session.
    """
    if isinstance(step_idx, int) and not isinstance(step_idx, bool):
        return f"agy-{step_idx}"
    blob = tool_name + "\x00" + json.dumps(tool_input, sort_keys=True, default=str)
    return "agy-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def to_claude_payload(payload: dict[str, Any], claude_event: str | None) -> dict[str, Any]:
    """Translate an agy stdin payload into the Claude Code payload shape.

    Returns a NEW dict and never mutates ``payload``. Every mapped key is
    OMITTED (never ``None``) when agy left it unusable, so the downstream
    pipeline's own ``isinstance`` guards keep their meaning. ``hook_event_name``
    (from ``claude_event``) and ``harness`` (``antigravity``) are always
    present; the three tool keys exist only on tool events.
    """
    result: dict[str, Any] = {}
    conversation_id = payload.get("conversationId")
    if isinstance(conversation_id, str) and conversation_id:
        result["session_id"] = conversation_id
    workspace_paths = payload.get("workspacePaths")
    if isinstance(workspace_paths, list) and workspace_paths:
        first = workspace_paths[0]
        if isinstance(first, str) and first:
            result["cwd"] = first
    transcript_path = payload.get("transcriptPath")
    if isinstance(transcript_path, str) and transcript_path:
        result["transcript_path"] = transcript_path
    prompt = payload.get("lastUserInput")
    if isinstance(prompt, str) and prompt:
        result["prompt"] = prompt
    result["hook_event_name"] = claude_event
    result["harness"] = "antigravity"
    if claude_event in _TOOL_EVENTS:
        tool_name, tool_input = _map_tool_call(payload.get("toolCall"))
        if tool_name:
            result["tool_name"] = tool_name
        result["tool_input"] = tool_input
        result["tool_use_id"] = _tool_use_id(payload.get("stepIdx"), tool_name, tool_input)
    if claude_event == "PostToolUseFailure":
        # ``events.post_failure`` and the failure context both read the error
        # string the harness reported; agy puts it at the payload's top level.
        error = payload.get("error")
        if isinstance(error, str) and error:
            result["error"] = error
    return result


def render_reply(claude_event: str | None, captured: str) -> dict[str, Any]:
    """Render agy's reply object from the pipeline's captured stdout.

    ``captured`` is whatever the Claude Code pipeline wrote (an
    ``{"hookSpecificOutput": ...}`` envelope, or nothing). Only
    ``additionalContext`` and ``permissionDecisionReason`` are read — a
    captured ``deny`` is never copied, because agy's first version never
    denies.

    * ``PreToolUse`` always answers, with the gate's reason when the soft gate
      would have denied: ``{"decision": "allow"[ , "reason": <text>]}``.
    * ``SessionStart`` with text -> ``injectSteps[0].systemMessage.systemMessage``.
    * ``UserPromptSubmit`` with text -> ``injectSteps[0].ephemeralMessage``.
    * Everything else (``None``, ``PostToolUse``, ``PostToolUseFailure``,
      ``Stop``) -> ``{}``: agy cannot inject context on those events.
    """
    text: str | None = None
    reason: str | None = None
    obj: object = None
    if captured.strip():
        try:
            obj = json.loads(captured)
        except (json.JSONDecodeError, ValueError):
            obj = None
    if isinstance(obj, dict):
        hook_specific_output = obj.get("hookSpecificOutput")
        if isinstance(hook_specific_output, dict):
            raw_reason = hook_specific_output.get("permissionDecisionReason")
            if isinstance(raw_reason, str) and raw_reason:
                reason = raw_reason
            raw_text = hook_specific_output.get("additionalContext")
            if isinstance(raw_text, str) and raw_text:
                text = raw_text
    if claude_event == "PreToolUse":
        if reason is not None:
            return {"decision": "allow", "reason": reason}
        return {"decision": "allow"}
    if claude_event == "SessionStart" and text is not None:
        return {"injectSteps": [{"systemMessage": {"systemMessage": text}}]}
    if claude_event == "UserPromptSubmit" and text is not None:
        return {"injectSteps": [{"ephemeralMessage": text}]}
    return {}
