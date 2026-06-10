"""Render the factual map directory string and the claude-json envelope.

The injected text is intentionally factual — never imperative. Imperative
phrasing trips prompt-injection defenses and gets surfaced to the user
instead of read by the model. ``BANNED_TOKENS`` is the closed checklist
the unit tests assert against.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personal_kb_hook.index_reader import MapEntry

# Em-dash separator after the project_ref. Matches preflight.py's separator
# style in the main package; the test asserts the U+2014 codepoint, not a hyphen.
_EM_DASH = "—"

# Words that turn the directory into a command. The hook must inject a
# directory, never an instruction.
BANNED_TOKENS: frozenset[str] = frozenset(
    {
        "load",
        "use",
        "read",
        "fetch",
        "pull",
        "open",
        "retrieve",
        "get",
        "review",
        "consult",
    }
)


def render_directory(project_ref: str, maps: list[MapEntry]) -> str:
    """Render the factual directory string for a project's maps.

    Format: ``Maps for <project_ref> — [<id>] <short_title>: <long_title>;
    [<id>] <short_title>: <long_title>``. Entries are joined with ``"; "``
    (semicolon-space). An entry whose ``long_title`` is empty/None renders
    as ``[<id>] <short_title>`` with no trailing ``": "``.
    """
    parts: list[str] = []
    for entry in maps:
        long_title = entry.get("long_title") or ""
        if long_title:
            parts.append(f"[{entry['id']}] {entry['short_title']}: {long_title}")
        else:
            parts.append(f"[{entry['id']}] {entry['short_title']}")
    body = "; ".join(parts)
    return f"Maps for {project_ref} {_EM_DASH} {body}"


def render_cross_directory(current_project: str, index: dict[str, list[MapEntry]]) -> str | None:
    """Render the cross-project roster line (Line 2).

    Format: ``Maps in other domains — {proj}: [{id}] {short_title}, [{id}]
    {short_title}; {proj2}: ...``. SHORT titles only (no long_title). Projects
    are sorted alphabetically; maps appear in index order within a project.
    Maps within a project are joined with ``", "``; projects are joined with
    ``"; "``. The resolved project (``current_project``) is excluded.

    Returns ``None`` when no other project has maps.
    """
    parts: list[str] = []
    for proj in sorted(index.keys()):
        if proj == current_project:
            continue
        proj_maps = index[proj]
        if not proj_maps:
            continue
        map_parts = [f"[{m['id']}] {m['short_title']}" for m in proj_maps]
        parts.append(f"{proj}: {', '.join(map_parts)}")
    if not parts:
        return None
    body = "; ".join(parts)
    return f"Maps in other domains {_EM_DASH} {body}"


def compose_directory(
    project_ref: str,
    maps: list[MapEntry],
    index: dict[str, list[MapEntry]],
) -> str | None:
    """Compose the full injection string: line 1 and/or line 2.

    * Line 1 (``render_directory``) — omitted when ``maps`` is empty.
    * Line 2 (``render_cross_directory``) — omitted when no other project has maps.

    The two lines are joined with a single newline when both are present.
    Returns ``None`` when neither line has any content.
    """
    line1 = render_directory(project_ref, maps) if maps else None
    line2 = render_cross_directory(project_ref, index)
    if line1 and line2:
        return f"{line1}\n{line2}"
    if line1:
        return line1
    if line2:
        return line2
    return None


def render_claude_json(event_name: str, directory: str) -> str:
    """Render the ``--format=claude-json`` envelope.

    Exact shape:
    ``{"hookSpecificOutput": {"hookEventName": <event>, "additionalContext":
    <directory>}}`` — serialized via :func:`json.dumps`.
    """
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": directory,
        }
    }
    return json.dumps(payload)
