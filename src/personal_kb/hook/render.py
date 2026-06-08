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
    from personal_kb.hook.index_reader import MapEntry

# Em-dash separator after the project_ref. Matches preflight.py's separator
# style; the test asserts the U+2014 codepoint, not a hyphen.
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
