"""Render the factual map directory string and the claude-json envelope.

The injected text is intentionally factual — never imperative. Imperative
phrasing trips prompt-injection defenses and gets surfaced to the user
instead of read by the model. ``BANNED_TOKENS`` is the closed checklist
the unit tests assert against.

Multi-KB Line-2 rendering
-------------------------
:func:`render_cross_directory` and :func:`compose_directory` operate over
the post-P1 index shape ``dict[str, list[tuple[str, MapEntry]]]`` — each
list element is a ``(label, MapEntry)`` tuple carrying the source KB
roster label.

* When exactly ONE distinct label is present across the whole index, Line 2
  is byte-identical to the pre-P1 single-KB output:
  ``Maps in other domains — {proj}: [{id}] {short}, ...``.
* When MORE THAN ONE distinct label is present, each cross-project group is
  prefixed with the source label and a forward slash:
  ``Maps in other domains — {label}/{proj}: [{id}] {short}, ...; {label2}/{proj2}: ...``,
  sorted by ``(label, project_ref)``.

Within a single project group, multiple ``(label, MapEntry)`` tuples
appearing under the same ``project_ref`` are bucketed by their label so
each label produces its own ``{label}/{proj}: ...`` group — preserving the
guarantee that every Line-2 cell names exactly one source KB.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from personal_kb_hook.index_reader import MapEntry, MapKey

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

# Display cap for the new-maps delta FYI (render_new_maps). A display limit
# only: every new map is still recorded as surfaced and still gets a
# telemetry row (see personal_kb_hook.cli) — this constant governs only how
# many are NAMED in the emitted text before an "(+N more)" suffix.
NEW_MAPS_DISPLAY_CAP = 5


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


def render_cross_directory(
    current_project: str,
    index: dict[str, list[tuple[str, MapEntry]]],
) -> str | None:
    """Render the cross-project roster line (Line 2).

    The ``index`` value is the post-P1 fan-out shape: each project_ref maps
    to a list of ``(label, MapEntry)`` tuples carrying their source KB label.

    * When exactly ONE distinct label appears in the index, the output is
      byte-identical to the pre-P1 single-KB form:
      ``Maps in other domains — {proj}: [{id}] {short}, ...; {proj2}: ...``
      with projects sorted alphabetically and maps in index order.
    * When MORE THAN ONE distinct label appears, each group is prefixed
      with the source label and a forward slash, and groups are sorted by
      ``(label, project_ref)``:
      ``Maps in other domains — {label}/{proj}: [{id}] {short}, ...; {label2}/{proj2}: ...``.

    Within a single project, if entries arrive under more than one label,
    each label produces its own ``{label}/{proj}: ...`` group so every cell
    names exactly one source KB. SHORT titles only (no long_title). The
    resolved project (``current_project``) is excluded.

    Returns ``None`` when no other project has maps.
    """
    # Collect all distinct labels actually present in the (filtered) index
    # to decide between byte-identical single-KB output and the multi-KB
    # label-prefixed form.
    distinct_labels: set[str] = set()
    for proj, pairs in index.items():
        if proj == current_project:
            continue
        for label, _entry in pairs:
            distinct_labels.add(label)
    if not distinct_labels:
        return None
    multi_label = len(distinct_labels) > 1

    # Build (label, project_ref) -> ordered list[MapEntry] bucketing.
    # In single-label mode we still iterate label-bucketed but the prefix
    # is dropped, and we sort by project_ref (matching pre-P1 ordering).
    grouped: dict[tuple[str, str], list[MapEntry]] = {}
    for proj, pairs in index.items():
        if proj == current_project:
            continue
        for label, entry in pairs:
            grouped.setdefault((label, proj), []).append(entry)

    # Drop any (label, proj) whose entry list is empty — defence-in-depth,
    # _map_projects already guarantees this.
    keys = [(label, proj) for (label, proj), entries in grouped.items() if entries]
    if not keys:
        return None
    keys.sort()  # sort by (label, project_ref)

    parts: list[str] = []
    for label, proj in keys:
        entries = grouped[(label, proj)]
        map_parts = [f"[{m['id']}] {m['short_title']}" for m in entries]
        prefix = f"{label}/{proj}" if multi_label else proj
        parts.append(f"{prefix}: {', '.join(map_parts)}")

    body = "; ".join(parts)
    return f"Maps in other domains {_EM_DASH} {body}"


def compose_directory(
    project_ref: str,
    maps: list[MapEntry],
    index: dict[str, list[tuple[str, MapEntry]]],
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


def render_new_maps(new_maps: list[tuple[MapKey, MapEntry]]) -> str | None:
    """Render the delta FYI line for maps newly surfaced this session.

    ``new_maps`` is the FULL delta — every map not already in the
    session's ``surfaced_map_ids`` — as ``(MapKey, MapEntry)`` pairs, NOT
    pre-capped by the caller. Returns ``None`` when ``new_maps`` is empty
    (defensive; the empty-delta guard is enforced by the caller in
    :mod:`personal_kb_hook.cli`, which must not call this at all in that
    case).

    Entries are sorted into stable order by :class:`MapKey`'s natural
    ``(label, id)`` tuple ordering before display, so repeated calls over
    the same delta always name the same maps in the same order. When more
    than :data:`NEW_MAPS_DISPLAY_CAP` maps are new, only the first
    ``NEW_MAPS_DISPLAY_CAP`` (in that order) are named and a
    ``(+N more)`` suffix reports how many were left off — a display limit
    only; the caller still records and telemeters every map in the delta.

    Format (SHORT titles only, no long_title — matches render_whisper):
    * exactly one new map:
      ``New map — <label>/[<id>] <short_title>``
    * more than one:
      ``New maps — <label>/[<id>] <short_title>, <label>/[<id>] <short_title>``
    * capped:
      ``New maps — ... <label>/[<id>] <short_title> (+N more)``

    The source-KB ``label`` is ALWAYS included — unlike :func:`render_whisper`,
    which only prefixes it in multi-KB rosters — so a single-KB roster's
    delta FYI still reads consistently with the (always-labelled) form
    other short-form surfaces use.
    """
    if not new_maps:
        return None
    ordered = sorted(new_maps, key=lambda pair: pair[0])
    shown = ordered[:NEW_MAPS_DISPLAY_CAP]
    overflow = len(ordered) - len(shown)
    parts = [f"{key.label}/[{entry['id']}] {entry['short_title']}" for key, entry in shown]
    header = "New map" if len(ordered) == 1 else "New maps"
    body = ", ".join(parts)
    if overflow > 0:
        body = f"{body} (+{overflow} more)"
    return f"{header} {_EM_DASH} {body}"


def render_whisper(
    map_entry: Mapping[str, object],
    label: str | None = None,
    multi_kb: bool = False,
) -> str:
    """Render the whisper line for a pending listener map.

    The output is keyed on whether the roster contains more than one KB
    (``multi_kb``), NOT on a distinct-label count (which
    :func:`render_cross_directory` uses) — at the call site,
    :func:`render_whisper` sees one candidate at a time, so it cannot
    inspect distinct labels itself.

    Four format branches (all use the U+2014 em dash with a single space
    on each side):

    * single-KB, empty long_title:
      ``Possibly relevant map — [<id>] <short_title>``
    * single-KB, populated long_title:
      ``Possibly relevant map — [<id>] <short_title>: <long_title>``
    * multi-KB, empty long_title:
      ``Possibly relevant map — <label>/[<id>] <short_title>``
    * multi-KB, populated long_title:
      ``Possibly relevant map — <label>/[<id>] <short_title>: <long_title>``

    When ``long_title`` is empty the ``': <long_title>'`` suffix is omitted.
    The scaffold phrase contains no banned tokens; no runtime filtering of
    server-supplied titles is performed.
    """
    entry_id = map_entry.get("id")
    short_title = map_entry.get("short_title")
    long_title = map_entry.get("long_title")
    if not isinstance(entry_id, str):
        entry_id = ""
    if not isinstance(short_title, str):
        short_title = ""
    if not isinstance(long_title, str) or not long_title:
        long_title = ""
    prefix = f"{label}/" if (multi_kb and label) else ""
    if long_title:
        return f"Possibly relevant map {_EM_DASH} {prefix}[{entry_id}] {short_title}: {long_title}"
    return f"Possibly relevant map {_EM_DASH} {prefix}[{entry_id}] {short_title}"


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
