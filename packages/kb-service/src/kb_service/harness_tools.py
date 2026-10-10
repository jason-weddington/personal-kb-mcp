"""Harness tool-name normalization for turn digests and gate data.

Contract:

1. Canonical tool names are Claude Code's (Bash, Edit, Write, Read, Glob,
   Grep, LS) plus ``run_checks`` and ``finish``, which have no Claude Code
   equivalent. Talos has no write_file tool (edit_file also creates files),
   and its list_files lists a directory tree, so it maps to LS, not Glob.
2. Talos POSTs harness "talos" with its native tool names and never renames
   them.
3. Talos fills the existing ``target`` field with the raw value of its native
   target arg (bash: ``command``; edit_file/read_file: ``path``; list_files:
   ``path``, a workspace-relative directory defaulting to "."; run_checks and
   finish: ""), at most 500 chars (TURN_TARGET_MAX). It must do this because
   TurnToolCallItem has no args field and unknown keys are dropped.
4. For a mapped harness the KB recomputes target_class with
   ``kb_core.cues.target_class``.
5. Every per-harness map is injective, so the native name of a stored call is
   the inverse map applied to (harness, tool), and no native-name field is
   stored.
6. GET /api/kb/prevention serves the map as ``tool_map`` while the gate index
   stays canonical, so a mapped harness translates its native call name
   through ``tool_map`` before matching the index; the presence of the
   ``tool_map`` key in that response is the signal that this KB normalizes
   that harness's digests.
7. Harness "claude-code" and any harness that is not a key of
   HARNESS_TOOL_MAPS are never remapped.
"""

from kb_core.cues import target_class

from kb_service.models import TurnDigestRequest, TurnToolCallItem

HARNESS_TOOL_MAPS: dict[str, dict[str, str]] = {
    "talos": {
        "bash": "Bash",
        "edit_file": "Edit",
        "read_file": "Read",
        "list_files": "LS",
        "run_checks": "run_checks",
        "finish": "finish",
    }
}


def canonical_tool(harness: str | None, tool: str) -> str:
    """Return the canonical name of *tool* for *harness* (unchanged if unmapped)."""
    if harness is None:
        return tool
    return HARNESS_TOOL_MAPS.get(harness, {}).get(tool, tool)


def tool_map(harness: str | None) -> dict[str, str]:
    """Return a fresh copy of the native-to-canonical map for *harness*."""
    if harness is None:
        return {}
    return dict(HARNESS_TOOL_MAPS.get(harness, {}))


def normalize_turn_tools(
    body: TurnDigestRequest,
) -> tuple[TurnDigestRequest, list[str]]:
    """Return ``(normalized_body, unmapped_tools)`` without mutating *body*."""
    mapping = HARNESS_TOOL_MAPS.get(body.harness)
    if mapping is None:
        return body, []
    unmapped: set[str] = set()
    new_items = []
    for item in body.items:
        if isinstance(item, TurnToolCallItem):
            if item.tool not in mapping:
                unmapped.add(item.tool)
            canonical = mapping.get(item.tool, item.tool)
            item = item.model_copy(
                update={
                    "tool": canonical,
                    "target_class": target_class(canonical, item.target),
                }
            )
        new_items.append(item)
    return body.model_copy(update={"items": new_items}), sorted(unmapped)
