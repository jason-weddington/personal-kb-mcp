"""Opt-in CLI hook tooling for surfacing mental_map directories.

The ``personal-kb-hook`` console script lives here. It is designed to be wired
into a harness's ``SessionStart`` / ``UserPromptSubmit`` hooks. The hook itself
is intentionally tiny and stdlib-only — it never touches the KB database or
talks to the MCP server. It reads a denormalized JSONL index that the MCP
server writes on ``mental_map`` create/update/deactivate, walks up from the
session's working directory to find a committed ``.kb_project`` file, and
prints a factual directory of the resolved project's maps.
"""
