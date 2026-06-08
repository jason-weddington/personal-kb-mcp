"""Standalone, stdlib-only ``personal-kb-hook`` console script.

The hook surfaces a project's ``mental_map`` directory into a Claude Code
session via ``SessionStart`` and ``UserPromptSubmit`` hooks. It is wired up
by adding a hook entry that invokes ``personal-kb-hook --format=claude-json``.

This package is the RUNTIME half of the hook — installed via
``uv tool install --from "git+ssh://...#subdirectory=packages/personal-kb-hook"
personal-kb-hook``. It is intentionally tiny, stdlib-only, and never imports
the main ``personal_kb`` server package — that's the whole reason it lives
in its own distribution. The MCP-side writer that populates the on-disk
JSONL index stays in the main ``personal_kb`` package
(``personal_kb.maps_index_writer``).

The on-disk path contract is duplicated across both packages (the main
package's ``personal_kb.config`` and this package's
``personal_kb_hook.paths``) and a drift-guard test in the main repo asserts
the two implementations agree byte-for-byte — including under a custom
``KB_DB_PATH``.
"""
