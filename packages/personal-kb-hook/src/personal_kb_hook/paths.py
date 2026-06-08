"""Stdlib-only on-disk path helpers used by the standalone hook.

These two functions DUPLICATE the equivalents in
``personal_kb.config.get_maps_index_path`` /
``personal_kb.config.get_hook_scratch_path``. The duplication is deliberate
— the standalone hook package must not import the main ``personal_kb``
distribution (that is the whole point of the split).

A drift-guard test in the main repo
(``tests/test_path_drift_guard.py``) imports both implementations and
asserts they produce identical paths for representative inputs, including
a custom ``KB_DB_PATH``. If you change one side, change the other and
re-run that test.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DB_PATH = "~/.local/share/personal_kb/knowledge.db"


def get_maps_index_path() -> Path:
    """Return the on-disk maps index JSONL path.

    Sibling of the KB database file: ``<db_dir>/maps_index.jsonl``. The MCP
    server writes this file (via ``personal_kb.maps_index_writer``) on
    every ``mental_map`` create/update/deactivate; the standalone hook
    reads it.
    """
    raw = os.environ.get("KB_DB_PATH", _DEFAULT_DB_PATH)
    return Path(raw).expanduser().parent / "maps_index.jsonl"


def get_hook_scratch_path(session_id: str) -> Path:
    """Return the per-session hook scratch file path.

    Used by the hook to suppress re-injection of the same map directory in
    a session. Stored under ``~/.cache/personal_kb/`` so it survives across
    the pair of ``SessionStart`` / ``UserPromptSubmit`` invocations within
    one session, but is naturally torn down with the cache.
    """
    return Path(f"~/.cache/personal_kb/injected-{session_id}.json").expanduser()
