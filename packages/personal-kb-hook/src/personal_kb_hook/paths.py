"""Stdlib-only on-disk path helpers used by the standalone hook.

These two functions DUPLICATE the equivalents in
``personal_kb.config.get_maps_index_path`` /
``personal_kb.config.get_hook_scratch_path``. The duplication is deliberate
— the standalone hook package must not import the main ``personal_kb``
distribution (that is the whole point of the split).

The role-keying formula for ``get_maps_index_path`` MUST stay in sync with
``personal_kb.config.get_maps_index_path``. A drift-guard test in the main
repo (``tests/test_path_drift_guard.py``) imports both implementations and
asserts they produce identical paths for representative inputs, including
a custom ``KB_DB_PATH`` and every supported ``KB_INSTANCE_ROLE`` value. If
you change one side, change the other and re-run that test.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DB_PATH = "~/.local/share/personal_kb/knowledge.db"


def get_maps_index_path() -> Path:
    """Return the on-disk maps index JSONL path, scoped by instance role.

    Sibling of the KB database file: ``<db_dir>/maps_index.{role}.jsonl``,
    where ``role = KB_INSTANCE_ROLE.lower() or "default"``. The MCP server
    writes one such file per instance (via
    ``personal_kb.maps_index_writer``) on every ``mental_map``
    create/update/deactivate; the standalone hook globs all
    ``maps_index*.jsonl`` files in this directory and merges them.

    The role keying MUST stay in sync with
    ``personal_kb.config.get_maps_index_path`` (drift-guard test).
    """
    role = os.environ.get("KB_INSTANCE_ROLE", "").lower() or "default"
    raw = os.environ.get("KB_DB_PATH", _DEFAULT_DB_PATH)
    return Path(raw).expanduser().parent / f"maps_index.{role}.jsonl"


def get_hook_scratch_path(session_id: str) -> Path:
    """Return the per-session hook scratch file path.

    Used by the hook to suppress re-injection of the same map directory in
    a session. Stored under ``~/.cache/personal_kb/`` so it survives across
    the pair of ``SessionStart`` / ``UserPromptSubmit`` invocations within
    one session, but is naturally torn down with the cache.
    """
    return Path(f"~/.cache/personal_kb/injected-{session_id}.json").expanduser()


def get_listener_cache_path(session_id: str) -> Path:
    """Return the per-session listener cache file path.

    Used by the listener worker to store pending whisper entries and the set
    of already-whispered map ids for the session. Stored under
    ``~/.cache/personal_kb/`` as a sibling of the hook scratch file.
    """
    return Path(f"~/.cache/personal_kb/listener-{session_id}.json").expanduser()
