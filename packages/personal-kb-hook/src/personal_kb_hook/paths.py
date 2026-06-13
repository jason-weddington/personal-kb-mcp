"""Stdlib-only on-disk path helpers used by the standalone hook.

``get_hook_scratch_path`` DUPLICATES the equivalent in
``personal_kb.config.get_hook_scratch_path``. The duplication is deliberate
— the standalone hook package must not import the main ``personal_kb``
distribution (that is the whole point of the split). ``get_listener_cache_path``
is hook-only.
"""

from __future__ import annotations

from pathlib import Path


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
