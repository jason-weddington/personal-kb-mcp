"""Local-mode defaults for the hook's service URL + API key (stdlib-only).

Mirrors ``personal_kb.config.LOCAL_KB_URL`` / ``LOCAL_KB_API_KEY``; the
drift-guard test in the main repo asserts both packages agree. The hook can
never import ``personal_kb`` and must NEVER spawn the daemon: with nothing
listening it stays silent like on every other error path.
"""

from __future__ import annotations

import os
import urllib.parse

LOCAL_KB_URL = "http://127.0.0.1:8765"
LOCAL_KB_API_KEY = "local-no-auth"

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _is_loopback(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except (ValueError, TypeError):
        return False
    return host.lower() in _LOOPBACK_HOSTS


def resolve_url_key() -> tuple[str, str] | None:
    """Return ``(url, key)`` from env with local-mode defaults, or ``None``.

    Unset/empty URL means :data:`LOCAL_KB_URL`. An unset/empty key gets the
    local sentinel only for loopback URLs; a remote URL with no key yields
    ``None`` (fail closed, silently).
    """
    url = os.environ.get("PERSONAL_KB_URL", "") or LOCAL_KB_URL
    key = os.environ.get("PERSONAL_KB_API_KEY", "")
    if not key:
        if not _is_loopback(url):
            return None
        key = LOCAL_KB_API_KEY
    return url, key
