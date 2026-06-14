"""Multi-KB roster loader (P0 plumbing — no behavior change).

Loads the multi-KB roster from
``<XDG_CONFIG_HOME or ~/.config>/personal_kb/kbs.json`` and returns a
list of typed :class:`KbEntry` records (``label``, ``url``, ``key``).

The on-disk roster file is a JSON list whose elements look like::

    {"label": "team", "url": "https://team.kb/", "key_file": "~/.team_kb_key"}

``key_file`` holds a path to a secrets file whose stripped contents
become :attr:`KbEntry.key`. An inline ``key`` field is **not** accepted
in this phase — secrets stay out of any future git-synced config.

Legacy fallback
---------------
When the config file is absent, empty, non-JSON, parses to a non-list,
or any read/parse error occurs, :func:`load_roster` synthesizes a single
legacy entry from the existing global env vars used elsewhere in the
package (``PERSONAL_KB_URL`` / ``PERSONAL_KB_API_KEY``) — but ONLY when
both are present and non-empty. Otherwise it returns ``[]``.

An *explicitly empty* roster (``[]`` on disk) and an *all-entries-dropped*
roster (every element fails validation) both return ``[]`` — they are
intentional zero-KB configurations and MUST NOT fall back to the
legacy-env entry.

This module is **stdlib-only** (``json``, ``os``, ``urllib.parse``,
``pathlib``, ``typing``). The package's ``dependencies = []`` invariant
is preserved. The entire body is wrapped so :func:`load_roster` NEVER
raises.

Note: this module is intentionally NOT wired into any caller in P0; the
fan-out into ``http_index.load_index`` / the listener / ``cli.py`` is
the work of P1 / P2.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import NamedTuple


class KbEntry(NamedTuple):
    """One KB in the roster: a label, a base URL, and a bearer key.

    Deliberately distinct from :class:`personal_kb_hook.index_reader.MapEntry`
    (a ``TypedDict`` of ``id`` / ``short_title`` / ``long_title``) so a bare
    KB-id cannot be passed where a ``KbEntry`` is expected — the type checker
    will reject the swap.
    """

    label: str
    url: str
    key: str


_CONFIG_DIR_SEGMENT = "personal_kb"  # underscore, NOT hyphen (pinned by kb-01828)
_CONFIG_FILE_NAME = "kbs.json"
_LEGACY_LABEL = "personal"


def _config_home() -> Path:
    """Return the config root: ``$XDG_CONFIG_HOME`` if set+non-empty, else ``~/.config``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg)
    return Path.home() / ".config"


def _legacy_fallback() -> list[KbEntry]:
    """Synthesize a single 'personal' entry from legacy env vars, or return ``[]``."""
    url = os.environ.get("PERSONAL_KB_URL", "")
    key = os.environ.get("PERSONAL_KB_API_KEY", "")
    if url and key:
        return [KbEntry(label=_LEGACY_LABEL, url=url, key=key)]
    return []


def _is_http_url(url: str) -> bool:
    """Return True iff ``url``'s scheme is http or https."""
    try:
        scheme = urllib.parse.urlparse(url).scheme.lower()
    except (ValueError, TypeError):
        return False
    return scheme in {"http", "https"}


def _read_key_file(key_file: str) -> str | None:
    """Read ``key_file``'s text, strip it, return the result or ``None`` on failure/blank.

    Errors (missing file, decode error, permission error, etc.) are caught
    individually so one bad entry does not abort the whole roster.
    """
    try:
        contents = Path(key_file).expanduser().read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    stripped = contents.strip()
    if not stripped:
        return None
    return stripped


def _parse_entry(raw: object) -> KbEntry | None:
    """Validate one raw roster element. Return a :class:`KbEntry` or ``None`` if invalid.

    Validation rules (all required):
      * ``raw`` is a ``dict``.
      * ``label`` is a non-empty ``str``.
      * ``url`` is a non-empty ``str`` whose scheme is http or https.
      * ``key_file`` is a non-empty ``str`` path whose ``expanduser()``'d,
        utf-8 text contents are non-empty after ``.strip()``.

    Any failure → silently dropped (returns ``None``).
    """
    if not isinstance(raw, dict):
        return None
    label = raw.get("label")
    url = raw.get("url")
    key_file = raw.get("key_file")
    if not isinstance(label, str) or not label:
        return None
    if not isinstance(url, str) or not url:
        return None
    if not _is_http_url(url):
        return None
    if not isinstance(key_file, str) or not key_file:
        return None
    key = _read_key_file(key_file)
    if key is None:
        return None
    return KbEntry(label=label, url=url, key=key)


def load_roster() -> list[KbEntry]:
    """Load the multi-KB roster, falling back to the legacy env entry on failure.

    Reads ``<XDG_CONFIG_HOME or ~/.config>/personal_kb/kbs.json``. Returns:

    * A list of validated :class:`KbEntry` records in file order when the
      config file parses to a JSON list. Invalid elements are silently
      dropped.
    * ``[]`` when the file parses to an empty JSON list ``[]`` OR every
      element fails validation — these are intentional zero-KB
      configurations, NOT triggers for the legacy fallback.
    * The legacy fallback (a single ``KbEntry(label='personal', ...)``
      built from ``PERSONAL_KB_URL`` + ``PERSONAL_KB_API_KEY``) when the
      file is absent / empty / non-JSON / a non-list / unreadable, and
      both legacy env vars are present and non-empty.
    * ``[]`` when the legacy-fallback trigger fires but either legacy env
      var is absent or empty.

    Never raises.
    """
    try:
        path = _config_home() / _CONFIG_DIR_SEGMENT / _CONFIG_FILE_NAME
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError):
            return _legacy_fallback()
        if not raw_text.strip():
            return _legacy_fallback()
        try:
            parsed: object = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            return _legacy_fallback()
        if not isinstance(parsed, list):
            return _legacy_fallback()
        # Parseable list (including []): never falls back to legacy.
        entries: list[KbEntry] = []
        for raw in parsed:
            entry = _parse_entry(raw)
            if entry is not None:
                entries.append(entry)
        return entries
    except Exception:
        return _legacy_fallback()
