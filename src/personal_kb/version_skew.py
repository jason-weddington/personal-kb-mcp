"""Warn when this client is newer than the hosted KB server it talks to."""

from __future__ import annotations

import importlib.metadata
import logging

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)


def client_version() -> str | None:
    """Return the installed ``personal-kb`` version, or None if unknown."""
    try:
        return importlib.metadata.version("personal-kb")
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_newer(client: str, server: str) -> bool:
    """True when *client* is newer than *server*; dev/local suffixes count as newer."""
    c, s = Version(client), Version(server)
    if c.release == s.release and c.local is None and not c.is_devrelease:
        return c > s
    # Same release with a dev/local suffix on the client is newer than a plain server.
    if c.release == s.release and (c.is_devrelease or c.local is not None):
        return s.local is None and not s.is_devrelease
    return c > s


def compare_versions(client: str | None, server: object) -> str | None:
    """Return a skew message if the client is newer than the server, else None."""
    if not client or not isinstance(server, str) or not server:
        logger.debug("version skew check skipped: client=%r server=%r", client, server)
        return None
    try:
        newer = _is_newer(client, server)
    except InvalidVersion:
        logger.debug("version skew check: unparseable versions %r / %r", client, server)
        return None
    if not newer:
        return None
    return (
        f"Client personal-kb {client} is newer than the server ({server}); "
        "server-side features may be missing."
    )


async def check_version_skew(base_url: str) -> str | None:
    """Return a skew message for a non-loopback *base_url*, logging a WARNING."""
    from personal_kb.daemon import _fetch_health, is_loopback_url

    if is_loopback_url(base_url):
        return None
    body = await _fetch_health(base_url)
    server = body.get("version") if body else None
    message = compare_versions(client_version(), server)
    if message:
        logger.warning(message)
    return message
