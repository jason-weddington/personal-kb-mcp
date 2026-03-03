"""RDS/Aurora IAM authentication helpers.

Keeps all AWS/boto3 knowledge out of PostgresBackend. Three components:

- ``parse_dsn()`` — extract host/port/username from a connection URL
- ``make_token_factory()`` — returns a callable that generates IAM auth tokens
- ``make_ssl_context()`` — returns an SSL context for IAM auth (requires TLS)
"""

from __future__ import annotations

import ssl
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class DSNComponents:
    """Parsed components from a PostgreSQL connection URL."""

    host: str
    port: int
    username: str


def parse_dsn(url: str) -> DSNComponents:
    """Extract host, port, and username from a PostgreSQL connection URL.

    Args:
        url: A ``postgresql://`` connection URL.

    Returns:
        Parsed DSN components.

    Raises:
        ValueError: If hostname or username is missing from the URL.
    """
    parsed = urllib.parse.urlparse(url)

    if not parsed.hostname:
        msg = f"DSN missing hostname: {url}"
        raise ValueError(msg)
    if not parsed.username:
        msg = f"DSN missing username: {url}"
        raise ValueError(msg)

    return DSNComponents(
        host=parsed.hostname,
        port=parsed.port or 5432,
        username=parsed.username,
    )


def make_token_factory(host: str, port: int, username: str, region: str) -> Callable[[], str]:
    """Return a callable that generates RDS IAM auth tokens.

    The boto3 client is created once (captured in the closure).
    Each call generates a fresh token (valid ~15 minutes).

    Args:
        host: RDS/Aurora endpoint hostname.
        port: Database port.
        username: Database username.
        region: AWS region for SigV4 signing.

    Returns:
        A zero-arg callable that returns a fresh auth token string.
    """
    import boto3

    client = boto3.client("rds", region_name=region)

    def _generate_token() -> str:
        token: str = client.generate_db_auth_token(
            DBHostname=host, Port=port, DBUsername=username, Region=region
        )
        return token

    return _generate_token


def make_ssl_context() -> ssl.SSLContext:
    """Return an SSL context suitable for RDS/Aurora IAM auth.

    Uses system CA certificates. Users can override the CA bundle
    via the ``SSL_CERT_FILE`` environment variable if needed.

    Returns:
        An ``ssl.SSLContext`` with certificate verification enabled.
    """
    ctx = ssl.create_default_context()
    # create_default_context() already sets:
    #   verify_mode = CERT_REQUIRED
    #   check_hostname = True
    return ctx
