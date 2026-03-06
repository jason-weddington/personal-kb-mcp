"""TTL parsing utilities for entry expiry."""

import re
from datetime import UTC, datetime, timedelta

_TTL_PATTERN = re.compile(r"^(\d+)([hdw])$")

_UNIT_MAP = {
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_ttl(ttl: str) -> timedelta:
    """Parse a TTL string like '7d', '24h', '2w' into a timedelta.

    Raises ValueError on invalid format or zero duration.
    """
    match = _TTL_PATTERN.match(ttl)
    if not match:
        raise ValueError(
            f'Invalid TTL format "{ttl}". Use Nh (hours), Nd (days), or Nw (weeks). '
            "Examples: 24h, 7d, 2w"
        )
    amount = int(match.group(1))
    unit = match.group(2)
    if amount == 0:
        raise ValueError("TTL must be greater than zero.")
    return timedelta(**{_UNIT_MAP[unit]: amount})


def compute_expires_at(ttl: str, now: datetime | None = None) -> datetime:
    """Compute an absolute expiry datetime from a TTL string."""
    if now is None:
        now = datetime.now(UTC)
    return now + parse_ttl(ttl)
