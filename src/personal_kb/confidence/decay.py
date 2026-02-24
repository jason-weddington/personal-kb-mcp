"""Confidence decay by entry type."""

import math
from datetime import UTC, datetime

from personal_kb.models.entry import EntryType

# Half-life in days per entry type
HALF_LIVES: dict[EntryType, float] = {
    EntryType.FACTUAL_REFERENCE: 90.0,  # 3 months — facts go stale fast
    EntryType.DECISION: 365.0,  # 1 year — decisions persist but context shifts
    EntryType.PATTERN_CONVENTION: 730.0,  # 2 years — conventions are durable
    EntryType.LESSON_LEARNED: 1825.0,  # 5 years — hard-won lessons stick
}

STALENESS_THRESHOLD = 0.5


def compute_effective_confidence(
    base_confidence: float,
    entry_type: EntryType,
    created_at: datetime,
    now: datetime | None = None,
) -> float:
    """Compute confidence after time-based decay.

    Uses exponential decay: effective = base * 2^(-age_days / half_life)
    """
    if now is None:
        now = datetime.now(UTC)

    # Ensure both are timezone-aware for comparison
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age_days = (now - created_at).total_seconds() / 86400.0
    if age_days <= 0:
        return base_confidence

    half_life = HALF_LIVES[entry_type]
    decay_factor = math.pow(2, -age_days / half_life)
    return round(base_confidence * decay_factor, 4)


def staleness_warning(effective_confidence: float, entry_type: EntryType) -> str | None:
    """Return a warning string if the entry is stale, else None."""
    if effective_confidence >= STALENESS_THRESHOLD:
        return None
    return (
        f"Stale {entry_type.value} entry (confidence: {effective_confidence:.0%}). "
        "Consider verifying this information is still current."
    )
