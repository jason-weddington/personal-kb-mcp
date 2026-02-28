"""Tests for confidence decay model."""

from datetime import UTC, datetime

from personal_kb.confidence.decay import (
    STALENESS_THRESHOLD,
    compute_effective_confidence,
    staleness_warning,
)
from personal_kb.models.entry import EntryType


def test_no_decay_at_creation():
    now = datetime.now(UTC)
    eff = compute_effective_confidence(0.9, EntryType.FACTUAL_REFERENCE, now, now)
    assert eff == 0.9


def test_factual_reference_decays_fast():
    created = datetime(2025, 1, 1, tzinfo=UTC)
    now = datetime(2025, 7, 1, tzinfo=UTC)  # ~180 days, 2 half-lives
    eff = compute_effective_confidence(1.0, EntryType.FACTUAL_REFERENCE, created, now)
    # After 2 half-lives: 1.0 * 0.25 = 0.25
    assert 0.2 < eff < 0.3


def test_lesson_learned_decays_slowly():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    now = datetime(2026, 1, 1, tzinfo=UTC)  # 2 years
    eff = compute_effective_confidence(1.0, EntryType.LESSON_LEARNED, created, now)
    # 2 years / 5 year half-life = 0.4 half-lives → ~0.76
    assert eff > 0.7


def test_decision_moderate_decay():
    created = datetime(2025, 1, 1, tzinfo=UTC)
    now = datetime(2026, 1, 1, tzinfo=UTC)  # 1 year = 1 half-life
    eff = compute_effective_confidence(1.0, EntryType.DECISION, created, now)
    # After 1 half-life: 1.0 * 0.5 = 0.5
    assert 0.45 < eff < 0.55


def test_pattern_convention_durable():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    now = datetime(2026, 1, 1, tzinfo=UTC)  # 2 years, 730 day half-life
    eff = compute_effective_confidence(1.0, EntryType.PATTERN_CONVENTION, created, now)
    # ~1 half-life → ~0.5
    assert 0.45 < eff < 0.55


def test_staleness_warning_below_threshold():
    warning = staleness_warning(0.3, EntryType.FACTUAL_REFERENCE)
    assert warning is not None
    assert "Stale" in warning


def test_no_staleness_warning_above_threshold():
    warning = staleness_warning(0.7, EntryType.FACTUAL_REFERENCE)
    assert warning is None


def test_staleness_warning_at_threshold():
    warning = staleness_warning(STALENESS_THRESHOLD, EntryType.DECISION)
    assert warning is None


def test_future_date_no_decay():
    """If created_at is in the future (clock skew), no decay applied."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    future = datetime(2026, 1, 1, tzinfo=UTC)
    eff = compute_effective_confidence(0.9, EntryType.FACTUAL_REFERENCE, future, now)
    assert eff == 0.9


def test_last_accessed_resets_decay_clock():
    """Accessing an entry should reset the decay anchor."""
    created = datetime(2025, 1, 1, tzinfo=UTC)
    now = datetime(2025, 7, 1, tzinfo=UTC)  # ~180 days, 2 half-lives

    # Without access: heavily decayed
    eff_no_access = compute_effective_confidence(1.0, EntryType.FACTUAL_REFERENCE, created, now)
    assert eff_no_access < 0.3  # ~0.25

    # With recent access: much less decay
    last_accessed = datetime(2025, 6, 1, tzinfo=UTC)  # 30 days ago
    eff_accessed = compute_effective_confidence(
        1.0, EntryType.FACTUAL_REFERENCE, created, now, last_accessed=last_accessed
    )
    assert eff_accessed > 0.7  # ~30 days into 90-day half-life


def test_last_accessed_none_falls_back_to_created():
    """When last_accessed is None, behavior matches old code."""
    created = datetime(2025, 1, 1, tzinfo=UTC)
    now = datetime(2025, 7, 1, tzinfo=UTC)

    eff_none = compute_effective_confidence(
        1.0, EntryType.FACTUAL_REFERENCE, created, now, last_accessed=None
    )
    eff_default = compute_effective_confidence(1.0, EntryType.FACTUAL_REFERENCE, created, now)
    assert eff_none == eff_default


def test_last_accessed_older_than_created_uses_created():
    """If last_accessed is older than created_at, created_at is used."""
    created = datetime(2025, 6, 1, tzinfo=UTC)
    old_access = datetime(2025, 1, 1, tzinfo=UTC)  # before creation
    now = datetime(2025, 7, 1, tzinfo=UTC)

    eff = compute_effective_confidence(
        1.0, EntryType.FACTUAL_REFERENCE, created, now, last_accessed=old_access
    )
    eff_no_access = compute_effective_confidence(1.0, EntryType.FACTUAL_REFERENCE, created, now)
    assert eff == eff_no_access


def test_last_accessed_naive_timezone_handled():
    """Naive last_accessed datetime should work (assumed UTC)."""
    created = datetime(2025, 1, 1, tzinfo=UTC)
    now = datetime(2025, 7, 1, tzinfo=UTC)
    last_accessed = datetime(2025, 6, 15)  # naive -- should be treated as UTC

    eff = compute_effective_confidence(
        1.0, EntryType.FACTUAL_REFERENCE, created, now, last_accessed=last_accessed
    )
    # 16 days with 90-day half-life -> ~0.88
    assert eff > 0.8
