"""Tests for TTL parsing utilities."""

from datetime import UTC, datetime, timedelta

import pytest

from personal_kb.tools.ttl import compute_expires_at, parse_ttl


class TestParseTtl:
    def test_hours(self):
        assert parse_ttl("24h") == timedelta(hours=24)

    def test_days(self):
        assert parse_ttl("7d") == timedelta(days=7)

    def test_weeks(self):
        assert parse_ttl("2w") == timedelta(weeks=2)

    def test_single_hour(self):
        assert parse_ttl("1h") == timedelta(hours=1)

    def test_large_number(self):
        assert parse_ttl("365d") == timedelta(days=365)

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid TTL format"):
            parse_ttl("7m")

    def test_no_number(self):
        with pytest.raises(ValueError, match="Invalid TTL format"):
            parse_ttl("d")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid TTL format"):
            parse_ttl("")

    def test_zero(self):
        with pytest.raises(ValueError, match="greater than zero"):
            parse_ttl("0d")

    def test_negative_not_matched(self):
        with pytest.raises(ValueError, match="Invalid TTL format"):
            parse_ttl("-1d")

    def test_float_not_matched(self):
        with pytest.raises(ValueError, match="Invalid TTL format"):
            parse_ttl("1.5d")


class TestComputeExpiresAt:
    def test_computes_from_now(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = compute_expires_at("7d", now=now)
        assert result == datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)

    def test_hours(self):
        now = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        result = compute_expires_at("24h", now=now)
        assert result == datetime(2025, 6, 16, 10, 0, 0, tzinfo=UTC)

    def test_weeks(self):
        now = datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
        result = compute_expires_at("2w", now=now)
        assert result == datetime(2025, 3, 15, 0, 0, 0, tzinfo=UTC)

    def test_uses_current_time_when_no_now(self):
        before = datetime.now(UTC)
        result = compute_expires_at("1d")
        after = datetime.now(UTC)
        assert before + timedelta(days=1) <= result <= after + timedelta(days=1)

    def test_invalid_ttl_raises(self):
        with pytest.raises(ValueError):
            compute_expires_at("bad")
