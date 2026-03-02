"""Tests for new config functions."""

from personal_kb.config import (
    get_contributor,
    get_pg_pool_max,
    get_pg_pool_min,
    get_team,
    is_safety_skip,
)


def test_get_contributor_default():
    """Default contributor is None."""
    assert get_contributor() is None


def test_get_contributor_set(monkeypatch):
    monkeypatch.setenv("KB_CONTRIBUTOR", "jason")
    assert get_contributor() == "jason"


def test_get_contributor_empty_string(monkeypatch):
    """Empty string returns None, not empty string."""
    monkeypatch.setenv("KB_CONTRIBUTOR", "")
    assert get_contributor() is None


def test_get_team_default():
    assert get_team() is None


def test_get_team_set(monkeypatch):
    monkeypatch.setenv("KB_TEAM", "platform")
    assert get_team() == "platform"


def test_get_team_empty_string(monkeypatch):
    monkeypatch.setenv("KB_TEAM", "")
    assert get_team() is None


def test_get_pg_pool_min_default():
    assert get_pg_pool_min() == 1


def test_get_pg_pool_min_set(monkeypatch):
    monkeypatch.setenv("KB_PG_POOL_MIN", "3")
    assert get_pg_pool_min() == 3


def test_get_pg_pool_max_default():
    assert get_pg_pool_max() == 5


def test_get_pg_pool_max_set(monkeypatch):
    monkeypatch.setenv("KB_PG_POOL_MAX", "20")
    assert get_pg_pool_max() == 20


def test_is_safety_skip_default():
    assert is_safety_skip() is False


def test_is_safety_skip_true(monkeypatch):
    monkeypatch.setenv("KB_SKIP_SAFETY", "TRUE")
    assert is_safety_skip() is True


def test_is_safety_skip_false(monkeypatch):
    monkeypatch.setenv("KB_SKIP_SAFETY", "FALSE")
    assert is_safety_skip() is False
