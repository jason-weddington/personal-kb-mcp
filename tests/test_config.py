"""Tests for config functions."""

import pytest

from personal_kb.config import (
    get_contributor,
    get_embedding_dim,
    get_extraction_provider,
    get_ollama_timeout,
    get_pg_pool_max,
    get_pg_pool_min,
    get_pg_region,
    get_query_provider,
    get_team,
    is_pg_iam_auth,
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


# -- IAM auth config --


def test_is_pg_iam_auth_default():
    assert is_pg_iam_auth() is False


def test_is_pg_iam_auth_true(monkeypatch):
    monkeypatch.setenv("KB_PG_IAM_AUTH", "TRUE")
    assert is_pg_iam_auth() is True


def test_is_pg_iam_auth_case_insensitive(monkeypatch):
    monkeypatch.setenv("KB_PG_IAM_AUTH", "true")
    assert is_pg_iam_auth() is True


def test_is_pg_iam_auth_false(monkeypatch):
    monkeypatch.setenv("KB_PG_IAM_AUTH", "FALSE")
    assert is_pg_iam_auth() is False


def test_get_pg_region_default():
    assert get_pg_region() == "us-east-1"


def test_get_pg_region_set(monkeypatch):
    monkeypatch.setenv("KB_PG_REGION", "eu-west-1")
    assert get_pg_region() == "eu-west-1"


# -- Numeric validation --


def test_int_validation_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("KB_EMBEDDING_DIM", "abc")
    with pytest.raises(ValueError, match="KB_EMBEDDING_DIM='abc' is not a valid integer"):
        get_embedding_dim()


def test_float_validation_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="KB_OLLAMA_TIMEOUT='not-a-number' is not a valid number"):
        get_ollama_timeout()


def test_int_validation_accepts_valid(monkeypatch):
    monkeypatch.setenv("KB_EMBEDDING_DIM", "512")
    assert get_embedding_dim() == 512


def test_float_validation_accepts_valid(monkeypatch):
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "5.5")
    assert get_ollama_timeout() == 5.5


# -- Provider validation --


def test_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "openai")
    with pytest.raises(ValueError, match="KB_EXTRACTION_PROVIDER='openai' is not valid"):
        get_extraction_provider()


def test_provider_accepts_valid(monkeypatch):
    for provider in ("anthropic", "bedrock", "ollama"):
        monkeypatch.setenv("KB_QUERY_PROVIDER", provider)
        assert get_query_provider() == provider


def test_provider_case_insensitive(monkeypatch):
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "Anthropic")
    assert get_extraction_provider() == "anthropic"
