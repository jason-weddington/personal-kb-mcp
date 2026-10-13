"""Tests for config functions."""

import logging

import pytest

from personal_kb.config import (
    build_anthropic_config,
    get_anthropic_model,
    get_anthropic_thinking,
    get_contributor,
    get_embedding_dim,
    get_embedding_keep_alive,
    get_extraction_provider,
    get_ollama_timeout,
    get_pg_pool_max,
    get_pg_pool_min,
    get_pg_region,
    get_query_provider,
    get_synthesis_effort,
    get_synthesis_model,
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


def test_get_embedding_keep_alive_default():
    assert get_embedding_keep_alive() == "30m"


def test_get_embedding_keep_alive_set(monkeypatch):
    monkeypatch.setenv("KB_OLLAMA_KEEP_ALIVE", "10m")
    assert get_embedding_keep_alive() == "10m"


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


# -- Anthropic model and thinking defaults --


def test_get_anthropic_model_default(monkeypatch):
    """Env unset → the new extraction/query model default."""
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    assert get_anthropic_model() == "claude-haiku-5-5"


def test_get_anthropic_thinking_unset(monkeypatch):
    monkeypatch.delenv("KB_ANTHROPIC_THINKING", raising=False)
    assert get_anthropic_thinking() == "disabled"


def test_get_anthropic_thinking_empty(monkeypatch):
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "")
    assert get_anthropic_thinking() == "disabled"


def test_get_anthropic_thinking_disabled(monkeypatch):
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "disabled")
    assert get_anthropic_thinking() == "disabled"


def test_get_anthropic_thinking_adaptive(monkeypatch):
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    assert get_anthropic_thinking() == "adaptive"


def test_get_anthropic_thinking_default_sends_none(monkeypatch):
    """The `default` sentinel maps to None, so no thinking field is sent."""
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "default")
    assert get_anthropic_thinking() is None


def test_get_anthropic_thinking_normalizes_case(monkeypatch):
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", " Adaptive ")
    assert get_anthropic_thinking() == "adaptive"


def test_get_anthropic_thinking_invalid_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "bogus")
    with caplog.at_level(logging.WARNING, logger="personal_kb.config"):
        assert get_anthropic_thinking() == "disabled"
    hits = [r for r in caplog.records if r.levelno == logging.WARNING and "bogus" in r.getMessage()]
    assert len(hits) == 1


def test_build_anthropic_config_ignores_thinking_env(monkeypatch):
    """The builder never reads KB_ANTHROPIC_THINKING itself."""
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    assert build_anthropic_config().thinking is None


def test_build_anthropic_config_ignores_synthesis_env(monkeypatch):
    """Called with no arguments the builder reads neither synthesis variable."""
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "medium")
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    config = build_anthropic_config()
    assert config.model == "claude-haiku-5-5"
    assert config.effort is None
    assert config.thinking is None


# -- Synthesis model and effort getters --


def test_get_synthesis_model_unset(monkeypatch):
    monkeypatch.delenv("KB_SYNTHESIS_MODEL", raising=False)
    assert get_synthesis_model() == "claude-sonnet-5-5"


def test_get_synthesis_model_empty(monkeypatch):
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", "")
    assert get_synthesis_model() == "claude-sonnet-5-5"


def test_get_synthesis_model_verbatim(monkeypatch):
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", " claude-sonnet-4-6 ")
    assert get_synthesis_model() == "claude-sonnet-4-6"


def test_get_synthesis_effort_unset(monkeypatch):
    monkeypatch.delenv("KB_SYNTHESIS_EFFORT", raising=False)
    assert get_synthesis_effort() == "low"


def test_get_synthesis_effort_empty(monkeypatch):
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "")
    assert get_synthesis_effort() == "low"


def test_get_synthesis_effort_valid_values(monkeypatch):
    for value in ("low", "medium", "high", "xhigh", "max"):
        monkeypatch.setenv("KB_SYNTHESIS_EFFORT", value)
        assert get_synthesis_effort() == value


def test_get_synthesis_effort_normalizes(monkeypatch):
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", " MEDIUM ")
    assert get_synthesis_effort() == "medium"


def test_get_synthesis_effort_default_sends_none(monkeypatch):
    """The `default` sentinel maps to None, so no effort field is sent."""
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "default")
    assert get_synthesis_effort() is None


def test_get_synthesis_effort_invalid_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "bogus")
    with caplog.at_level(logging.WARNING, logger="personal_kb.config"):
        assert get_synthesis_effort() == "low"
    hits = [r for r in caplog.records if r.levelno == logging.WARNING and "bogus" in r.getMessage()]
    assert len(hits) == 1
