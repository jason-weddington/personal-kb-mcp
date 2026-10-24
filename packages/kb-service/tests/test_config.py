"""Tests for kb_service.config: Anthropic model/thinking getters and role wiring.

Covers the extraction/query model default (claude-haiku-5-5), the
``KB_ANTHROPIC_THINKING`` getter's every branch, the builder's refusal to read
that variable itself (so the surprise capture clients cannot pick it up), and
the ``build_provider_config`` wiring that applies thinking to only the
extraction and query roles.
"""

from __future__ import annotations

import logging

import pytest

import kb_service.config as _cfg
from kb_service.config import (
    build_anthropic_config,
    build_provider_config,
    get_anthropic_model,
    get_anthropic_thinking,
    get_bedrock_model,
    get_bedrock_synthesis_model,
    get_synthesis_effort,
    get_synthesis_model,
)

_SONNET = "claude-sonnet-5-5"


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("KB_ANTHROPIC_THINKING", raising=False)
    monkeypatch.delenv("KB_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("KB_SYNTHESIS_EFFORT", raising=False)
    monkeypatch.delenv("KB_BEDROCK_MODEL", raising=False)
    monkeypatch.delenv("KB_BEDROCK_SYNTHESIS_MODEL", raising=False)


# --- getters --------------------------------------------------------------------


def test_get_anthropic_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    assert get_anthropic_model() == "claude-haiku-5-5"


def test_get_anthropic_thinking_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    assert get_anthropic_thinking() == "disabled"


def test_get_anthropic_thinking_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "")
    assert get_anthropic_thinking() == "disabled"


def test_get_anthropic_thinking_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "disabled")
    assert get_anthropic_thinking() == "disabled"


def test_get_anthropic_thinking_adaptive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    assert get_anthropic_thinking() == "adaptive"


def test_get_anthropic_thinking_default_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `default` sentinel maps to None here, never reaching the dataclass."""
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "default")
    assert get_anthropic_thinking() is None


def test_get_anthropic_thinking_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", " Adaptive ")
    assert get_anthropic_thinking() == "adaptive"


def test_get_anthropic_thinking_invalid_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "bogus")
    with caplog.at_level(logging.WARNING, logger="kb_service.config"):
        assert get_anthropic_thinking() == "disabled"
    hits = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "bogus" in r.getMessage()
    ]
    assert len(hits) == 1


# --- build_anthropic_config isolation -------------------------------------------


def test_build_anthropic_config_ignores_thinking_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder reads no KB_ANTHROPIC_THINKING; capture clients stay silent."""
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    assert build_anthropic_config().thinking is None
    assert build_anthropic_config(model="claude-sonnet-5-5").thinking is None


def test_build_anthropic_config_ignores_synthesis_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Called with no arguments the builder reads neither synthesis variable."""
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "medium")
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    config = build_anthropic_config()
    assert config.model == "claude-haiku-5-5"
    assert config.effort is None
    assert config.thinking is None


# --- synthesis model / effort getters --------------------------------------------


def test_get_synthesis_model_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    assert get_synthesis_model() == "claude-sonnet-5-5"


def test_get_synthesis_model_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", "")
    assert get_synthesis_model() == "claude-sonnet-5-5"


def test_get_synthesis_model_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any non-empty value is used verbatim, whitespace stripped, no validation."""
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", " claude-sonnet-4-6 ")
    assert get_synthesis_model() == "claude-sonnet-4-6"


def test_get_synthesis_effort_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    assert get_synthesis_effort() == "low"


def test_get_synthesis_effort_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "")
    assert get_synthesis_effort() == "low"


def test_get_synthesis_effort_valid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("low", "medium", "high", "xhigh", "max"):
        monkeypatch.setenv("KB_SYNTHESIS_EFFORT", value)
        assert get_synthesis_effort() == value


def test_get_synthesis_effort_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", " MEDIUM ")
    assert get_synthesis_effort() == "medium"


def test_get_synthesis_effort_default_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `default` sentinel maps to None: no effort field is sent at all."""
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "default")
    assert get_synthesis_effort() is None


def test_get_synthesis_effort_invalid_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unknown value warns once per call naming the raw value, falls back low."""
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "bogus")
    with caplog.at_level(logging.WARNING, logger="kb_service.config"):
        assert get_synthesis_effort() == "low"
    hits = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "bogus" in r.getMessage()
    ]
    assert len(hits) == 1


# --- build_provider_config wiring ------------------------------------------------


def test_build_provider_config_thinking_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    cfg = build_provider_config()
    assert cfg.extraction.anthropic.thinking == "disabled"
    assert cfg.query.anthropic.thinking == "disabled"
    assert cfg.extraction.anthropic.model == "claude-haiku-5-5"
    assert cfg.query.anthropic.model == "claude-haiku-5-5"
    assert cfg.synthesis.anthropic.thinking is None
    assert cfg.synthesis.anthropic.model == _SONNET
    assert cfg.synthesis.anthropic.effort == "low"


def test_build_provider_config_synthesis_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KB_SYNTHESIS_MODEL/KB_SYNTHESIS_EFFORT reach only the synthesis role."""
    monkeypatch.setenv("KB_SYNTHESIS_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "medium")
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("KB_ANTHROPIC_THINKING", raising=False)
    cfg = build_provider_config()
    assert cfg.synthesis.anthropic.model == "claude-sonnet-4-6"
    assert cfg.synthesis.anthropic.effort == "medium"
    # Extraction and query stay on the KB_ANTHROPIC_MODEL default, thinking
    # disabled, no effort field.
    assert cfg.extraction.anthropic.model == "claude-haiku-5-5"
    assert cfg.query.anthropic.model == "claude-haiku-5-5"
    assert cfg.extraction.anthropic.thinking == "disabled"
    assert cfg.query.anthropic.thinking == "disabled"
    assert cfg.extraction.anthropic.effort is None
    assert cfg.query.anthropic.effort is None


def test_build_provider_config_thinking_adaptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    cfg = build_provider_config()
    assert cfg.extraction.anthropic.thinking == "adaptive"
    assert cfg.query.anthropic.thinking == "adaptive"
    assert cfg.synthesis.anthropic.thinking is None
    assert cfg.synthesis.anthropic.model == _SONNET


def test_build_provider_config_thinking_default_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "default")
    cfg = build_provider_config()
    assert cfg.extraction.anthropic.thinking is None
    assert cfg.query.anthropic.thinking is None
    assert cfg.synthesis.anthropic.thinking is None


def test_build_provider_config_invalid_thinking_warns_per_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One warning per call, falling back to disabled without raising."""
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "bogus")
    with caplog.at_level(logging.WARNING, logger="kb_service.config"):
        cfg = build_provider_config()
        assert cfg.extraction.anthropic.thinking == "disabled"
        assert cfg.query.anthropic.thinking == "disabled"
        assert cfg.synthesis.anthropic.thinking is None
        first = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "bogus" in r.getMessage()
        ]
        assert len(first) == 1
        build_provider_config()
        second = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "bogus" in r.getMessage()
        ]
        assert len(second) == 2


def test_build_provider_config_logs_llm_roles(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The one INFO decision line lands in kb-service startup logs."""
    _clear_env(monkeypatch)
    with caplog.at_level(logging.INFO, logger="kb_service.config"):
        build_provider_config()
    shape = (
        "llm roles extraction_model=claude-haiku-5-5 extraction_thinking=disabled "
        "query_model=claude-haiku-5-5 query_thinking=disabled "
        f"synthesis_model={_SONNET} synthesis_thinking=None synthesis_effort=low"
    )
    assert shape in caplog.text


# --- Bedrock parity -----------------------------------------------------------


def test_bedrock_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bedrock defaults are the 5.5 US inference profiles, as on the Anthropic path."""
    _clear_env(monkeypatch)
    assert get_bedrock_model() == "us.anthropic.claude-haiku-5-5"
    assert get_bedrock_synthesis_model() == "us.anthropic.claude-sonnet-5-5"


def test_bedrock_synthesis_model_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_BEDROCK_SYNTHESIS_MODEL", " us.anthropic.claude-sonnet-4-6 ")
    assert get_bedrock_synthesis_model() == "us.anthropic.claude-sonnet-4-6"


def test_build_provider_config_bedrock_mirrors_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock roles get the same thinking and effort settings as Anthropic."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "bedrock")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "bedrock")
    cfg = build_provider_config()
    for role in (cfg.extraction, cfg.query):
        assert role.provider == "bedrock"
        assert role.bedrock.model == "us.anthropic.claude-haiku-5-5"
        assert role.bedrock.thinking == "disabled"
        assert role.bedrock.effort is None
    assert cfg.synthesis.provider == "bedrock"
    assert cfg.synthesis.bedrock.model == "us.anthropic.claude-sonnet-5-5"
    assert cfg.synthesis.bedrock.effort == "low"
    assert cfg.synthesis.bedrock.thinking is None


def test_build_provider_config_bedrock_default_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default sentinels send neither field, e.g. for a non-Claude model."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "default")
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "default")
    cfg = build_provider_config()
    assert cfg.query.bedrock.thinking is None
    assert cfg.synthesis.bedrock.effort is None


_EMB_VARS = (
    "KB_EMBEDDING_PROVIDER",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_DIM",
    "KB_OLLAMA_TIMEOUT",
    "KB_BEDROCK_TIMEOUT",
    "KB_BEDROCK_REGION",
    "KB_AWS_PROFILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
)


def _clean_embedding_env(monkeypatch):
    for var in _EMB_VARS:
        monkeypatch.delenv(var, raising=False)


def test_embedding_provider_unset_is_ollama(monkeypatch):
    _clean_embedding_env(monkeypatch)
    assert _cfg.get_embedding_provider() == "ollama"


def test_embedding_provider_empty_is_ollama(monkeypatch):
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "  ")
    assert _cfg.get_embedding_provider() == "ollama"


def test_embedding_provider_normalizes(monkeypatch):
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "BEDROCK")
    assert _cfg.get_embedding_provider() == "bedrock"


def test_embedding_provider_rejects_anthropic(monkeypatch):
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "anthropic")
    with pytest.raises(ValueError):
        _cfg.get_embedding_provider()


def test_embedding_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "foo")
    with pytest.raises(
        ValueError,
        match=r"KB_EMBEDDING_PROVIDER='foo' is not valid. Choose from: bedrock, ollama",
    ):
        _cfg.get_embedding_provider()


def test_embedding_model_defaults_by_provider(monkeypatch):
    _clean_embedding_env(monkeypatch)
    assert _cfg.get_embedding_model() == "qwen3-embedding:0.6b"
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "bedrock")
    assert _cfg.get_embedding_model() == "amazon.titan-embed-text-v2:0"
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("KB_EMBEDDING_MODEL", "foo")
    assert _cfg.get_embedding_model() == "foo"


def test_build_embedding_config_ollama(monkeypatch):
    _clean_embedding_env(monkeypatch)
    cfg = _cfg.build_embedding_config()
    assert cfg.provider == "ollama"
    assert cfg.bedrock is None
    assert cfg.timeout == 10.0
    assert cfg.model == "qwen3-embedding:0.6b"
    assert cfg.dim == 1024


def test_build_embedding_config_bedrock(monkeypatch):
    _clean_embedding_env(monkeypatch)
    monkeypatch.setenv("KB_EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setenv("KB_BEDROCK_REGION", "eu-west-1")
    monkeypatch.setenv("KB_BEDROCK_TIMEOUT", "7")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    cfg = _cfg.build_embedding_config()
    assert cfg.provider == "bedrock"
    assert cfg.model == "amazon.titan-embed-text-v2:0"
    assert cfg.timeout == 7.0
    assert cfg.bedrock is not None
    assert cfg.bedrock.region == "eu-west-1"
    assert cfg.bedrock.bearer_token == "tok"  # noqa: S105
