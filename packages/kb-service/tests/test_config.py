"""Tests for kb_service.config: Anthropic model/thinking getters and role wiring.

Covers the extraction/query model default (claude-haiku-5-5), the
``KB_ANTHROPIC_THINKING`` getter's every branch, the builder's refusal to read
that variable itself (so the surprise capture clients cannot pick it up), and
the ``build_provider_config`` wiring that applies thinking to only the
extraction and query roles.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kb_service.config import (
    build_anthropic_config,
    build_provider_config,
    get_anthropic_model,
    get_anthropic_thinking,
)

if TYPE_CHECKING:
    import pytest

_SONNET = "claude-sonnet-4-6"


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("KB_ANTHROPIC_THINKING", raising=False)


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
        f"synthesis_model={_SONNET} synthesis_thinking=None"
    )
    assert shape in caplog.text
