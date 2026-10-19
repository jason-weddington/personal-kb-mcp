"""Surprise capture on Bedrock: provider rule, getters, caching, ledger rows.

Hermetic: every Bedrock client gets a fake ``_get_client`` whose ``converse``
is an ``AsyncMock``. No test reaches AWS.
"""

import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from kb_core.config import BedrockProviderConfig
from kb_core.llm.anthropic import AnthropicLLMClient
from kb_core.llm.bedrock import _SONNET_MODEL, BedrockLLMClient

import kb_service.database as database
from kb_service import llm_calls, surprise_worker
from kb_service.db_sqlite import SqlitePool
from kb_service.llm_calls import (
    candidate_scope,
    drain_pending_llm_calls,
    record_llm_call,
)
from kb_service.store_distill import STORE_DISTILLER_SYSTEM
from kb_service.surprise_distill import SURPRISE_DISTILLER_SYSTEM
from kb_service.surprise_worker import detector_model_name

_GETTERS = [
    ("get_detector_llm", "_DETECTOR_LLM_CACHE"),
    ("get_distiller_llm", "_DISTILLER_LLM_CACHE"),
    ("get_critic_llm", "_CRITIC_LLM_CACHE"),
]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_calls, "_PENDING", set())
    monkeypatch.setattr(surprise_worker, "_DETECTOR_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_DISTILLER_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_CRITIC_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_SURPRISE_PROVIDER_WARNED", False)


@pytest.fixture
def bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SURPRISE_PROVIDER", "bedrock")


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.delenv("KB_SERVICE_DATABASE_URL", raising=False)
    monkeypatch.delenv("KB_DATABASE_URL", raising=False)
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(database, "_pool", None)
    return tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(
            value=SimpleNamespace(content=[SimpleNamespace(value="{}")])
        ),
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=12,
            cache_write_input_tokens=13,
            cache_read_input_tokens=15,
            cache_details=[SimpleNamespace(ttl="1h", input_tokens=14)],
        ),
    )


def _fake() -> SimpleNamespace:
    return SimpleNamespace(converse=AsyncMock(return_value=_response()))


def _messages(caplog: pytest.LogCaptureFixture, level: int, needle: str) -> list[str]:
    found = [r.getMessage() for r in caplog.records if r.levelno == level]
    return [m for m in found if needle in m]


# --- AC-12: provider rule -------------------------------------------------------

_UNSET = object()


@pytest.mark.parametrize(
    ("surprise", "query", "expected"),
    [
        (_UNSET, _UNSET, ("anthropic", "default")),
        (_UNSET, "anthropic", ("anthropic", "KB_QUERY_PROVIDER")),
        (_UNSET, "bedrock", ("bedrock", "KB_QUERY_PROVIDER")),
        (_UNSET, " BEDROCK ", ("bedrock", "KB_QUERY_PROVIDER")),
        (_UNSET, "ollama", ("anthropic", "default")),
        (_UNSET, "nonsense", ("anthropic", "default")),
        ("", "bedrock", ("bedrock", "KB_QUERY_PROVIDER")),
        ("   ", _UNSET, ("anthropic", "default")),
        (" Bedrock ", _UNSET, ("bedrock", "KB_SURPRISE_PROVIDER")),
        ("anthropic", "bedrock", ("anthropic", "KB_SURPRISE_PROVIDER")),
        ("bedrock", "anthropic", ("bedrock", "KB_SURPRISE_PROVIDER")),
        ("ollama", "bedrock", ("bedrock", "KB_QUERY_PROVIDER")),
    ],
)
def test_provider_rule(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    surprise: Any,
    query: Any,
    expected: tuple[str, str],
) -> None:
    env = (("KB_SURPRISE_PROVIDER", surprise), ("KB_QUERY_PROVIDER", query))
    for name, value in env:
        if value is _UNSET:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING, logger="kb_service.surprise_worker"):
        assert surprise_worker._resolve_surprise_provider() == expected
        assert surprise_worker.get_surprise_provider() == expected[0]
    if surprise == "   ":
        assert _messages(caplog, logging.WARNING, "surprise_provider bad_value") == []


def test_bad_value_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(surprise_worker, "_SURPRISE_PROVIDER_WARNED", False)
    monkeypatch.setenv("KB_SURPRISE_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "bedrock")
    with caplog.at_level(logging.WARNING, logger="kb_service.surprise_worker"):
        surprise_worker._resolve_surprise_provider()
        surprise_worker._resolve_surprise_provider()
    needle = "surprise_provider bad_value value='ollama'"
    warnings = _messages(caplog, logging.WARNING, needle)
    assert len(warnings) == 1


def test_resolver_does_not_use_config_get_query_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kb_service.config as config

    def boom() -> str:
        raise AssertionError("must not be called")

    monkeypatch.setattr(config, "get_query_provider", boom)
    monkeypatch.setenv("KB_QUERY_PROVIDER", "garbage")
    assert surprise_worker._resolve_surprise_provider() == ("anthropic", "default")


# --- AC-13: getters on Bedrock ----------------------------------------------------


@pytest.mark.parametrize(("getter", "cache"), _GETTERS)
def test_bedrock_getter_defaults(bedrock: None, getter: str, cache: str) -> None:
    llm = getattr(surprise_worker, getter)()
    assert isinstance(llm, BedrockLLMClient)
    assert llm._config.model == _SONNET_MODEL
    assert llm._recorder is record_llm_call
    assert getattr(surprise_worker, getter)() is llm
    assert set(getattr(surprise_worker, cache)) == {("bedrock", _SONNET_MODEL)}


def test_switching_provider_keys_cache(
    bedrock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = surprise_worker.get_detector_llm()
    assert isinstance(first, BedrockLLMClient)
    monkeypatch.setenv("KB_SURPRISE_PROVIDER", "anthropic")
    second = surprise_worker.get_detector_llm()
    assert isinstance(second, AnthropicLLMClient)
    assert second._config.model == "claude-sonnet-5-5"
    assert set(surprise_worker._DETECTOR_LLM_CACHE) == {
        ("bedrock", _SONNET_MODEL),
        ("anthropic", "claude-sonnet-5-5"),
    }


def test_bedrock_model_override(
    bedrock: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_SURPRISE_DETECTOR_MODEL", " us.anthropic.claude-haiku-5-5 ")
    with caplog.at_level(logging.WARNING, logger="kb_service.surprise_worker"):
        llm = surprise_worker.get_detector_llm()
    assert isinstance(llm, BedrockLLMClient)
    assert llm._config.model == "us.anthropic.claude-haiku-5-5"
    assert detector_model_name(llm) == "us.anthropic.claude-haiku-5-5"
    assert _messages(caplog, logging.WARNING, "looks like") == []


@pytest.mark.parametrize("module", ["kb_core.llm.bedrock", "aws_sdk_bedrock_runtime"])
@pytest.mark.parametrize(("getter", "cache"), _GETTERS)
def test_bedrock_import_gates(
    bedrock: None, monkeypatch: pytest.MonkeyPatch, module: str, getter: str, cache: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    assert getattr(surprise_worker, getter)() is None
    assert getattr(surprise_worker, cache) == {}


def test_bedrock_one_hour_purposes(bedrock: None) -> None:
    distiller = surprise_worker.get_distiller_llm()
    detector = surprise_worker.get_detector_llm()
    critic = surprise_worker.get_critic_llm()
    assert isinstance(distiller, BedrockLLMClient)
    assert isinstance(detector, BedrockLLMClient)
    assert isinstance(critic, BedrockLLMClient)
    assert distiller._system_cache_1h_purposes == frozenset({"surprise_distiller"})
    assert detector._system_cache_1h_purposes == frozenset()
    assert critic._system_cache_1h_purposes == frozenset()


def test_anthropic_id_on_bedrock_warns_once(
    bedrock: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CRITIC_MODEL", "claude-opus-5-5")
    with caplog.at_level(logging.WARNING, logger="kb_service.surprise_worker"):
        first = surprise_worker.get_critic_llm()
        second = surprise_worker.get_critic_llm()
    assert isinstance(first, BedrockLLMClient)
    assert first is second
    needle = (
        "surprise capture model=claude-opus-5-5 looks like an Anthropic API id"
        " on provider bedrock"
    )
    assert len(_messages(caplog, logging.WARNING, needle)) == 1


def test_routing_info_line_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("KB_SURPRISE_PROVIDER", raising=False)
    monkeypatch.setenv("KB_QUERY_PROVIDER", "bedrock")
    with caplog.at_level(logging.INFO, logger="kb_service.surprise_worker"):
        surprise_worker.get_detector_llm()
        surprise_worker.get_detector_llm()
    needle = (
        "surprise_llm client stage=detector provider=bedrock source=KB_QUERY_PROVIDER"
        " model=us.anthropic.claude-sonnet-5-5"
    )
    assert len(_messages(caplog, logging.INFO, needle)) == 1


def test_bedrock_id_on_anthropic_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_SURPRISE_PROVIDER", "anthropic")
    monkeypatch.setenv("KB_SURPRISE_DETECTOR_MODEL", "us.anthropic.claude-haiku-5-5")
    with caplog.at_level(logging.WARNING, logger="kb_service.surprise_worker"):
        llm = surprise_worker.get_detector_llm()
    assert isinstance(llm, AnthropicLLMClient)
    needle = "looks like a Bedrock model id on provider anthropic"
    assert len(_messages(caplog, logging.WARNING, needle)) == 1


# --- AC-14: no thinking on Bedrock surprise clients ------------------------------


@pytest.mark.parametrize(("getter", "cache"), _GETTERS)
async def test_bedrock_clients_send_no_thinking(
    bedrock: None, monkeypatch: pytest.MonkeyPatch, getter: str, cache: str
) -> None:
    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "disabled")
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "medium")
    llm = getattr(surprise_worker, getter)()
    assert isinstance(llm, BedrockLLMClient)
    assert llm._config.thinking is None
    assert llm._config.effort is None
    monkeypatch.setattr(llm, "_recorder", None)
    fake = _fake()
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    assert await llm.generate("p", purpose="surprise_critic") == "{}"
    assert fake.converse.call_args[0][0].additional_model_request_fields is None


def test_thinking_tripwire_on_bedrock(
    bedrock: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        surprise_worker,
        "build_bedrock_config",
        lambda *, model=None, **_: BedrockProviderConfig(
            model=model, thinking="adaptive"
        ),
    )
    with caplog.at_level(logging.WARNING, logger="kb_service.surprise_worker"):
        surprise_worker.get_detector_llm()
        surprise_worker.get_detector_llm()
    needle = "surprise capture client built with thinking=adaptive (expected None)"
    assert len(_messages(caplog, logging.WARNING, needle)) == 1


# --- AC-15: distiller TTL end to end ------------------------------------------------


async def test_distiller_system_cache_point_ttl(
    bedrock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = surprise_worker.get_distiller_llm()
    assert isinstance(llm, BedrockLLMClient)
    monkeypatch.setattr(llm, "_recorder", None)
    fake = _fake()
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    await llm.generate(
        "p", purpose="surprise_distiller", system=SURPRISE_DISTILLER_SYSTEM
    )
    await llm.generate("p", purpose="store_distiller", system=STORE_DISTILLER_SYSTEM)
    first = fake.converse.call_args_list[0].args[0]
    second = fake.converse.call_args_list[1].args[0]
    assert first.system[1].value.ttl == "1h"
    assert second.system[1].value.ttl is None
    assert first.system[0].value == SURPRISE_DISTILLER_SYSTEM
    assert second.system[0].value == STORE_DISTILLER_SYSTEM


# --- AC-16: ledger row and candidate attribution ---------------------------------


async def test_bedrock_ledger_row_with_candidate(
    pool: SqlitePool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_PROVIDER", "bedrock")
    monkeypatch.setattr(surprise_worker, "_DISTILLER_LLM_CACHE", {})
    llm = surprise_worker.get_distiller_llm()
    assert isinstance(llm, BedrockLLMClient)
    fake = _fake()
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    with candidate_scope(7):
        await llm.generate(
            "p", purpose="surprise_distiller", system=SURPRISE_DISTILLER_SYSTEM
        )
    await drain_pending_llm_calls()
    rows = await pool.fetch("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["purpose"] == "surprise_distiller"
    assert row["model"] == "us.anthropic.claude-sonnet-5-5"
    assert row["outcome"] == "ok"
    assert row["stop_reason"] == "end_turn"
    assert row["diagnostics_state"] == "off_config"
    assert row["response_id"] is None
    assert row["previous_message_id"] is None
    assert row["input_tokens"] == 11
    assert row["output_tokens"] == 12
    assert row["cache_creation_input_tokens"] == 13
    assert row["cache_creation_1h_input_tokens"] == 14
    assert row["cache_read_input_tokens"] == 15
    sha = row["system_sha256"]
    assert isinstance(sha, str)
    assert len(sha) == 16
    int(sha, 16)
    assert row["thinking"] is None
    assert row["effort"] is None
    assert row["candidate_id"] == 7
