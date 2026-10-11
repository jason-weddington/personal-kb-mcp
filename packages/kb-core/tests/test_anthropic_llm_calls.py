"""Tests for the Anthropic call recorder: purpose, usage, cache diagnostics, 400 fallback.

Every test is hermetic: the SDK client is replaced by a stub whose
``messages.create`` is an ``AsyncMock`` returning ``SimpleNamespace`` responses.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import typing
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_core.config import AnthropicProviderConfig, BedrockProviderConfig, OllamaProviderConfig
from kb_core.llm import anthropic as anthropic_mod
from kb_core.llm.anthropic import AnthropicLLMClient
from kb_core.llm.bedrock import BedrockLLMClient
from kb_core.llm.ollama import OllamaLLMClient
from kb_core.llm.provider import (
    DiagnosticsState,
    LLMCallOutcome,
    LLMCallRecord,
    LLMProvider,
    LLMPurpose,
)

_MODEL = "claude-haiku-4-5"


@pytest.fixture(autouse=True)
def _clear_module_state() -> typing.Iterator[None]:
    anthropic_mod._PREVIOUS_IDS.clear()
    anthropic_mod._UNPARSED_WARNED.clear()
    yield
    anthropic_mod._PREVIOUS_IDS.clear()
    anthropic_mod._UNPARSED_WARNED.clear()


def _usage(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "input_tokens": 11,
        "output_tokens": 12,
        "cache_creation_input_tokens": 13,
        "cache_read_input_tokens": 15,
        "cache_creation": SimpleNamespace(ephemeral_1h_input_tokens=14),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _response(
    text: str | None = "hello",
    *,
    rid: Any = "msg_1",
    stop_reason: Any = "end_turn",
    usage: Any = None,
    model_extra: Any = None,
) -> SimpleNamespace:
    content = (
        [SimpleNamespace(type="text", text=text)]
        if text is not None
        else [SimpleNamespace(type="thinking", thinking="hmm")]
    )
    return SimpleNamespace(
        id=rid,
        stop_reason=stop_reason,
        content=content,
        usage=usage if usage is not None else _usage(),
        model_extra=model_extra if model_extra is not None else {},
    )


def _client(
    *responses: Any,
    recorder: Any = None,
    model: str = _MODEL,
    diagnostics: bool = True,
) -> tuple[AnthropicLLMClient, AsyncMock]:
    llm = AnthropicLLMClient(
        AnthropicProviderConfig(model=model, api_key="k", diagnostics=diagnostics),
        recorder=recorder,
    )
    create = AsyncMock(side_effect=list(responses))
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm, create


class _Records(list[LLMCallRecord]):
    def __call__(self, record: LLMCallRecord) -> None:
        self.append(record)


class _Fake400(Exception):  # noqa: N818 - name fixed by the spec
    status_code = 400


class _Fake500(Exception):  # noqa: N818
    status_code = 500


# --- AC-1 / AC-2: types and signatures ---------------------------------------


def test_literal_members() -> None:
    assert set(typing.get_args(LLMPurpose)) == {
        "agent",
        "chat",
        "classifier",
        "coverage",
        "dedup_agent",
        "enricher",
        "enricher_batch",
        "feedback_summary",
        "ingest_extract",
        "ingest_summarize",
        "listener_vote",
        "planner",
        "query_synthesis",
        "seed_resolutions",
        "store_critic",
        "store_distiller",
        "surprise_critic",
        "surprise_detector_shape2",
        "surprise_detector_shape3",
        "surprise_distiller",
    }
    assert set(typing.get_args(LLMCallOutcome)) == {"ok", "no_text", "error"}
    assert set(typing.get_args(DiagnosticsState)) == {
        "sent",
        "off_config",
        "off_disabled",
        "fallback",
    }


@pytest.mark.parametrize(
    "cls", [LLMProvider, AnthropicLLMClient, BedrockLLMClient, OllamaLLMClient]
)
@pytest.mark.parametrize("method", ["generate", "generate_chat"])
def test_purpose_is_required_keyword(cls: type, method: str) -> None:
    param = inspect.signature(getattr(cls, method)).parameters["purpose"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_record_field_order() -> None:
    import dataclasses

    assert [f.name for f in dataclasses.fields(LLMCallRecord)] == [
        "ts",
        "purpose",
        "model",
        "outcome",
        "error_type",
        "response_id",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_creation_1h_input_tokens",
        "cache_read_input_tokens",
        "diagnostics_state",
        "previous_message_id",
        "cache_miss_reason",
        "cache_missed_input_tokens",
        "diagnostics_raw",
        "system_sha256",
        "latency_ms",
    ]


def test_other_clients_construct() -> None:
    BedrockLLMClient(BedrockProviderConfig())
    OllamaLLMClient(OllamaProviderConfig())


# --- AC-6 / AC-9: no recorder is a plain passthrough -------------------------


async def test_no_recorder_sends_no_extra_body() -> None:
    llm, create = _client(_response())
    assert await llm.generate("p", purpose="planner") == "hello"
    assert "extra_body" not in create.call_args.kwargs
    assert create.call_args.kwargs["timeout"] == llm._config.timeout
    assert anthropic_mod._PREVIOUS_IDS == {}


async def test_no_client_records_nothing() -> None:
    rec = _Records()
    llm = AnthropicLLMClient(AnthropicProviderConfig(), recorder=rec)
    llm._get_client = lambda: None  # type: ignore[method-assign]
    assert await llm.generate("p", purpose="planner") is None
    assert rec == []


async def test_off_config_state() -> None:
    rec = _Records()
    llm, create = _client(_response(), recorder=rec, diagnostics=False)
    await llm.generate("p", purpose="planner")
    assert "extra_body" not in create.call_args.kwargs
    assert rec[0].diagnostics_state == "off_config"
    assert rec[0].previous_message_id is None


async def test_first_call_sends_diagnostics() -> None:
    rec = _Records()
    llm, create = _client(_response(), recorder=rec)
    await llm.generate("p", purpose="planner")
    assert create.call_args.kwargs["extra_body"] == {"diagnostics": {"previous_message_id": None}}
    assert rec[0].diagnostics_state == "sent"


async def test_create_kwargs_not_mutated() -> None:
    rec = _Records()
    llm, _ = _client(_response(), recorder=rec)
    kwargs = {"model": _MODEL, "max_tokens": 5, "messages": []}
    snapshot = dict(kwargs)
    await llm._create(llm._client, kwargs, purpose="planner")
    assert kwargs == snapshot


# --- AC-7: record contents ---------------------------------------------------


async def test_full_text_response_record() -> None:
    rec = _Records()
    llm, _ = _client(_response(), recorder=rec)
    assert await llm.generate("p", purpose="planner", system="sys") == "hello"
    assert len(rec) == 1
    r = rec[0]
    assert r.purpose == "planner"
    assert r.model == _MODEL
    assert r.outcome == "ok"
    assert r.error_type is None
    assert r.response_id == "msg_1"
    assert r.stop_reason == "end_turn"
    assert (
        r.input_tokens,
        r.output_tokens,
        r.cache_creation_input_tokens,
        r.cache_creation_1h_input_tokens,
        r.cache_read_input_tokens,
    ) == (11, 12, 13, 14, 15)
    assert r.diagnostics_state == "sent"
    assert r.previous_message_id is None
    assert r.cache_miss_reason is None
    assert r.cache_missed_input_tokens is None
    assert r.diagnostics_raw is None
    assert isinstance(r.system_sha256, str) and len(r.system_sha256) == 16
    assert isinstance(r.latency_ms, int) and r.latency_ms >= 0
    assert r.ts.endswith("+00:00")


async def test_thinking_only_response_is_no_text() -> None:
    rec = _Records()
    llm, _ = _client(_response(None, stop_reason="max_tokens"), recorder=rec)
    assert await llm.generate("p", purpose="planner") is None
    r = rec[0]
    assert r.outcome == "no_text"
    assert r.stop_reason == "max_tokens"
    assert r.response_id == "msg_1"
    assert r.input_tokens == 11
    assert r.system_sha256 is None


async def test_cache_creation_none_gives_null_1h() -> None:
    rec = _Records()
    llm, _ = _client(_response(usage=_usage(cache_creation=None)), recorder=rec)
    await llm.generate("p", purpose="planner")
    assert rec[0].cache_creation_1h_input_tokens is None
    assert rec[0].input_tokens == 11


async def test_magicmock_usage_gives_null_fields() -> None:
    rec = _Records()
    llm, _ = _client(_response(usage=MagicMock()), recorder=rec)
    await llm.generate("p", purpose="planner")
    r = rec[0]
    assert (
        r.input_tokens,
        r.output_tokens,
        r.cache_creation_input_tokens,
        r.cache_creation_1h_input_tokens,
        r.cache_read_input_tokens,
    ) == (None, None, None, None, None)


async def test_bool_usage_is_not_an_int() -> None:
    rec = _Records()
    llm, _ = _client(_response(usage=_usage(input_tokens=True)), recorder=rec)
    await llm.generate("p", purpose="planner")
    assert rec[0].input_tokens is None


async def test_error_record() -> None:
    rec = _Records()
    llm, _ = _client(ValueError("boom"), recorder=rec)
    assert await llm.generate("p", purpose="coverage", system="s") is None
    r = rec[0]
    assert r.outcome == "error"
    assert r.error_type == "ValueError"
    assert r.response_id is None
    assert r.stop_reason is None
    assert r.input_tokens is None
    assert r.cache_read_input_tokens is None
    assert r.cache_miss_reason is None
    assert r.diagnostics_raw is None
    assert r.system_sha256 is not None
    assert r.diagnostics_state == "sent"


async def test_cancelled_error_records_once_and_reraises() -> None:
    rec = _Records()
    llm, _ = _client(asyncio.CancelledError(), recorder=rec)
    with pytest.raises(asyncio.CancelledError):
        await llm.generate("p", purpose="planner")
    assert len(rec) == 1
    assert rec[0].outcome == "error"
    assert rec[0].error_type == "CancelledError"


async def test_system_sha256_stability() -> None:
    rec = _Records()
    llm, _ = _client(
        _response(rid="a"), _response(rid="b"), _response(rid="c"), _response(rid="d"), recorder=rec
    )
    await llm.generate("p1", purpose="planner", system="S")
    await llm.generate("p2", purpose="planner", system="S")
    await llm.generate("p3", purpose="planner", system="T")
    await llm.generate("p4", purpose="planner")
    assert rec[0].system_sha256 == rec[1].system_sha256
    assert rec[2].system_sha256 != rec[0].system_sha256
    assert rec[3].system_sha256 is None


async def test_non_str_id_and_stop_reason() -> None:
    rec = _Records()
    llm, _ = _client(_response(rid=MagicMock(), stop_reason=None), recorder=rec)
    await llm.generate("p", purpose="planner")
    assert rec[0].response_id is None
    assert rec[0].stop_reason is None
    assert anthropic_mod._PREVIOUS_IDS == {}


# --- AC-8: recorder failure never fails the call ------------------------------


def _raising_recorder(record: LLMCallRecord) -> None:
    raise RuntimeError("recorder down")


@pytest.mark.parametrize("method", ["generate", "generate_chat"])
async def test_recorder_failure_does_not_change_result(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    results = []
    for recorder in (None, _raising_recorder, _Records()):
        llm, _ = _client(_response("same"), recorder=recorder)
        arg: Any = "p" if method == "generate" else [{"role": "user", "content": "p"}]
        with caplog.at_level(logging.WARNING, logger="kb_core.llm.anthropic"):
            results.append(await getattr(llm, method)(arg, purpose="planner"))
    assert results == ["same", "same", "same"]
    assert any("llm-call recorder failed purpose=planner" in r.message for r in caplog.records)


async def test_recorder_failure_does_not_mask_create_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm, _ = _client(ValueError("boom"), recorder=_raising_recorder)
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.anthropic"):
        assert await llm.generate("p", purpose="planner") is None
    assert any("Anthropic generation failed" in r.message for r in caplog.records)


# --- AC-10: previous_message_id chain ----------------------------------------


async def test_chain_per_purpose_and_model() -> None:
    rec = _Records()
    llm, create = _client(
        _response(rid="msg_1"), _response(rid="msg_2"), _response(rid="cov_1"), recorder=rec
    )
    await llm.generate("p", purpose="planner")
    assert create.call_args.kwargs["extra_body"]["diagnostics"]["previous_message_id"] is None
    await llm.generate("p", purpose="planner")
    assert create.call_args.kwargs["extra_body"]["diagnostics"]["previous_message_id"] == "msg_1"
    assert rec[1].previous_message_id == "msg_1"
    await llm.generate("p", purpose="coverage")
    assert create.call_args.kwargs["extra_body"]["diagnostics"]["previous_message_id"] is None

    other, other_create = _client(_response(rid="x"), recorder=_Records())
    await other.generate("p", purpose="planner")
    assert (
        other_create.call_args.kwargs["extra_body"]["diagnostics"]["previous_message_id"] == "msg_2"
    )

    diff, diff_create = _client(_response(rid="y"), recorder=_Records(), model="claude-sonnet-4-6")
    await diff.generate("p", purpose="planner")
    assert diff_create.call_args.kwargs["extra_body"]["diagnostics"]["previous_message_id"] is None


async def test_chain_first_two_calls_on_one_client() -> None:
    llm, create = _client(_response(rid="msg_1"), _response(rid="msg_2"), recorder=_Records())
    await llm.generate("p", purpose="planner")
    await llm.generate("p", purpose="planner")
    sent = [
        c.kwargs["extra_body"]["diagnostics"]["previous_message_id"] for c in create.call_args_list
    ]
    assert sent == [None, "msg_1"]


async def test_no_text_updates_chain_error_does_not() -> None:
    llm, create = _client(
        _response(None, rid="msg_9"), ValueError("x"), _response(rid="msg_10"), recorder=_Records()
    )
    await llm.generate("p", purpose="planner")
    await llm.generate("p", purpose="planner")
    await llm.generate("p", purpose="planner")
    sent = [
        c.kwargs["extra_body"]["diagnostics"]["previous_message_id"] for c in create.call_args_list
    ]
    assert sent == [None, "msg_9", "msg_9"]


async def test_off_config_does_not_update_chain() -> None:
    llm, _ = _client(_response(rid="z"), recorder=_Records(), diagnostics=False)
    await llm.generate("p", purpose="planner")
    assert anthropic_mod._PREVIOUS_IDS == {}


# --- AC-11: diagnostics parsing ----------------------------------------------


async def _parsed(model_extra: Any) -> LLMCallRecord:
    rec = _Records()
    llm, _ = _client(_response(model_extra=model_extra), recorder=rec)
    await llm.generate("p", purpose="planner")
    return rec[0]


async def test_parse_no_diagnostics_key() -> None:
    r = await _parsed({})
    assert (r.cache_miss_reason, r.cache_missed_input_tokens, r.diagnostics_raw) == (
        None,
        None,
        None,
    )


async def test_parse_null_diagnostics() -> None:
    r = await _parsed({"diagnostics": None})
    assert (r.cache_miss_reason, r.cache_missed_input_tokens, r.diagnostics_raw) == (
        None,
        None,
        "null",
    )


@pytest.mark.parametrize("diag", [{"cache_miss_reason": None}, {"cache_miss_reason": None, "x": 1}])
async def test_parse_pending(diag: dict[str, Any]) -> None:
    r = await _parsed({"diagnostics": diag})
    assert r.cache_miss_reason == "pending"
    assert r.cache_missed_input_tokens is None
    assert r.diagnostics_raw is not None


async def test_parse_reason_with_tokens() -> None:
    diag = {"cache_miss_reason": {"type": "system_changed", "cache_missed_input_tokens": 4200}}
    r = await _parsed({"diagnostics": diag})
    assert (r.cache_miss_reason, r.cache_missed_input_tokens) == ("system_changed", 4200)
    assert r.diagnostics_raw == (
        '{"cache_miss_reason":{"type":"system_changed","cache_missed_input_tokens":4200}}'
    )


async def test_parse_reason_without_tokens() -> None:
    r = await _parsed(
        {"diagnostics": {"cache_miss_reason": {"type": "previous_message_not_found"}}}
    )
    assert (r.cache_miss_reason, r.cache_missed_input_tokens) == (
        "previous_message_not_found",
        None,
    )


@pytest.mark.parametrize(
    "diag",
    [
        {"cache_miss_reason": "system_changed"},
        {},
        {"cache_miss_reason": {"tokens": 1}},
    ],
)
async def test_parse_unparsed(diag: dict[str, Any]) -> None:
    r = await _parsed({"diagnostics": diag})
    assert r.cache_miss_reason == "unparsed"
    assert r.cache_missed_input_tokens is None
    assert r.diagnostics_raw is not None


async def test_unparsed_warns_once_per_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.anthropic"):
        await _parsed({"diagnostics": {}})
        await _parsed({"diagnostics": {"cache_miss_reason": "x"}})
    warned = [r for r in caplog.records if "llm-diagnostics unparsed shape" in r.message]
    assert len(warned) == 1


async def test_parse_magicmock_model_extra() -> None:
    r = await _parsed(MagicMock())
    assert (r.cache_miss_reason, r.cache_missed_input_tokens, r.diagnostics_raw) == (
        None,
        None,
        None,
    )


async def test_parse_skipped_when_not_sent() -> None:
    rec = _Records()
    llm, _ = _client(
        _response(model_extra={"diagnostics": {"cache_miss_reason": None}}),
        recorder=rec,
        diagnostics=False,
    )
    await llm.generate("p", purpose="planner")
    assert rec[0].cache_miss_reason is None
    assert rec[0].diagnostics_raw is None


# --- AC-12: 400 fallback -----------------------------------------------------


@pytest.mark.parametrize("method", ["generate", "generate_chat"])
async def test_400_naming_diagnostics_falls_back(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    rec = _Records()
    llm, create = _client(
        _Fake400("Unknown field: diagnostics"),
        _response(rid="r1"),
        _response(rid="r2"),
        recorder=rec,
    )
    arg: Any = "p" if method == "generate" else [{"role": "user", "content": "p"}]
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.anthropic"):
        assert await getattr(llm, method)(arg, purpose="planner") == "hello"
        assert create.call_count == 2
        assert "extra_body" in create.call_args_list[0].kwargs
        assert "extra_body" not in create.call_args_list[1].kwargs
        assert len(rec) == 1
        assert rec[0].diagnostics_state == "fallback"
        assert rec[0].previous_message_id is None
        assert rec[0].outcome == "ok"

        assert await getattr(llm, method)(arg, purpose="planner") == "hello"
    assert create.call_count == 3
    assert "extra_body" not in create.call_args_list[2].kwargs
    assert rec[1].diagnostics_state == "off_disabled"
    assert anthropic_mod._PREVIOUS_IDS == {}
    disabled = [r for r in caplog.records if "llm-diagnostics disabled" in r.message]
    assert len(disabled) == 1
    assert disabled[0].getMessage() == f"llm-diagnostics disabled model={_MODEL} reason=400"
    assert not [r for r in caplog.records if "generation failed" in r.message]


async def test_400_retry_failure_follows_normal_path(caplog: pytest.LogCaptureFixture) -> None:
    rec = _Records()
    llm, create = _client(_Fake400("diagnostics bad"), ValueError("again"), recorder=rec)
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.anthropic"):
        assert await llm.generate("p", purpose="planner") is None
    assert create.call_count == 2
    assert rec[0].outcome == "error"
    assert rec[0].error_type == "ValueError"
    assert rec[0].diagnostics_state == "fallback"
    assert any("Anthropic generation failed" in r.message for r in caplog.records)


async def test_400_without_diagnostics_does_not_retry() -> None:
    rec = _Records()
    llm, create = _client(_Fake400("bad request"), _response(), recorder=rec)
    assert await llm.generate("p", purpose="planner") is None
    assert create.call_count == 1
    assert rec[0].outcome == "error"
    assert rec[0].diagnostics_state == "sent"
    await llm.generate("p", purpose="planner")
    assert "extra_body" in create.call_args.kwargs


async def test_500_does_not_retry() -> None:
    rec = _Records()
    llm, create = _client(_Fake500("diagnostics overloaded"), recorder=rec)
    assert await llm.generate("p", purpose="planner") is None
    assert create.call_count == 1
    assert rec[0].diagnostics_state == "sent"
    assert llm._diagnostics_disabled is False
