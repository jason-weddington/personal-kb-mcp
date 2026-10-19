"""Tests for the Bedrock client's cache points, per-attempt timeout and call recording.

Every test is hermetic: ``BedrockLLMClient._get_client`` returns a fake whose
``converse`` is an ``AsyncMock`` or an async function. No test reaches AWS.
The SDK is imported at module top with no skip marker: it is in the workspace
dev group, so a missing SDK fails loudly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import aws_sdk_bedrock_runtime.models as br_models
import pytest
import smithy_json
from smithy_core.exceptions import SmithyIdentityError

from kb_core.config import BedrockProviderConfig
from kb_core.llm import bedrock as bedrock_mod
from kb_core.llm.bedrock import BedrockLLMClient

if TYPE_CHECKING:
    from kb_core.llm.provider import LLMCallRecord, LLMPurpose

_MODEL = "us.anthropic.claude-sonnet-5-5"
_LLAMA = "meta.llama3-70b-instruct-v1:0"
_ARN = "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc123"
_ONE_HOUR: frozenset[LLMPurpose] = frozenset({"surprise_distiller"})


def _usage(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "input_tokens": 11,
        "output_tokens": 12,
        "cache_write_input_tokens": 13,
        "cache_read_input_tokens": 15,
        "cache_details": [SimpleNamespace(ttl="1h", input_tokens=14)],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _response(text: Any = "hello", **usage: Any) -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(value=SimpleNamespace(content=[SimpleNamespace(value=text)])),
        stop_reason="end_turn",
        usage=_usage(**usage),
    )


def _reasoning_only() -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(value=SimpleNamespace(content=[SimpleNamespace(other="x")])),
        stop_reason="end_turn",
        usage=_usage(),
    )


def _client(
    model: str = _MODEL,
    *,
    converse: Any = None,
    recorder: Any = None,
    one_hour: frozenset[LLMPurpose] = frozenset(),
    timeout: float = 60.0,
) -> tuple[BedrockLLMClient, SimpleNamespace]:
    llm = BedrockLLMClient(
        BedrockProviderConfig(model=model, timeout=timeout),
        recorder=recorder,
        system_cache_1h_purposes=one_hour,
    )
    fake = SimpleNamespace(converse=converse or AsyncMock(return_value=_response()))
    llm._get_client = lambda: fake  # type: ignore[method-assign]
    return llm, fake


def _sent(fake: SimpleNamespace) -> Any:
    return fake.converse.call_args[0][0]


@pytest.fixture
def no_sleep() -> Any:
    with patch("kb_core.llm.bedrock.asyncio.sleep", new_callable=AsyncMock) as sleep:
        yield sleep


# --- AC-2: constructor ---


def test_constructor_defaults() -> None:
    llm = BedrockLLMClient(BedrockProviderConfig())
    assert llm._recorder is None
    assert llm._system_cache_1h_purposes == frozenset()


def test_constructor_stores_recorder_and_purposes() -> None:
    rec = Mock()
    llm = BedrockLLMClient(
        BedrockProviderConfig(), recorder=rec, system_cache_1h_purposes=_ONE_HOUR
    )
    assert llm._recorder is rec
    assert llm._system_cache_1h_purposes == _ONE_HOUR


# --- AC-3 / AC-4: system blocks and cache points ---


@pytest.mark.parametrize(
    ("purpose", "one_hour", "expected"),
    [
        (
            "surprise_distiller",
            _ONE_HOUR,
            b'"system":[{"text":"S"},{"cachePoint":{"type":"default","ttl":"1h"}}]',
        ),
        ("planner", _ONE_HOUR, b'"system":[{"text":"S"},{"cachePoint":{"type":"default"}}]'),
    ],
)
async def test_claude_model_sends_system_cache_point(
    purpose: LLMPurpose, one_hour: frozenset[LLMPurpose], expected: bytes
) -> None:
    llm, fake = _client(one_hour=one_hour)
    assert await llm.generate("p", purpose=purpose, system="S") == "hello"
    converse_input = _sent(fake)
    payload = smithy_json.JSONCodec().serialize(converse_input)
    assert expected in payload
    _, wire = llm._system_blocks("S", purpose)
    assert json.loads(payload)["system"] == wire


@pytest.mark.parametrize("model", [_LLAMA, _ARN])
async def test_non_claude_model_sends_bare_system(model: str) -> None:
    llm, fake = _client(model)
    await llm.generate("p", purpose="planner", system="S")
    converse_input = _sent(fake)
    assert converse_input.system == [br_models.SystemContentBlockText(value="S")]
    blocks, wire = llm._system_blocks("S", "planner")
    assert wire == [{"text": "S"}]
    assert blocks == [br_models.SystemContentBlockText(value="S")]
    payload = smithy_json.JSONCodec().serialize(converse_input)
    assert json.loads(payload)["system"] == wire


async def test_no_system_sends_no_system() -> None:
    llm, fake = _client()
    await llm.generate("p", purpose="planner")
    assert _sent(fake).system is None


async def test_generate_chat_has_system_cache_point_and_no_message_cache_point() -> None:
    llm, fake = _client(one_hour=_ONE_HOUR)
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    await llm.generate_chat(
        [*msgs, {"role": "user", "content": "c"}], purpose="surprise_distiller", system="S"
    )
    converse_input = _sent(fake)
    assert converse_input.system[1].value.ttl == "1h"
    for message in converse_input.messages:
        for block in message.content:
            assert not isinstance(block, br_models.ContentBlockCachePoint)


def test_additional_fields_dict() -> None:
    plain = BedrockLLMClient(BedrockProviderConfig())
    assert plain._additional_fields_dict() is None
    assert plain._additional_fields() is None
    both = BedrockLLMClient(BedrockProviderConfig(thinking="adaptive", effort="high"))
    assert both._additional_fields_dict() == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert both._additional_fields().as_value() == both._additional_fields_dict()


# --- AC-5: per-attempt timeout ---


async def test_timeout_is_a_failed_attempt(no_sleep: AsyncMock) -> None:
    entered = 0

    async def hang(_ci: Any) -> Any:
        nonlocal entered
        entered += 1
        await asyncio.Event().wait()

    records: list[LLMCallRecord] = []
    llm, _ = _client(converse=hang, recorder=records.append, timeout=0.01)
    assert await llm.generate("p", purpose="planner") is None
    assert entered == 4
    assert len(records) == 4
    assert {r.outcome for r in records} == {"error"}
    assert {r.error_type for r in records} == {"TimeoutError"}


# --- AC-6: field mapping ---


async def test_record_field_mapping() -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(recorder=records.append)
    assert await llm.generate("p", purpose="planner", system="S") == "hello"
    (r,) = records
    assert (
        r.input_tokens,
        r.output_tokens,
        r.cache_creation_input_tokens,
        r.cache_creation_1h_input_tokens,
        r.cache_read_input_tokens,
    ) == (11, 12, 13, 14, 15)
    assert r.purpose == "planner"
    assert r.model == _MODEL
    assert r.outcome == "ok"
    assert r.error_type is None
    assert r.response_id is None
    assert r.stop_reason == "end_turn"
    assert r.diagnostics_state == "off_config"
    assert r.previous_message_id is None
    assert r.cache_miss_reason is None
    assert r.cache_missed_input_tokens is None
    assert r.diagnostics_raw is None
    assert r.system_sha256 == "1edf84da6dca82f6"
    assert r.latency_ms >= 0
    assert r.thinking is None
    assert r.effort is None
    assert r.candidate_id is None
    assert "T" in r.ts


@pytest.mark.parametrize(
    ("model", "purpose", "digest"),
    [
        (_MODEL, "planner", "1edf84da6dca82f6"),
        (_MODEL, "surprise_distiller", "e44e44c0e0311ff9"),
        (_LLAMA, "planner", "bfe235a05e939566"),
    ],
)
async def test_system_sha256_pinned(model: str, purpose: LLMPurpose, digest: str) -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(model, recorder=records.append, one_hour=_ONE_HOUR)
    await llm.generate("p", purpose=purpose, system="S")
    assert records[0].system_sha256 == digest


async def test_no_system_hash_is_none() -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(recorder=records.append)
    await llm.generate("p", purpose="planner")
    assert records[0].system_sha256 is None


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (None, None),
        ([SimpleNamespace(ttl="5m", input_tokens=9)], 0),
        ([SimpleNamespace(ttl="1h", input_tokens=True), SimpleNamespace(ttl="1h")], 0),
        (
            [SimpleNamespace(ttl="1h", input_tokens=3), SimpleNamespace(ttl="1h", input_tokens=4)],
            7,
        ),
    ],
)
async def test_cache_details_1h(details: Any, expected: int | None) -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(
        converse=AsyncMock(return_value=_response(cache_details=details)),
        recorder=records.append,
    )
    await llm.generate("p", purpose="planner")
    assert records[0].cache_creation_1h_input_tokens == expected


async def test_non_int_tokens_are_none() -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(
        converse=AsyncMock(return_value=_response(input_tokens=True, output_tokens="3")),
        recorder=records.append,
    )
    await llm.generate("p", purpose="planner")
    assert records[0].input_tokens is None
    assert records[0].output_tokens is None


async def test_response_without_usage() -> None:
    records: list[LLMCallRecord] = []
    response = SimpleNamespace(
        output=SimpleNamespace(value=SimpleNamespace(content=[SimpleNamespace(value="hello")])),
        stop_reason="end_turn",
    )
    llm, _ = _client(converse=AsyncMock(return_value=response), recorder=records.append)
    assert await llm.generate("p", purpose="planner") == "hello"
    (r,) = records
    assert r.outcome == "ok"
    assert (
        r.input_tokens,
        r.output_tokens,
        r.cache_creation_input_tokens,
        r.cache_creation_1h_input_tokens,
        r.cache_read_input_tokens,
    ) == (None, None, None, None, None)


async def test_thinking_recorded_from_sent_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[LLMCallRecord] = []
    llm, fake = _client(recorder=records.append)
    assert llm._config.thinking is None
    monkeypatch.setattr(llm, "_additional_fields_dict", lambda: {"thinking": {"type": "adaptive"}})
    await llm.generate("p", purpose="planner")
    assert records[0].thinking == "adaptive"
    assert records[0].effort is None
    assert _sent(fake).additional_model_request_fields.as_value() == {
        "thinking": {"type": "adaptive"}
    }


async def test_effort_recorded() -> None:
    records: list[LLMCallRecord] = []
    llm = BedrockLLMClient(
        BedrockProviderConfig(model=_MODEL, effort="medium"), recorder=records.append
    )
    fake = SimpleNamespace(converse=AsyncMock(return_value=_response()))
    llm._get_client = lambda: fake  # type: ignore[method-assign]
    await llm.generate("p", purpose="planner")
    assert records[0].effort == "medium"


# --- AC-7: outcomes and retries ---


async def test_error_then_ok(no_sleep: AsyncMock) -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(
        converse=AsyncMock(side_effect=[Exception("throttled"), _response()]),
        recorder=records.append,
    )
    assert await llm.generate("p", purpose="planner") == "hello"
    assert [r.outcome for r in records] == ["error", "ok"]
    assert records[0].error_type == "Exception"
    assert records[0].stop_reason is None
    assert records[0].input_tokens is None
    assert llm._available is True
    no_sleep.assert_awaited_once_with(1.0)


async def test_reasoning_only_is_no_text(no_sleep: AsyncMock) -> None:
    records: list[LLMCallRecord] = []
    llm, fake = _client(converse=AsyncMock(return_value=_reasoning_only()), recorder=records.append)
    assert await llm.generate("p", purpose="planner") is None
    assert fake.converse.await_count == 4
    assert [r.outcome for r in records] == ["no_text"] * 4
    assert [c.args[0] for c in no_sleep.await_args_list] == [1.0, 2.0, 4.0]


async def test_unreadable_response_is_no_text(no_sleep: AsyncMock) -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(converse=AsyncMock(return_value=SimpleNamespace()), recorder=records.append)
    assert await llm.generate("p", purpose="planner") is None
    assert [r.outcome for r in records] == ["no_text"] * 4


async def test_identity_error_recorded(no_sleep: AsyncMock) -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(
        converse=AsyncMock(side_effect=SmithyIdentityError("x")), recorder=records.append
    )
    assert await llm.generate("p", purpose="planner") is None
    assert len(records) == 4
    assert {r.error_type for r in records} == {"SmithyIdentityError"}


async def test_cancelled_records_once_and_propagates(no_sleep: AsyncMock) -> None:
    records: list[LLMCallRecord] = []
    llm, _ = _client(
        converse=AsyncMock(side_effect=asyncio.CancelledError()), recorder=records.append
    )
    with pytest.raises(asyncio.CancelledError):
        await llm.generate("p", purpose="planner")
    assert len(records) == 1
    assert records[0].outcome == "error"
    assert records[0].error_type == "CancelledError"


# --- AC-8: recorder safety and silent paths ---


async def test_recorder_failure_is_logged(
    no_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    rec = Mock(side_effect=RuntimeError("sink down"))
    llm, fake = _client(
        converse=AsyncMock(side_effect=[Exception("throttled"), _response()]), recorder=rec
    )
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.bedrock"):
        assert await llm.generate("p", purpose="planner") == "hello"
    assert fake.converse.await_count == 2
    assert rec.call_count == 2
    needle = "llm-call recorder failed purpose=planner"
    failures = [r for r in caplog.records if needle in r.getMessage()]
    assert len(failures) == 2
    assert all(r.exc_info for r in failures)


async def test_no_recorder_builds_no_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bedrock_mod, "LLMCallRecord", Mock(side_effect=AssertionError))
    llm, _ = _client()
    assert await llm.generate("p", purpose="planner", system="S") == "hello"


async def test_no_client_records_nothing() -> None:
    rec = Mock()
    llm = BedrockLLMClient(BedrockProviderConfig(model=_MODEL), recorder=rec)
    llm._get_client = lambda: None  # type: ignore[method-assign]
    assert await llm.generate("p", purpose="planner") is None
    assert await llm.generate_chat([{"role": "user", "content": "x"}], purpose="planner") is None
    rec.assert_not_called()


# --- AC-9: failure logs name the caller ---


async def test_failure_logs_name_purpose_and_model(
    no_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    llm, _ = _client(converse=AsyncMock(side_effect=Exception("bad model")))
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.bedrock"):
        assert await llm.generate("p", purpose="surprise_detector_shape2") is None
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    needle = (
        "purpose=surprise_detector_shape2 model=us.anthropic.claude-sonnet-5-5 error_type=Exception"
    )
    assert any(needle in m for m in messages)
    assert any("Bedrock generation failed purpose=surprise_detector_shape2" in m for m in messages)


async def test_chat_failure_log_names_purpose(
    no_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    llm, _ = _client(converse=AsyncMock(side_effect=Exception("bad model")))
    with caplog.at_level(logging.WARNING, logger="kb_core.llm.bedrock"):
        assert await llm.generate_chat([{"role": "user", "content": "x"}], purpose="chat") is None
    assert any(
        f"Bedrock chat generation failed purpose=chat model={_MODEL}" in r.getMessage()
        for r in caplog.records
    )
