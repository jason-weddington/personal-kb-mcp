"""Hermetic tests for ``BedrockEmbeddingClient`` — no live AWS calls."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aws_sdk_bedrock_runtime  # noqa: F401 — a missing SDK must fail loudly

import kb_core.llm.bedrock as bedrock_mod
from kb_core.config import BedrockProviderConfig, EmbeddingConfig
from kb_core.search.bedrock_embeddings import TITAN_MAX_INPUT_CHARS, BedrockEmbeddingClient


def _response(vec: list[float]) -> SimpleNamespace:
    return SimpleNamespace(body=json.dumps({"embedding": vec}).encode())


def _setup(monkeypatch, *, dim: int = 1024, runtime: Any = "default", auth: Any = "bearer"):
    fake = MagicMock()
    fake.invoke_model = AsyncMock(side_effect=lambda *_a, **_k: _response([0.5] * dim))
    build = MagicMock(return_value=(fake if runtime == "default" else runtime, auth))
    monkeypatch.setattr(bedrock_mod, "build_runtime_client", build)
    config = EmbeddingConfig(
        provider="bedrock",
        model="amazon.titan-embed-text-v2:0",
        dim=dim,
        bedrock=BedrockProviderConfig(),
    )
    return BedrockEmbeddingClient(MagicMock(), config), fake, build


def _bodies(fake) -> list[dict]:
    return [json.loads(c.args[0].body) for c in fake.invoke_model.call_args_list]


async def test_embed_sends_titan_request(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch)
    vec = await client.embed("hello")
    assert vec == [0.5] * 1024
    fake.invoke_model.assert_awaited_once()
    sent = fake.invoke_model.call_args.args[0]
    assert sent.model_id == "amazon.titan-embed-text-v2:0"
    assert sent.accept == "application/json"
    # Setting content_type makes the SDK send two Content-Type values and break SigV4.
    assert sent.content_type is None
    assert json.loads(sent.body) == {"inputText": "hello", "dimensions": 1024, "normalize": True}


async def test_dimensions_follow_config(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch, dim=256)
    await client.embed("x")
    assert _bodies(fake)[0]["dimensions"] == 256


async def test_long_input_truncated_and_logged(monkeypatch, caplog) -> None:
    client, fake, _ = _setup(monkeypatch)
    text = "a" * 60_000
    with caplog.at_level(logging.INFO, logger="kb_core.search.bedrock_embeddings"):
        await client.embed(text)
    assert _bodies(fake)[0]["inputText"] == text[:TITAN_MAX_INPUT_CHARS]
    assert "Bedrock embedding input truncated from 60000 to 50000 chars" in caplog.text


async def test_wrong_length_vector_returns_none(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch)
    fake.invoke_model = AsyncMock(return_value=_response([0.1] * 512))
    assert await client.embed("x") is None


async def test_embed_batch(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch)
    fake.invoke_model = AsyncMock(
        side_effect=lambda inp, **_k: _response([float(json.loads(inp.body)["inputText"])] * 1024)
    )
    out = await client.embed_batch(["1", "2", "3"])
    assert out is not None
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]
    assert fake.invoke_model.await_count == 3


async def test_embed_batch_empty(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch)
    assert await client.embed_batch([]) == []
    fake.invoke_model.assert_not_awaited()


async def test_embed_batch_failure_returns_none(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch)
    fake.invoke_model = AsyncMock(
        side_effect=[_response([0.1] * 1024), RuntimeError("boom"), _response([0.1] * 1024)]
    )
    assert await client.embed_batch(["a", "b", "c"]) is None


async def test_no_credentials(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch, auth=None)
    assert await client.is_available() is False
    assert await client.embed("x") is None
    fake.invoke_model.assert_not_awaited()


async def test_sdk_missing(monkeypatch) -> None:
    client, _, _ = _setup(monkeypatch, runtime=None, auth=None)
    assert await client.is_available() is False


async def test_invalid_dimension_unavailable(monkeypatch) -> None:
    client, _, build = _setup(monkeypatch, dim=768)
    assert await client.is_available() is False
    build.assert_not_called()


async def test_invoke_failure_returns_none(monkeypatch) -> None:
    client, fake, _ = _setup(monkeypatch)
    fake.invoke_model = AsyncMock(side_effect=RuntimeError("boom"))
    assert await client.embed("x") is None


async def test_enabled_logged_once(monkeypatch, caplog) -> None:
    client, _, _ = _setup(monkeypatch)
    with caplog.at_level(logging.INFO, logger="kb_core.search.bedrock_embeddings"):
        assert await client.is_available()
        assert await client.is_available()
    hits = [
        r for r in caplog.records if "Bedrock embeddings enabled (auth=bearer" in r.getMessage()
    ]
    assert len(hits) == 1
