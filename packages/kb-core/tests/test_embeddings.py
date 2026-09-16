"""Hermetic payload-assertion tests for ``kb_core.search.embeddings.EmbeddingClient``.

Verifies the per-request Ollama ``keep_alive`` field (GTD e6c01c04 — keep the
embedding model warm via per-request keep_alive, not a host-global
``OLLAMA_KEEP_ALIVE`` default) is included in both the ``embed()`` and
``embed_batch()`` request bodies, that the configured value — including an
"empty-ish" string like ``"0"`` — passes through to the request payload
unchanged (never coerced, dropped when falsy, or validated against a
whitelist), and that :class:`~kb_core.config.EmbeddingConfig` defaults
``keep_alive`` to ``"30m"``.

No live Ollama, no network — the ``httpx.AsyncClient`` is a mock double
handed in via the ``http_client=`` constructor kwarg.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from kb_core.config import EmbeddingConfig
from kb_core.search.embeddings import EmbeddingClient


def _mock_http_client(embed_response: dict[str, Any]) -> AsyncMock:
    """Mock httpx.AsyncClient: GET /api/tags succeeds, POST /api/embed returns embed_response."""
    http = AsyncMock()

    tags_resp = MagicMock()
    tags_resp.raise_for_status = MagicMock()
    http.get = AsyncMock(return_value=tags_resp)

    embed_resp = MagicMock()
    embed_resp.raise_for_status = MagicMock()
    embed_resp.json = MagicMock(return_value=embed_response)
    http.post = AsyncMock(return_value=embed_resp)

    return http


# ---------------------------------------------------------------------------
# EmbeddingConfig default
# ---------------------------------------------------------------------------


def test_embedding_config_default_keep_alive_is_30m() -> None:
    """A bare EmbeddingConfig() defaults keep_alive to "30m"."""
    assert EmbeddingConfig().keep_alive == "30m"


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


async def test_embed_includes_configured_keep_alive() -> None:
    config = EmbeddingConfig(keep_alive="15m")
    http = _mock_http_client({"embeddings": [[0.1, 0.2, 0.3]]})
    client = EmbeddingClient(db=MagicMock(), config=config, http_client=http)

    result = await client.embed("probe")

    assert result == [0.1, 0.2, 0.3]
    call_kwargs = http.post.call_args.kwargs
    assert call_kwargs["json"]["keep_alive"] == "15m"
    assert call_kwargs["json"]["model"] == config.model
    assert call_kwargs["json"]["input"] == "probe"


async def test_embed_keep_alive_passthrough_not_coerced() -> None:
    """An "empty-ish" configured value ("0", Ollama's unload-immediately signal)
    reaches the request payload verbatim — never coerced, dropped when falsy,
    or validated against a whitelist.
    """
    config = EmbeddingConfig(keep_alive="0")
    http = _mock_http_client({"embeddings": [[0.1]]})
    client = EmbeddingClient(db=MagicMock(), config=config, http_client=http)

    await client.embed("probe")

    call_kwargs = http.post.call_args.kwargs
    assert call_kwargs["json"]["keep_alive"] == "0"


# ---------------------------------------------------------------------------
# embed_batch()
# ---------------------------------------------------------------------------


async def test_embed_batch_includes_configured_keep_alive() -> None:
    config = EmbeddingConfig(keep_alive="15m")
    http = _mock_http_client({"embeddings": [[0.1], [0.2]]})
    client = EmbeddingClient(db=MagicMock(), config=config, http_client=http)

    result = await client.embed_batch(["a", "b"])

    assert result == [[0.1], [0.2]]
    call_kwargs = http.post.call_args.kwargs
    assert call_kwargs["json"]["keep_alive"] == "15m"
    assert call_kwargs["json"]["model"] == config.model
    assert call_kwargs["json"]["input"] == ["a", "b"]


async def test_embed_batch_keep_alive_passthrough_not_coerced() -> None:
    config = EmbeddingConfig(keep_alive="0")
    http = _mock_http_client({"embeddings": [[0.1], [0.2]]})
    client = EmbeddingClient(db=MagicMock(), config=config, http_client=http)

    await client.embed_batch(["a", "b"])

    call_kwargs = http.post.call_args.kwargs
    assert call_kwargs["json"]["keep_alive"] == "0"
