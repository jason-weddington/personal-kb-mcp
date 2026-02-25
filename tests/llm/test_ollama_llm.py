"""Tests for OllamaLLMClient (mocked HTTP)."""

import httpx
import pytest

from personal_kb.llm.ollama import OllamaLLMClient


@pytest.fixture
def mock_transport():
    """Create a mock transport for httpx."""
    return httpx.MockTransport(lambda req: _default_handler(req))


def _default_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": []})
    if request.url.path == "/api/generate":
        return httpx.Response(200, json={"response": "test output"})
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_generate_success():
    transport = httpx.MockTransport(_default_handler)
    client = OllamaLLMClient(http_client=httpx.AsyncClient(transport=transport))
    try:
        result = await client.generate("hello")
        assert result == "test output"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generate_unavailable():
    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(fail_handler)
    client = OllamaLLMClient(http_client=httpx.AsyncClient(transport=transport))
    try:
        result = await client.generate("hello")
        assert result is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generate_with_system_prompt():
    """Verify system prompt is included in the request payload."""
    captured: list[dict[str, object]] = []

    def capture_handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"response": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(capture_handler)
    client = OllamaLLMClient(http_client=httpx.AsyncClient(transport=transport))
    try:
        await client.generate("hello", system="be helpful")
        assert len(captured) == 1
        assert captured[0]["system"] == "be helpful"
        assert captured[0]["prompt"] == "hello"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_availability_caching():
    """Success is cached; failure resets so next call retries."""
    call_count = 0

    def counting_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/api/tags":
            call_count += 1
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"response": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(counting_handler)
    client = OllamaLLMClient(http_client=httpx.AsyncClient(transport=transport))
    try:
        # First call checks availability
        assert await client.is_available() is True
        assert call_count == 1

        # Second call uses cache
        assert await client.is_available() is True
        assert call_count == 1  # No additional call

        # Simulate failure by resetting availability
        client._available = None
        assert await client.is_available() is True
        assert call_count == 2  # Retried
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_cleans_up():
    transport = httpx.MockTransport(_default_handler)
    client = OllamaLLMClient(http_client=httpx.AsyncClient(transport=transport))
    assert client._http is not None
    await client.close()
    assert client._http is None
