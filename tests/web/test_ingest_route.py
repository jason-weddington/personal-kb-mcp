"""Tests for POST /api/ingest_url and POST /api/ingest/stream routes."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from personal_kb.ingest.ingester import FileResult
from personal_kb.web.app import create_app_with_deps
from tests.conftest import FakeLLM


@pytest.fixture
def ingest_app(web_kb):
    """App with store + extraction_llm so ingest route is available."""
    from personal_kb.graph.builder import GraphBuilder
    from personal_kb.store.knowledge_store import KnowledgeStore

    db, embedder, _ids = web_kb
    store = KnowledgeStore(db)
    builder = GraphBuilder(db)
    llm = FakeLLM()

    return create_app_with_deps(
        db,
        embedder,
        llm,
        store=store,
        graph_builder=builder,
        extraction_llm=llm,
    )


def _patch_ingest(fake_result):
    """Patch FileIngester.ingest_url to return fake_result."""
    return patch(
        "personal_kb.ingest.ingester.FileIngester.ingest_url",
        new_callable=AsyncMock,
        return_value=fake_result,
    )


@pytest.mark.asyncio
async def test_ingest_url_success(ingest_app):
    """Successful URL ingestion returns entry IDs."""
    fake_result = FileResult(
        path="https://example.com/article",
        action="ingested",
        entry_count=2,
        entry_ids=["kb-00010", "kb-00011"],
        summary="An article about testing",
    )

    with _patch_ingest(fake_result):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest_url",
                json={"url": "https://example.com/article"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "ingested"
    assert data["entry_ids"] == ["kb-00010", "kb-00011"]
    assert data["entry_count"] == 2
    assert data["summary"] == "An article about testing"


@pytest.mark.asyncio
async def test_ingest_url_with_project(ingest_app):
    """project_ref is forwarded to ingester."""
    fake_result = FileResult(
        path="https://x.com",
        action="ingested",
        entry_count=1,
        entry_ids=["kb-00010"],
    )

    with _patch_ingest(fake_result) as mock_ingest:
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest_url",
                json={"url": "https://x.com", "project_ref": "my-project"},
            )

    assert resp.status_code == 200
    # First arg is self (instance), so check the call args
    mock_ingest.assert_called_once()
    call_kwargs = mock_ingest.call_args
    assert call_kwargs[0][0] == "https://x.com"  # positional: url
    assert call_kwargs[1]["project_ref"] == "my-project"


@pytest.mark.asyncio
async def test_ingest_url_empty_url(ingest_app):
    """Missing URL returns 400."""
    transport = ASGITransport(app=ingest_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/ingest_url", json={"url": ""})

    assert resp.status_code == 400
    assert "url is required" in resp.json()["error"]


@pytest.mark.asyncio
async def test_ingest_url_no_store(web_kb):
    """Without store on app.state, returns 503."""
    db, embedder, _ids = web_kb
    llm = FakeLLM()
    app = create_app_with_deps(db, embedder, llm)  # no store/extraction_llm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ingest_url",
            json={"url": "https://example.com"},
        )

    assert resp.status_code == 503
    assert "not available" in resp.json()["error"]


@pytest.mark.asyncio
async def test_ingest_url_error_result(ingest_app):
    """Ingester returning action=error is forwarded as 200 with error details."""
    fake_result = FileResult(
        path="https://bad.com",
        action="error",
        reason="Could not extract content",
    )

    with _patch_ingest(fake_result):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest_url",
                json={"url": "https://bad.com"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "error"
    assert "extract" in data["reason"]


@pytest.mark.asyncio
async def test_ingest_url_exception(ingest_app):
    """Ingester raising an exception returns 500."""
    with patch(
        "personal_kb.ingest.ingester.FileIngester.ingest_url",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest_url",
                json={"url": "https://example.com"},
            )

    assert resp.status_code == 500
    assert "boom" in resp.json()["error"]


@pytest.mark.asyncio
async def test_ingest_url_unchanged(ingest_app):
    """Already-ingested URL returns unchanged action."""
    fake_result = FileResult(path="https://example.com", action="unchanged")

    with _patch_ingest(fake_result):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest_url",
                json={"url": "https://example.com"},
            )

    assert resp.status_code == 200
    assert resp.json()["action"] == "unchanged"


# --- Streaming endpoint tests ---


def _parse_sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse SSE text into (event_type, data) tuples."""
    events = []
    event_type = ""
    for line in text.split("\n"):
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: ") and event_type:
            data = json.loads(line[6:])
            events.append((event_type, data))
            event_type = ""
    return events


def _patch_ingest_url(fake_result):
    """Patch FileIngester.ingest_url for streaming tests."""
    return patch(
        "personal_kb.ingest.ingester.FileIngester.ingest_url",
        new_callable=AsyncMock,
        return_value=fake_result,
    )


def _patch_ingest_text(fake_result):
    """Patch FileIngester.ingest_text for streaming tests."""
    return patch(
        "personal_kb.ingest.ingester.FileIngester.ingest_text",
        new_callable=AsyncMock,
        return_value=fake_result,
    )


@pytest.mark.asyncio
async def test_stream_single_url(ingest_app):
    """Streaming a single URL emits batch_progress, item_done, batch_done, stream_end."""
    fake_result = FileResult(
        path="https://example.com/a",
        action="ingested",
        entry_count=2,
        entry_ids=["kb-00010", "kb-00011"],
    )

    with _patch_ingest_url(fake_result):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest/stream",
                json={"items": [{"type": "url", "value": "https://example.com/a"}]},
            )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    types = [e[0] for e in events]
    assert "batch_progress" in types
    assert "item_done" in types
    assert "batch_done" in types
    assert "stream_end" in types

    # batch_done should have correct counts
    batch_done = next(d for t, d in events if t == "batch_done")
    assert batch_done["total_entries"] == 2
    assert batch_done["entry_ids"] == ["kb-00010", "kb-00011"]


@pytest.mark.asyncio
async def test_stream_multi_url(ingest_app):
    """Multiple URLs produce multiple batch_progress events."""
    results = [
        FileResult(path="https://a.com", action="ingested", entry_count=1, entry_ids=["kb-00010"]),
        FileResult(path="https://b.com", action="ingested", entry_count=1, entry_ids=["kb-00011"]),
    ]

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        r = results[call_count]
        call_count += 1
        return r

    with patch(
        "personal_kb.ingest.ingester.FileIngester.ingest_url",
        new_callable=AsyncMock,
        side_effect=side_effect,
    ):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest/stream",
                json={
                    "items": [
                        {"type": "url", "value": "https://a.com"},
                        {"type": "url", "value": "https://b.com"},
                    ]
                },
            )

    events = _parse_sse_events(resp.text)
    batch_progress_events = [(t, d) for t, d in events if t == "batch_progress"]
    assert len(batch_progress_events) == 2
    assert batch_progress_events[0][1]["batch_index"] == 0
    assert batch_progress_events[1][1]["batch_index"] == 1

    batch_done = next(d for t, d in events if t == "batch_done")
    assert batch_done["total_entries"] == 2
    assert set(batch_done["entry_ids"]) == {"kb-00010", "kb-00011"}


@pytest.mark.asyncio
async def test_stream_file(ingest_app):
    """File content items route through ingest_text."""
    fake_result = FileResult(
        path="notes.md",
        action="ingested",
        entry_count=1,
        entry_ids=["kb-00010"],
    )

    with _patch_ingest_text(fake_result) as mock_text:
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest/stream",
                json={
                    "items": [
                        {"type": "file", "name": "notes.md", "content": "# Notes\nSome text"},
                    ]
                },
            )

    events = _parse_sse_events(resp.text)
    batch_done = next(d for t, d in events if t == "batch_done")
    assert batch_done["total_entries"] == 1
    mock_text.assert_called_once()


@pytest.mark.asyncio
async def test_stream_mixed_error(ingest_app):
    """One item failing doesn't stop the batch."""
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        return FileResult(
            path="https://b.com",
            action="ingested",
            entry_count=1,
            entry_ids=["kb-00010"],
        )

    with patch(
        "personal_kb.ingest.ingester.FileIngester.ingest_url",
        new_callable=AsyncMock,
        side_effect=side_effect,
    ):
        transport = ASGITransport(app=ingest_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/ingest/stream",
                json={
                    "items": [
                        {"type": "url", "value": "https://fail.com"},
                        {"type": "url", "value": "https://b.com"},
                    ]
                },
            )

    events = _parse_sse_events(resp.text)
    types = [e[0] for e in events]
    assert "ingest_error" in types
    assert "batch_done" in types

    # Should still have the successful entry
    batch_done = next(d for t, d in events if t == "batch_done")
    assert batch_done["total_entries"] == 1


@pytest.mark.asyncio
async def test_stream_no_items(ingest_app):
    """Empty items list returns 400."""
    transport = ASGITransport(app=ingest_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/ingest/stream", json={"items": []})

    assert resp.status_code == 400
    assert "items is required" in resp.json()["error"]


@pytest.mark.asyncio
async def test_stream_no_store(web_kb):
    """Without store on app.state, returns 503."""
    db, embedder, _ids = web_kb
    llm = FakeLLM()
    app = create_app_with_deps(db, embedder, llm)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ingest/stream",
            json={"items": [{"type": "url", "value": "https://example.com"}]},
        )

    assert resp.status_code == 503
    assert "not available" in resp.json()["error"]
