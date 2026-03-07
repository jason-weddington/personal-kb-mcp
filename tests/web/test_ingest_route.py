"""Tests for POST /api/ingest_url route."""

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
