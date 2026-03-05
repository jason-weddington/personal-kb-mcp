"""Web test fixtures — httpx AsyncClient + in-memory DB."""

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore
from tests.conftest import FakeEmbedder


@pytest_asyncio.fixture
async def web_kb():
    """In-memory KB with a few entries for web tests.

    Yields (db, embedder, entry_ids).
    """
    db = await create_connection(":memory:")
    embedder = FakeEmbedder(db)
    store = KnowledgeStore(db)
    builder = GraphBuilder(db)

    ids: dict[str, str] = {}
    for title, etype, project, tags in [
        ("sqlite-async", EntryType.FACTUAL_REFERENCE, "personal-kb", ["sqlite", "python"]),
        ("fastapi-decision", EntryType.DECISION, "web-service", ["python", "api"]),
    ]:
        entry = await store.create_entry(
            short_title=title,
            long_title=f"Long title for {title}",
            knowledge_details=f"Details about {title}",
            entry_type=etype,
            project_ref=project,
            tags=list(tags),
        )
        ids[title] = entry.id
        await builder.build_for_entry(entry)
        embedding = await embedder.embed(entry.embedding_text)
        if embedding:
            await embedder.store_embedding(entry.id, embedding)
            await store.mark_embedding(entry.id)

    yield db, embedder, ids
    await db.close()
