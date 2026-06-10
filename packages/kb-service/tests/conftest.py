"""Hermetic test fixtures: no live Postgres, Ollama, or network.

The app's lifespan normally calls ``init_db()`` and ``create_postgres()``. Both
are monkeypatched here so the suite never touches a real database or embedder.
``app.state.kb`` is replaced with a ``FakeKnowledgeBase`` whose ``search`` is
parametrizable, and ``database.get_db`` is replaced with a minimal fake pool.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult

import kb_service.auth as auth_module
import kb_service.database as database
import kb_service.main as main_module
from kb_service.main import app
from kb_service.models import User


class FakeKnowledgeBase:
    """Stand-in for the kb-core ``KnowledgeBase`` singleton."""

    def __init__(self, results: list[SearchResult], filtered_count: int) -> None:
        self.results = results
        self.filtered_count = filtered_count
        self.search_calls: list[tuple[Any, str | None]] = []

    async def search(
        self, query: Any, *, contributor: str | None = None
    ) -> tuple[list[SearchResult], int]:
        """Record the call and return the configured results."""
        self.search_calls.append((query, contributor))
        return self.results, self.filtered_count

    async def close(self) -> None:
        """No-op close."""


class FakeDbPool:
    """Minimal DbPool whose row lookups return None by default."""

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return []

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        return "OK"

    def acquire(self) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def make_search_result() -> SearchResult:
    """Build one SearchResult-shaped object for the non-empty search case."""
    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="Example entry",
        long_title="An example knowledge entry",
        knowledge_details="Some details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    return SearchResult(
        entry=entry,
        score=0.9,
        effective_confidence=0.9,
        staleness_warning=None,
        match_source="hybrid",
    )


def fake_user() -> User:
    """A fake authenticated user for dependency-override cases."""
    return User(
        id="00000000-0000-0000-0000-000000000001",
        email="tester@example.com",
        hashed_password="x",
        is_admin=False,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def fake_kb(request: pytest.FixtureRequest) -> FakeKnowledgeBase:
    """Fake KB. Returns no results by default; one result when indirectly
    parametrized with ``one_result``.
    """
    if getattr(request, "param", None) == "one_result":
        return FakeKnowledgeBase(results=[make_search_result()], filtered_count=1)
    return FakeKnowledgeBase(results=[], filtered_count=0)


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> Iterator[TestClient]:
    """A TestClient with a hermetic lifespan and mocked kb + service-auth DB."""

    async def _fake_init_db() -> None:
        return None

    async def _fake_close_db() -> None:
        return None

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        return fake_kb

    async def _fake_get_db() -> FakeDbPool:
        return FakeDbPool()

    monkeypatch.setattr(main_module, "init_db", _fake_init_db)
    monkeypatch.setattr(main_module, "close_db", _fake_close_db)
    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(database, "get_db", _fake_get_db)
    # auth.py binds get_db at import time, so patch its reference too.
    monkeypatch.setattr(auth_module, "get_db", _fake_get_db)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
