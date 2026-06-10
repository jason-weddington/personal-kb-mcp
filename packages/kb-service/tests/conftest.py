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

import kb_service.attribution as attribution_module
import kb_service.auth as auth_module
import kb_service.database as database
import kb_service.main as main_module
from kb_service.main import app
from kb_service.models import User


class _FakeCursor:
    """Minimal cursor returned by _FakeDb.execute."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeDb:
    """Minimal db handle whose execute() returns a _FakeCursor.

    Row source: if the FakeKnowledgeBase has ``maps_rows`` set (not None),
    those rows are returned verbatim (lets tests exercise raw None/empty refs).
    Otherwise rows are synthesised from the keys of ``maps_projects``.
    """

    def __init__(self, kb: "FakeKnowledgeBase") -> None:
        self._kb = kb

    async def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
    ) -> _FakeCursor:
        rows: list[tuple[Any, ...]] = (
            self._kb.maps_rows
            if self._kb.maps_rows is not None
            else [(ref,) for ref in self._kb.maps_projects]
        )
        return _FakeCursor(rows=rows)


class FakeKnowledgeBase:
    """Stand-in for the kb-core ``KnowledgeBase`` singleton."""

    def __init__(
        self,
        results: list[SearchResult],
        filtered_count: int,
        *,
        maps_projects: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self.results = results
        self.filtered_count = filtered_count
        self.maps_projects: dict[str, list[dict[str, str]]] = (
            maps_projects if maps_projects is not None else {}
        )
        self.maps_rows: list[tuple[Any, ...]] | None = None
        self.search_calls: list[tuple[Any, str | None]] = []
        self.db = _FakeDb(self)
        self.ask_calls: list[tuple[str, dict[str, Any]]] = []
        self.summarize_calls: list[tuple[str, dict[str, Any]]] = []
        self.ask_return: tuple[list[tuple[KnowledgeEntry, str]], int] = (
            [(make_search_result().entry, "fake ask context")],
            3,
        )
        self.summarize_return: str = "fake synthesized answer"

    async def search(
        self, query: Any, *, contributor: str | None = None
    ) -> tuple[list[SearchResult], int]:
        """Record the call and return the configured results."""
        self.search_calls.append((query, contributor))
        return self.results, self.filtered_count

    async def maps_for_project(
        self, project_ref: str, *, team: str | None = None
    ) -> list[dict[str, str]]:
        """Return configured maps for the given project_ref."""
        return self.maps_projects.get(project_ref, [])

    async def ask(
        self,
        question: str,
        *,
        scope: str | None = None,
        agentic: bool | None = None,
        max_tool_calls: int | None = None,
        limit: int = 20,
        include_graph_context: bool = True,
        event_callback: Any | None = None,
    ) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
        """Record the call and return the configured ask results."""
        self.ask_calls.append(
            (
                question,
                {
                    "scope": scope,
                    "agentic": agentic,
                    "max_tool_calls": max_tool_calls,
                    "limit": limit,
                    "include_graph_context": include_graph_context,
                    "event_callback": event_callback,
                },
            )
        )
        return self.ask_return

    async def summarize(
        self,
        question: str,
        *,
        scope: str | None = None,
        agentic: bool | None = None,
        agentic_synthesis: bool | None = None,
        max_tool_calls: int | None = None,
        limit: int = 20,
        event_callback: Any | None = None,
    ) -> str:
        """Record the call and return the configured summarize result."""
        self.summarize_calls.append(
            (
                question,
                {
                    "scope": scope,
                    "agentic": agentic,
                    "agentic_synthesis": agentic_synthesis,
                    "max_tool_calls": max_tool_calls,
                    "limit": limit,
                    "event_callback": event_callback,
                },
            )
        )
        return self.summarize_return

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


class StatefulFakeDbPool(FakeDbPool):
    """FakeDbPool extended with a stateful app_config store.

    Pass in a shared dict so that multiple calls to get_db() within the same
    test all see the same in-memory state — required for round-trip tests that
    PUT a setting then GET it back.
    """

    def __init__(self, app_config: dict[str, str]) -> None:
        self._app_config = app_config

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        if "app_config" in sql and "SELECT value" in sql:
            key = args[0]
            val = self._app_config.get(key)
            if val is None:
                return None
            return {"value": val}
        return await super().fetchrow(sql, *args)

    async def execute(self, sql: str, *args: Any) -> str:
        if "INSERT INTO app_config" in sql:
            # args: key, value, updated_at, updated_by
            self._app_config[args[0]] = args[1]
        elif "DELETE FROM app_config" in sql:
            self._app_config.pop(args[0], None)
        return "OK"


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


def fake_admin_user() -> User:
    """A fake admin user for dependency-override cases requiring admin privileges."""
    return User(
        id="00000000-0000-0000-0000-000000000002",
        email="admin@example.com",
        hashed_password="x",
        is_admin=True,
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

    # A single stateful pool instance shared across all get_db() calls in this
    # test, so round-trip PUT→GET tests see the same in-memory app_config state.
    _shared_pool = StatefulFakeDbPool({})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return _shared_pool

    monkeypatch.setattr(main_module, "init_db", _fake_init_db)
    monkeypatch.setattr(main_module, "close_db", _fake_close_db)
    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(database, "get_db", _fake_get_db)
    # auth.py and attribution.py each bind get_db at import time; patch both.
    monkeypatch.setattr(auth_module, "get_db", _fake_get_db)
    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
