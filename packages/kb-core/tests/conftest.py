"""Shared pytest fixtures for the kb-core test package.

Currently the only shared plumbing here is the real-Postgres integration
suite (see ``test_postgres_backend.py``).  Everything is env-var-gated:

* ``KB_TEST_DATABASE_URL`` unset  → the ``pg_url`` fixture calls
  :func:`pytest.skip`, the ``@pytest.mark.postgres`` suite skips, and the
  existing SQLite suites (``test_knowledge_base.py``,
  ``test_hybrid_signals.py``, …) stay green.  Hermetic default preserved.
* ``KB_TEST_DATABASE_URL`` set  → we mirror the house convention proven in
  ``agent_gtd/tests/test_pg_schema_bootstrap.py``: parse the DSN, swap the
  dbname to ``postgres`` to get a maintenance DSN, ``CREATE DATABASE
  kb_test_<uuid>`` there, run every test against that throwaway DB, then
  ``DROP DATABASE ... WITH (FORCE)`` on session teardown.

Tests MUST NEVER write to or truncate the DB named in
``KB_TEST_DATABASE_URL`` directly — it may be the live KB.  All schema /
inserts / TRUNCATEs go to the throwaway DB only (the DSN yielded by
``pg_temp_db`` / the ``PostgresBackend`` yielded by ``pg_kb``).

The env-var namespacing matches ``AGENT_GTD_TEST_DATABASE_URL`` in the
agent_gtd repo and deliberately avoids ``KB_DATABASE_URL`` (which points
at the real KB in dev / prod).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import pytest

from kb_core.db.postgres_backend import PostgresBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Return ``KB_TEST_DATABASE_URL`` or skip the whole postgres suite.

    Reading via :func:`os.environ.get` (not ``os.environ[...]``) means an
    unset OR empty value both skip cleanly — never error, never fail.
    """
    url = os.environ.get("KB_TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip("KB_TEST_DATABASE_URL not set — Postgres integration tests skipped")
    return url


@pytest.fixture(scope="session")
def pg_temp_db(pg_url: str) -> Iterator[str]:
    """Create a throwaway ``kb_test_<uuid>`` DB on the same server; drop on teardown.

    We derive a maintenance DSN from ``pg_url`` by swapping the path
    component (the dbname) to ``postgres``, connect there, and run
    ``CREATE DATABASE kb_test_<uuid4-hex-8>``.  On teardown we reconnect
    to the maintenance DSN and ``DROP DATABASE IF EXISTS ... WITH
    (FORCE)`` — ``WITH (FORCE)`` terminates any still-open connections
    that would otherwise pin the DB.

    Safety rail (also called out at module level): tests must never
    write to or truncate the DB named in ``KB_TEST_DATABASE_URL``
    directly — it may be the live KB.  Everything runs against the
    throwaway DB yielded here.
    """
    import asyncpg  # local import — asyncpg is only pulled in when suite enabled

    short_id = uuid.uuid4().hex[:8]
    temp_db_name = f"kb_test_{short_id}"
    parsed = urlparse(pg_url)
    maintenance_dsn = urlunparse(parsed._replace(path="/postgres"))
    temp_dsn = urlunparse(parsed._replace(path=f"/{temp_db_name}"))

    async def _create() -> None:
        conn = await asyncpg.connect(maintenance_dsn)
        try:
            await conn.execute(f'CREATE DATABASE "{temp_db_name}"')
        finally:
            await conn.close()

    async def _drop() -> None:
        conn = await asyncpg.connect(maintenance_dsn)
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{temp_db_name}" WITH (FORCE)')
        finally:
            await conn.close()

    asyncio.run(_create())
    try:
        yield temp_dsn
    finally:
        asyncio.run(_drop())


@pytest.fixture
async def pg_kb(pg_temp_db: str) -> AsyncIterator[PostgresBackend]:
    """Yield a schema-applied :class:`PostgresBackend` on the throwaway DB.

    Uses :meth:`PostgresBackend.create` + :meth:`apply_schema` — NOT
    ``create_postgres``.  ``create_postgres`` returns a ``KnowledgeBase``
    facade whose surface does not expose the Database-interface methods
    under test here (``delete_llm_edges``, ``fts_search``,
    ``vector_store`` / ``vector_search``, ``next_sequence_value``,
    ``vacuum``).

    Between tests we ``TRUNCATE`` the working tables to isolate state.
    The list is exact-and-complete and deliberately EXCLUDES three
    tables:

    * ``schema_version`` and ``deployment_config`` hold schema state
      that must survive between tests.
    * ``entry_id_seq`` holds the sequence counter — truncating it
      would break :meth:`PostgresBackend.next_sequence_value`.

    ``CASCADE`` handles the ``graph_edges → graph_nodes`` and
    ``entry_versions → knowledge_entries`` FK ordering.  ``TRUNCATE``
    is safe here because it targets only the throwaway DB.
    """
    backend = await PostgresBackend.create(pg_temp_db)
    await backend.apply_schema(embedding_dim=1024)
    await backend.execute(
        "TRUNCATE graph_edges, graph_nodes, knowledge_entries, knowledge_vec,"
        " entry_versions, ingested_files, search_events, agent_feedback,"
        " audit_events RESTART IDENTITY CASCADE"
    )
    try:
        yield backend
    finally:
        await backend.close()
