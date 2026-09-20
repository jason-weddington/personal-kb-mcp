"""Shared pytest fixtures for the kb-core test package.

Currently the only shared plumbing here is the real-Postgres integration
suite (see ``test_postgres_backend.py``).  Everything is env-var-gated:

* ``KB_TEST_DATABASE_URL`` unset AND ``KB_REQUIRE_POSTGRES_TESTS`` falsy →
  the ``pg_url`` fixture calls :func:`pytest.skip`, the
  ``@pytest.mark.postgres`` suite skips, and the existing SQLite suites
  (``test_knowledge_base.py``, ``test_hybrid_signals.py``, …) stay green.
  Hermetic default preserved.
* ``KB_TEST_DATABASE_URL`` unset AND ``KB_REQUIRE_POSTGRES_TESTS`` truthy →
  the ``pg_url`` fixture calls :func:`pytest.fail` instead of skipping.  A
  skip is the right default on a laptop with no Postgres; it is the wrong
  default anywhere the DSN is supposed to be present — a silent skip there
  means a cross-backend feature can ship having exercised zero Postgres
  tests (see kb-03277).
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
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse, urlunparse

import pytest

from kb_core.db.postgres_backend import PostgresBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from _pytest.terminal import TerminalReporter

_SKIP_MESSAGE = "KB_TEST_DATABASE_URL not set — Postgres integration tests skipped"

_PgGateAction = Literal["proceed", "skip", "fail"]


def _is_truthy_kb_flag(raw: str | None) -> bool:
    """Mirror the house ``KB_*`` boolean convention.

    Every other ``KB_*`` boolean in this codebase (see
    ``personal_kb/config.py`` / ``kb_service/config.py``) is read as
    ``os.environ.get(..., "").upper() == "TRUE"`` — case-insensitive
    ``TRUE``. Per this item's acceptance criteria we additionally accept
    ``"1"`` for ``KB_REQUIRE_POSTGRES_TESTS``.
    """
    value = (raw or "").strip().upper()
    return value in {"TRUE", "1"}


def _pg_gate_decision(
    require_raw: str | None, dsn_raw: str | None
) -> tuple[_PgGateAction, str | None]:
    """Pure decision logic backing the ``pg_url`` fixture.

    Extracted so it can be unit-tested directly without needing a real
    Postgres (or even a real pytest session) — see
    ``test_pg_gate_decision.py``.

    Returns ``(action, message)``:

    * ``("proceed", None)`` — a DSN was provided; the caller should use it.
    * ``("skip", message)`` — no DSN and the suite was not required; skip.
    * ``("fail", message)`` — no DSN but the suite WAS required; fail.
    """
    dsn = (dsn_raw or "").strip()
    if dsn:
        return ("proceed", None)
    if _is_truthy_kb_flag(require_raw):
        return (
            "fail",
            "KB_REQUIRE_POSTGRES_TESTS is set but KB_TEST_DATABASE_URL is "
            "missing or empty — the postgres suite was required but no DSN "
            "was provided.",
        )
    return ("skip", _SKIP_MESSAGE)


def _redact_dsn_password(dsn: str) -> str:
    """Return ``dsn`` with any password component replaced by ``***``."""
    parsed = urlparse(dsn)
    if not parsed.password:
        return dsn
    userinfo = parsed.username or ""
    userinfo += ":***"
    host_port = parsed.hostname or ""
    if parsed.port is not None:
        host_port += f":{parsed.port}"
    netloc = f"{userinfo}@{host_port}" if userinfo else host_port
    return urlunparse(parsed._replace(netloc=netloc))


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Return ``KB_TEST_DATABASE_URL`` or skip/fail the whole postgres suite.

    Reading via :func:`os.environ.get` (not ``os.environ[...]``) means an
    unset OR empty value never raises a ``KeyError``.

    * No DSN + ``KB_REQUIRE_POSTGRES_TESTS`` falsy or unset → :func:`pytest.skip`
      (byte-identical to the pre-``KB_REQUIRE_POSTGRES_TESTS`` behavior).
    * No DSN + ``KB_REQUIRE_POSTGRES_TESTS`` truthy → :func:`pytest.fail` — a
      required Postgres suite that silently skipped is exactly the failure
      mode this fixture exists to prevent (kb-03277).
    """
    dsn_raw = os.environ.get("KB_TEST_DATABASE_URL", "")
    require_raw = os.environ.get("KB_REQUIRE_POSTGRES_TESTS", "")
    action, message = _pg_gate_decision(require_raw, dsn_raw)
    if action == "fail":
        assert message is not None
        pytest.fail(message)
    if action == "skip":
        assert message is not None
        pytest.skip(message)
    return dsn_raw.strip()


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
    required = _is_truthy_kb_flag(os.environ.get("KB_REQUIRE_POSTGRES_TESTS", ""))

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

    try:
        asyncio.run(_create())
    except Exception as exc:
        # A DSN that is present but unusable (server down, auth rejected,
        # ...) must not surface as an opaque collection error when the
        # suite was explicitly required — name the underlying asyncpg/OS
        # error and the (password-redacted) DSN so the failure is
        # actionable. When the suite was NOT required, preserve prior
        # behavior and let the original exception propagate.
        if required:
            pytest.fail(
                "KB_REQUIRE_POSTGRES_TESTS is set but the postgres maintenance "
                f"connection could not be established: {exc!r} "
                f"(DSN: {_redact_dsn_password(maintenance_dsn)})"
            )
        raise
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
        " audit_events, embedding_retry_queue, map_eligibility_override,"
        " map_cluster_ledger"
        " RESTART IDENTITY CASCADE"
    )
    try:
        yield backend
    finally:
        await backend.close()


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print how many ``@pytest.mark.postgres`` tests ran vs. skipped.

    This is the whole point of this item: a silent zero (the postgres
    suite quietly skipping in an environment where it was supposed to run)
    is exactly the failure kb-03277 hit. The line below always prints —
    even when both counts are zero — so that outcome cannot go unnoticed
    in the terminal output.
    """
    ran = sum(
        1
        for report in terminalreporter.stats.get("passed", [])
        + terminalreporter.stats.get("failed", [])
        if report.when == "call" and "postgres" in report.keywords
    )
    skipped = sum(
        1
        for report in terminalreporter.stats.get("skipped", [])
        if report.when == "setup" and "postgres" in report.keywords
    )
    terminalreporter.write_line(f"postgres marker: {ran} ran, {skipped} skipped")
