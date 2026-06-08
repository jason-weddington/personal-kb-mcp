"""Unit tests for the Postgres LISTEN/NOTIFY wiring (with mocks).

These tests assert wiring only — the actual live ``pg_notify`` →
``add_listener`` round-trip on a real Postgres is NOT exercised here.
The sqlite/eval suite has no Postgres available, so the live round-trip
requires MANUAL verification against a real Postgres after merge to
origin (per the task spec and CLAUDE.md).

What we DO cover here:

* SQLite ``notify_maps_changed`` and ``start_maps_listener`` are no-ops.
* The Postgres listener loop opens a dedicated ``asyncpg.connect`` (NOT
  pooled), registers an ``add_listener`` against the right channel, and
  on every successful (re)connect calls the app-supplied
  ``on_reconnect`` callback.
* Reconnect on connection drop calls ``on_reconnect`` again.
* Teardown cancels the task and closes the dedicated connection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_kb.db.connection import create_connection

# ---------------------------------------------------------------------------
# SQLite no-op
# ---------------------------------------------------------------------------


@pytest.fixture()
async def sqlite_db():
    db = await create_connection(":memory:", embedding_dim=64)
    yield db
    await db.close()


async def test_sqlite_notify_maps_changed_is_noop(sqlite_db) -> None:
    """SQLite backend: ``notify_maps_changed`` returns None and raises nothing."""
    result = await sqlite_db.notify_maps_changed("any-project")
    assert result is None


async def test_sqlite_start_maps_listener_returns_noop_teardown(sqlite_db) -> None:
    """SQLite backend: listener starts no task; teardown is a no-op closure."""
    on_change = AsyncMock(return_value=None)
    on_reconnect = AsyncMock(return_value=None)

    teardown = await sqlite_db.start_maps_listener(on_change=on_change, on_reconnect=on_reconnect)
    # Callbacks must never be invoked on SQLite.
    on_change.assert_not_called()
    on_reconnect.assert_not_called()

    # Teardown must be awaitable and idempotent.
    result = await teardown()
    assert result is None
    # Calling again still no-ops without raising.
    assert await teardown() is None


# ---------------------------------------------------------------------------
# Postgres listener (mocked asyncpg)
# ---------------------------------------------------------------------------


def _make_pg_backend(
    *,
    url: str = "postgresql://test/db",
    password=None,
    ssl=None,
):
    """Construct a PostgresBackend with a mocked pool — for unit wiring tests."""
    from personal_kb.db.postgres_backend import PostgresBackend

    pool = MagicMock(name="pool")
    pool.close = AsyncMock(return_value=None)
    return PostgresBackend(pool, url=url, password=password, ssl=ssl)


async def test_pg_create_retains_listener_inputs() -> None:
    """``PostgresBackend.create`` stores url/password/ssl for the listener conn.

    The original code constructed the pool then discarded url/password/ssl
    — the dedicated listener connection needs them.
    """
    from personal_kb.db.postgres_backend import PostgresBackend

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock(return_value=None)

    async def _fake_create_pool(url: str, **kwargs):
        return mock_pool

    password_factory = lambda: "secret"  # noqa: E731
    ssl_obj = object()

    with patch("asyncpg.create_pool", side_effect=_fake_create_pool):
        backend = await PostgresBackend.create(
            "postgresql://x/y",
            password=password_factory,
            ssl=ssl_obj,
        )
    assert backend._url == "postgresql://x/y"
    assert backend._password is password_factory
    assert backend._ssl is ssl_obj


async def test_pg_listener_calls_on_reconnect_and_registers_listener() -> None:
    """First connect → ``on_reconnect`` invoked AND ``add_listener`` registered.

    Verifies the listener uses a DEDICATED asyncpg.connect (not pooled)
    and that the right channel is bound.
    """
    from personal_kb.db import postgres_backend as pgmod

    backend = _make_pg_backend()

    fake_conn = MagicMock(name="dedicated_conn")
    fake_conn.is_closed = MagicMock(return_value=False)
    fake_conn.close = AsyncMock(return_value=None)
    fake_conn.add_listener = AsyncMock(return_value=None)
    fake_conn.add_termination_listener = MagicMock(return_value=None)

    on_change = AsyncMock(return_value=None)
    on_reconnect = AsyncMock(return_value=None)

    connect_started = asyncio.Event()

    async def fake_connect(url: str, **kwargs):
        connect_started.set()
        return fake_conn

    with patch("asyncpg.connect", side_effect=fake_connect):
        teardown = await backend.start_maps_listener(on_change=on_change, on_reconnect=on_reconnect)
        # Give the loop a tick to run through connect + register.
        await asyncio.wait_for(connect_started.wait(), timeout=1.0)
        # Yield a few ticks so the loop reaches add_listener.
        for _ in range(5):
            await asyncio.sleep(0)

        assert backend._listener_conn is fake_conn
        on_reconnect.assert_awaited()
        fake_conn.add_listener.assert_awaited_once()
        args, _ = fake_conn.add_listener.call_args
        # First arg: channel name; second arg: callback callable.
        assert args[0] == pgmod.NOTIFY_CHANNEL
        assert callable(args[1])

        await teardown()
        assert backend._listener_task is None


async def test_pg_listener_reconnect_calls_on_reconnect_again() -> None:
    """Connection drop → loop reconnects and calls ``on_reconnect`` again."""
    from personal_kb.db import postgres_backend as pgmod

    backend = _make_pg_backend()

    on_change = AsyncMock(return_value=None)
    on_reconnect = AsyncMock(return_value=None)

    # Make the loop's sleep effectively instant so the test doesn't wait.
    original_sleep = asyncio.sleep

    async def _instant_sleep(delay, *args, **kwargs):
        if delay >= pgmod._LISTENER_RECONNECT_DELAY:
            return await original_sleep(0)
        return await original_sleep(delay)

    # Track # of connect calls; first conn drops quickly.
    connections: list[MagicMock] = []
    connect_count = 0

    async def fake_connect(url: str, **kwargs):
        nonlocal connect_count
        connect_count += 1
        conn = MagicMock(name=f"conn-{connect_count}")
        conn.is_closed = MagicMock(return_value=False)
        conn.close = AsyncMock(return_value=None)
        conn.add_listener = AsyncMock(return_value=None)
        # Capture the termination-listener so we can call it from outside.
        conn._term_cb = None

        def _ad_term(cb):
            conn._term_cb = cb

        conn.add_termination_listener = MagicMock(side_effect=_ad_term)
        connections.append(conn)
        return conn

    with (
        patch("asyncio.sleep", side_effect=_instant_sleep),
        patch("asyncpg.connect", side_effect=fake_connect),
    ):
        teardown = await backend.start_maps_listener(on_change=on_change, on_reconnect=on_reconnect)
        # Wait for first connect + add_listener.
        for _ in range(50):
            await original_sleep(0)
            if connections and connections[0].add_listener.await_count >= 1:
                break
        assert on_reconnect.await_count >= 1
        assert connections[0]._term_cb is not None

        # Simulate a connection drop: trigger the termination callback.
        connections[0]._term_cb(connections[0])

        # Yield until the second connection comes up.
        for _ in range(200):
            await original_sleep(0)
            if len(connections) >= 2 and connections[1].add_listener.await_count >= 1:
                break

        assert len(connections) >= 2
        assert on_reconnect.await_count >= 2

        await teardown()


async def test_pg_close_closes_listener_conn_when_set() -> None:
    """``PostgresBackend.close`` also closes the dedicated listener conn (guarded)."""
    backend = _make_pg_backend()

    # No listener conn → close still works.
    await backend.close()

    # Repeat with a fake listener conn assigned.
    backend2 = _make_pg_backend()
    fake_conn = MagicMock()
    fake_conn.is_closed = MagicMock(return_value=False)
    fake_conn.close = AsyncMock(return_value=None)
    backend2._listener_conn = fake_conn
    await backend2.close()
    fake_conn.close.assert_awaited_once()
    assert backend2._listener_conn is None


async def test_pg_teardown_closes_conn_and_cancels_task() -> None:
    """Teardown cancels the listener task and closes the dedicated conn."""
    backend = _make_pg_backend()
    fake_conn = MagicMock()
    fake_conn.is_closed = MagicMock(return_value=False)
    fake_conn.close = AsyncMock(return_value=None)
    fake_conn.add_listener = AsyncMock(return_value=None)
    fake_conn.add_termination_listener = MagicMock(return_value=None)

    async def fake_connect(url: str, **kwargs):
        return fake_conn

    on_change = AsyncMock(return_value=None)
    on_reconnect = AsyncMock(return_value=None)

    with patch("asyncpg.connect", side_effect=fake_connect):
        teardown = await backend.start_maps_listener(on_change=on_change, on_reconnect=on_reconnect)
        for _ in range(10):
            await asyncio.sleep(0)
        assert backend._listener_task is not None and not backend._listener_task.done()
        await teardown()
        assert backend._listener_task is None
        # connection close must have been awaited at least once.
        assert fake_conn.close.await_count >= 1
