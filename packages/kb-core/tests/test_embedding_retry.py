"""Hermetic tests for ``kb_core.embedding_retry`` — the self-healing embedding

retry queue + background worker (GTD 735a7e1d). No live Ollama, no network —
every embedder here is an in-process test double.

Every KB is opened via ``create_sqlite(path, embedder=<stub>)`` per the spec:
``create_sqlite`` defaults to FTS-only (``embedding=None``), so the explicit
``embedder=`` kwarg is required to exercise any of the embed-failure paths.

Entries that don't need to go through the facade's embed/enrich pipeline are
created directly via ``kb.knowledge_store.create_entry(...)`` — this keeps
queue-level tests (enqueue/backfill/resolve/queue_stats/drain_once) decoupled
from the facade's own auto-enqueue behavior (which is tested separately,
against ``kb.store()`` / ``kb.store_batch()``, in its own section below).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta, timezone

from kb_core import create_sqlite
from kb_core.config import EmbeddingConfig, EmbeddingRetryConfig
from kb_core.embedding_retry import (
    BACKOFF_SCHEDULE_SECONDS,
    MAX_ATTEMPTS,
    EmbeddingRetryWorker,
    _claim_due,
    backfill,
    enqueue,
    queue_stats,
    resolve,
)
from kb_core.models.entry import EntryType

# ---------------------------------------------------------------------------
# Test doubles — no network, no Ollama.
# ---------------------------------------------------------------------------


class NoneEmbedder:
    """Available, but ``embed``/``embed_batch`` report failure by returning ``None``."""

    async def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        return None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return None

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        raise AssertionError("store_embedding should never be reached")

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class RaisingEmbedder:
    """Available, but ``embed``/``embed_batch`` raise."""

    async def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        raise RuntimeError("boom")

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        raise RuntimeError("boom")

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        raise AssertionError("store_embedding should never be reached")

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class UnavailableEmbedder:
    """Always reports unavailable — the down-circuit test double."""

    async def is_available(self) -> bool:
        return False

    async def embed(self, text: str) -> list[float] | None:
        return None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return None

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        raise AssertionError("store_embedding should never be reached")

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class IsAvailableRaisesEmbedder:
    """``is_available()`` RAISES rather than returning ``False``.

    Reproduces the down-circuit-that-never-opens defect: an embedder whose
    availability probe raises (connection refused, DNS failure, auth error,
    ...) instead of cleanly returning ``False``. Before the fix this skipped
    ``_track_availability`` entirely, leaving ``_embedder_available`` at
    ``None`` forever and causing the worker loop to sleep at the fast
    ``poll_interval_seconds`` instead of the slower ``down_backoff_seconds``
    — 5x the intended request rate against a hard-down embedder.
    """

    async def is_available(self) -> bool:
        raise RuntimeError("connection refused")

    async def embed(self, text: str) -> list[float] | None:
        raise AssertionError("embed should never be reached")

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        raise AssertionError("embed_batch should never be reached")

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        raise AssertionError("store_embedding should never be reached")

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class RealShapeDownEmbedder:
    """``is_available()`` True (the 2026-09-16 shape) but ``embed_batch`` fails.

    During the real incident Ollama answered ``/api/tags`` throughout and
    failed only ``/api/embed`` — the is_available() circuit would NOT have
    opened. This is that exact shape.
    """

    async def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        return None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return None

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        raise AssertionError("store_embedding should never be reached")

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class WorkingEmbedder:
    """A fully working embedder stub.

    These tests only assert queue/``has_embedding`` bookkeeping, never actual
    vector content, so ``store_embedding``/``store_embeddings`` are no-ops
    (unlike ``tests/test_knowledge_base.py:57``'s ``FakeEmbedder``, which
    needs a live ``db`` handle to persist real vectors for search-ranking
    tests — a dependency this module's queue-level tests don't have, and
    which would create a construction-order problem against the spec's
    ``create_sqlite(path, embedder=<stub>)`` pattern).
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        return [0.1] * self.dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return [[0.1] * self.dim for _ in texts]

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        pass

    async def store_embeddings(self, entries: list[tuple[str, list[float]]]) -> None:
        pass

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class EmbedBatchOnlyEmbedder:
    """``embed_batch`` returns valid vectors but there is NO ``store_embeddings`` attribute.

    Reproduces the silent third failure shape at ``knowledge_base.py:659`` —
    ``if embeddings is not None and callable(store_embeddings):`` — an
    embedder that embeds successfully and then has its vectors discarded
    with no exception, no ``mark_embedding`` call, and no log line.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        return [0.1] * self.dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return [[0.1] * self.dim for _ in texts]

    async def search_similar(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class _FlakyMarkStore:
    """Wraps a real ``KnowledgeStore``; ``mark_embedding`` raises for one chosen id."""

    def __init__(self, inner: object, *, fail_id: str) -> None:
        self._inner = inner
        self._fail_id = fail_id

    async def mark_embedding(self, entry_id: str, has_embedding: bool = True) -> None:
        if entry_id == self._fail_id:
            raise RuntimeError("mark_embedding boom")
        await self._inner.mark_embedding(entry_id, has_embedding)  # type: ignore[attr-defined]


async def _create_entry(kb: object, *, title: str = "t") -> object:
    return await kb.knowledge_store.create_entry(  # type: ignore[attr-defined]
        short_title=title,
        long_title=title,
        knowledge_details=title,
        entry_type=EntryType.FACTUAL_REFERENCE,
    )


# ---------------------------------------------------------------------------
# enqueue() — idempotence, backoff offset, exhaustion revival (AC8)
# ---------------------------------------------------------------------------


async def test_enqueue_idempotent_preserves_backoff_position(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="first", now=now)
        await kb.db.execute(
            "UPDATE embedding_retry_queue SET attempts = 3 WHERE entry_id = ?", (entry.id,)
        )
        await kb.db.commit()
        cursor = await kb.db.execute(
            "SELECT next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        before_next = (await cursor.fetchone())["next_attempt_at"]

        await enqueue(kb.db, entry.id, error="second", now=now + timedelta(seconds=5))

        cursor = await kb.db.execute(
            "SELECT attempts, next_attempt_at, last_error, status "
            "FROM embedding_retry_queue WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        assert row["attempts"] == 3
        assert row["next_attempt_at"] == before_next
        assert row["last_error"] == "second"
        assert row["status"] == "pending"
    finally:
        await kb.close()


async def test_enqueue_fresh_row_due_60s_out(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now)

        cursor = await kb.db.execute(
            "SELECT next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        next_attempt_at = datetime.fromisoformat((await cursor.fetchone())["next_attempt_at"])
        assert next_attempt_at == now + timedelta(seconds=BACKOFF_SCHEDULE_SECONDS[0])

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=WorkingEmbedder(),
        )
        assert (await worker.drain_once(now=now))["claimed"] == 0
        assert (await worker.drain_once(now=now + timedelta(seconds=61)))["claimed"] == 1
    finally:
        await kb.close()


async def test_enqueue_revives_exhausted_row(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await kb.db.execute(
            "INSERT INTO embedding_retry_queue "
            "(entry_id, attempts, last_error, next_attempt_at, status, created_at, updated_at) "
            "VALUES (?, 6, 'boom', ?, 'exhausted', ?, ?)",
            (entry.id, now.isoformat(), now.isoformat(), now.isoformat()),
        )
        await kb.db.commit()

        await enqueue(kb.db, entry.id, error="revived", now=now)

        cursor = await kb.db.execute(
            "SELECT status, attempts FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        row = await cursor.fetchone()
        assert row["status"] == "pending"
        assert row["attempts"] == 0
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Timestamps (AC6)
# ---------------------------------------------------------------------------


async def test_next_attempt_at_ends_with_utc_offset(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        await enqueue(kb.db, entry.id, error="x", now=datetime.now(UTC))
        cursor = await kb.db.execute(
            "SELECT next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        assert (await cursor.fetchone())["next_attempt_at"].endswith("+00:00")
    finally:
        await kb.close()


async def test_next_attempt_at_orders_chronologically_across_microsecond_precision(
    tmp_path,
) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        earlier = await _create_entry(kb, title="earlier")
        later = await _create_entry(kb, title="later")
        base = datetime(2026, 1, 1, tzinfo=UTC)
        # `earlier` is due first but carries microseconds; `later` is due
        # after but has none — lexicographic ISO ordering must still win.
        await enqueue(
            kb.db, earlier.id, error="x", now=base - timedelta(seconds=59, microseconds=500000)
        )
        await enqueue(kb.db, later.id, error="x", now=base - timedelta(seconds=58))

        cursor = await kb.db.execute(
            "SELECT entry_id FROM embedding_retry_queue ORDER BY next_attempt_at"
        )
        rows = await cursor.fetchall()
        assert [r["entry_id"] for r in rows] == [earlier.id, later.id]
    finally:
        await kb.close()


async def test_claim_due_honors_non_utc_now(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now_utc = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now_utc)

        due_instant = now_utc + timedelta(seconds=61)
        non_utc_now = due_instant.astimezone(timezone(timedelta(hours=5)))
        claimed = await _claim_due(kb.db, now=non_utc_now, batch_size=10, lease_seconds=600)
        assert [c["entry_id"] for c in claimed] == [entry.id]
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Second-connection commit proof (AC5)
# ---------------------------------------------------------------------------


async def test_enqueue_commit_visible_from_second_sqlite_connection(tmp_path) -> None:
    path = tmp_path / "shared.db"
    kb1 = await create_sqlite(path, embedder=NoneEmbedder())
    entry = await _create_entry(kb1)
    await enqueue(kb1.db, entry.id, error="x", now=datetime.now(UTC))

    kb2 = await create_sqlite(path, embedder=NoneEmbedder())
    try:
        cursor = await kb2.db.execute(
            "SELECT COUNT(*) AS n FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        assert (await cursor.fetchone())["n"] == 1
    finally:
        await kb1.close()
        await kb2.close()


# ---------------------------------------------------------------------------
# _embed_one — both failure shapes (AC9)
# ---------------------------------------------------------------------------


async def test_store_enqueues_when_embed_returns_none(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await kb.store(
            short_title="t",
            long_title="t",
            knowledge_details="t",
            entry_type=EntryType.FACTUAL_REFERENCE,
            enrich=False,
        )
        cursor = await kb.db.execute(
            "SELECT last_error FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["last_error"] == "embed returned None"
    finally:
        await kb.close()


async def test_store_enqueues_when_embed_raises(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=RaisingEmbedder())
    try:
        entry = await kb.store(
            short_title="t",
            long_title="t",
            knowledge_details="t",
            entry_type=EntryType.FACTUAL_REFERENCE,
            enrich=False,
        )
        cursor = await kb.db.execute(
            "SELECT last_error FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert "boom" in rows[0]["last_error"]
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# store_batch — outcome-based enqueue, all three shapes (AC10)
# ---------------------------------------------------------------------------


async def test_store_batch_embed_batch_none_enqueues_all(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        created = await kb.store_batch(
            [
                {"short_title": "a", "long_title": "a", "knowledge_details": "a"},
                {"short_title": "b", "long_title": "b", "knowledge_details": "b"},
            ],
            enrich=False,
        )
        assert len(created) == 2
        cursor = await kb.db.execute("SELECT COUNT(*) AS n FROM embedding_retry_queue")
        assert (await cursor.fetchone())["n"] == 2
    finally:
        await kb.close()


async def test_store_batch_embed_batch_without_store_embeddings_enqueues_all(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=EmbedBatchOnlyEmbedder())
    try:
        created = await kb.store_batch(
            [
                {"short_title": "a", "long_title": "a", "knowledge_details": "a"},
                {"short_title": "b", "long_title": "b", "knowledge_details": "b"},
            ],
            enrich=False,
        )
        assert len(created) == 2
        cursor = await kb.db.execute("SELECT COUNT(*) AS n FROM embedding_retry_queue")
        assert (await cursor.fetchone())["n"] == 2
        for e in created:
            refreshed = await kb.get(e.id)
            assert refreshed is not None
            assert refreshed.has_embedding is False
    finally:
        await kb.close()


async def test_store_batch_fully_working_embedder_zero_queue_rows(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=WorkingEmbedder())
    try:
        created = await kb.store_batch(
            [
                {"short_title": "a", "long_title": "a", "knowledge_details": "a"},
                {"short_title": "b", "long_title": "b", "knowledge_details": "b"},
            ],
            enrich=False,
        )
        cursor = await kb.db.execute("SELECT COUNT(*) AS n FROM embedding_retry_queue")
        assert (await cursor.fetchone())["n"] == 0
        for e in created:
            refreshed = await kb.get(e.id)
            assert refreshed is not None
            assert refreshed.has_embedding is True
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# backfill() — source of truth, exhaustion revival (AC11)
# ---------------------------------------------------------------------------


async def test_backfill_queues_only_vectorless_active_entries(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        e1 = await _create_entry(kb, title="a")
        e2 = await _create_entry(kb, title="b")
        e3 = await _create_entry(kb, title="c")
        await kb.knowledge_store.mark_embedding(e3.id, True)

        n = await backfill(kb.db, kb.knowledge_store, now=datetime.now(UTC))
        assert n == 2

        cursor = await kb.db.execute("SELECT entry_id FROM embedding_retry_queue")
        queued = {r["entry_id"] for r in await cursor.fetchall()}
        assert queued == {e1.id, e2.id}
    finally:
        await kb.close()


async def test_backfill_revives_exhausted_leaves_pending_untouched(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        exhausted = await _create_entry(kb, title="a")
        pending = await _create_entry(kb, title="b")
        now = datetime.now(UTC)
        await kb.db.execute(
            "INSERT INTO embedding_retry_queue "
            "(entry_id, attempts, last_error, next_attempt_at, status, created_at, updated_at) "
            "VALUES (?, 6, 'boom', ?, 'exhausted', ?, ?)",
            (exhausted.id, now.isoformat(), now.isoformat(), now.isoformat()),
        )
        await kb.db.execute(
            "INSERT INTO embedding_retry_queue "
            "(entry_id, attempts, last_error, next_attempt_at, status, created_at, updated_at) "
            "VALUES (?, 3, 'boom', ?, 'pending', ?, ?)",
            (pending.id, now.isoformat(), now.isoformat(), now.isoformat()),
        )
        await kb.db.commit()

        await backfill(kb.db, kb.knowledge_store, now=now)

        cursor = await kb.db.execute(
            "SELECT status, attempts FROM embedding_retry_queue WHERE entry_id = ?",
            (exhausted.id,),
        )
        row = await cursor.fetchone()
        assert row["status"] == "pending"
        assert row["attempts"] == 0

        cursor = await kb.db.execute(
            "SELECT status, attempts FROM embedding_retry_queue WHERE entry_id = ?", (pending.id,)
        )
        row = await cursor.fetchone()
        assert row["status"] == "pending"
        assert row["attempts"] == 3
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# resolve() (AC31 kb-core half)
# ---------------------------------------------------------------------------


async def test_resolve_deletes_queue_rows_and_noops_on_empty(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        await enqueue(kb.db, entry.id, error="x", now=datetime.now(UTC))
        await resolve(kb.db, [entry.id])
        cursor = await kb.db.execute(
            "SELECT COUNT(*) AS n FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        assert (await cursor.fetchone())["n"] == 0

        await resolve(kb.db, [])  # must not raise
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Claim CAS (AC16)
# ---------------------------------------------------------------------------


async def test_claim_due_second_call_claims_zero(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now - timedelta(seconds=61))

        first = await _claim_due(kb.db, now=now, batch_size=10, lease_seconds=600)
        assert [c["entry_id"] for c in first] == [entry.id]

        second = await _claim_due(kb.db, now=now, batch_size=10, lease_seconds=600)
        assert second == []
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Embedder-down circuit (AC15)
# ---------------------------------------------------------------------------


async def test_embedder_unavailable_leaves_rows_byte_identical(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now - timedelta(seconds=61))
        cursor = await kb.db.execute(
            "SELECT attempts, next_attempt_at, status FROM embedding_retry_queue "
            "WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        before = (row["attempts"], row["next_attempt_at"], row["status"])

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=UnavailableEmbedder(),
        )
        result = await worker.drain_once(now=now)
        assert result == {"claimed": 0, "embedded": 0, "failed": 0, "exhausted": 0, "skipped": 0}

        cursor = await kb.db.execute(
            "SELECT attempts, next_attempt_at, status FROM embedding_retry_queue "
            "WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        assert (row["attempts"], row["next_attempt_at"], row["status"]) == before
    finally:
        await kb.close()


async def test_real_incident_shape_increments_attempts_each_drain(tmp_path) -> None:
    """is_available() True but embed_batch fails — the actual 2026-09-16 shape."""
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now - timedelta(seconds=61))

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=RealShapeDownEmbedder(),
        )
        clock = now
        for expected_attempts in range(1, 5):
            result = await worker.drain_once(now=clock)
            assert result["claimed"] == 1
            assert result["failed"] == 1
            cursor = await kb.db.execute(
                "SELECT attempts, status, next_attempt_at FROM embedding_retry_queue "
                "WHERE entry_id = ?",
                (entry.id,),
            )
            row = await cursor.fetchone()
            assert row["attempts"] == expected_attempts
            assert row["status"] == "pending"
            clock = datetime.fromisoformat(row["next_attempt_at"]) + timedelta(seconds=1)
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Batched drain success + per-entry write isolation (AC17)
# ---------------------------------------------------------------------------


async def test_successful_batch_drain_clears_queue_and_marks_embedded(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        e1 = await _create_entry(kb, title="a")
        e2 = await _create_entry(kb, title="b")
        now = datetime.now(UTC)
        await enqueue(kb.db, e1.id, error="x", now=now - timedelta(seconds=61))
        await enqueue(kb.db, e2.id, error="x", now=now - timedelta(seconds=61))

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=WorkingEmbedder(),
        )
        result = await worker.drain_once(now=now)
        assert result == {"claimed": 2, "embedded": 2, "failed": 0, "exhausted": 0, "skipped": 0}

        cursor = await kb.db.execute("SELECT COUNT(*) AS n FROM embedding_retry_queue")
        assert (await cursor.fetchone())["n"] == 0

        for eid in (e1.id, e2.id):
            refreshed = await kb.get(eid)
            assert refreshed is not None
            assert refreshed.has_embedding is True
    finally:
        await kb.close()


async def test_per_entry_write_failure_leaves_row_intact_and_isolates_others(
    tmp_path, caplog
) -> None:
    """A post-embed write failure (e.g. ``mark_embedding`` raising) must apply
    the SAME failure accounting as an embed failure: ``attempts`` increments,
    ``next_attempt_at`` reschedules per ``BACKOFF_SCHEDULE_SECONDS``, and the
    row reaches ``'exhausted'`` after ``MAX_ATTEMPTS`` write failures.

    Before the fix, ``_apply_success``'s exception branch only incremented
    the in-memory ``counters['failed']`` and never touched the row's
    ``attempts``/``next_attempt_at`` — a persistently-failing write retried
    FOREVER at the lease interval and could never exhaust. This test drives
    a persistently-broken write all the way to exhaustion to prove that no
    longer happens, while preserving the original isolation assertion: a
    write failure on one entry must not affect a healthy sibling claimed in
    the same batch.
    """
    caplog.set_level(logging.ERROR, logger="kb_core.embedding_retry")
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        ok = await _create_entry(kb, title="ok")
        broken = await _create_entry(kb, title="broken")
        now = datetime.now(UTC)
        await enqueue(kb.db, ok.id, error="x", now=now - timedelta(seconds=61))
        await enqueue(kb.db, broken.id, error="x", now=now - timedelta(seconds=61))

        flaky_store = _FlakyMarkStore(kb.knowledge_store, fail_id=broken.id)
        worker = EmbeddingRetryWorker(
            kb.db,
            flaky_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=WorkingEmbedder(),
        )
        caplog.clear()
        result = await worker.drain_once(now=now)
        assert result["claimed"] == 2
        assert result["embedded"] == 1
        assert result["failed"] == 1

        # Isolation: only the broken entry remains queued; `ok` was embedded
        # and its row deleted, unaffected by broken's write failure.
        cursor = await kb.db.execute("SELECT entry_id, attempts, status FROM embedding_retry_queue")
        rows = {r["entry_id"]: (r["attempts"], r["status"]) for r in await cursor.fetchall()}
        assert set(rows) == {broken.id}
        # Corrected accounting: the write failure counts as attempt 1, same
        # as an embed failure would — NOT left at attempts=0 forever.
        assert rows[broken.id] == (1, "pending")

        errors = [
            r
            for r in caplog.records
            if r.levelname == "ERROR" and "post-embed write failed" in r.getMessage()
        ]
        assert len(errors) == 1

        # Drive the persistently-broken row through the rest of the backoff
        # schedule — a write failure that never recovers must reach
        # 'exhausted' at MAX_ATTEMPTS, exactly like an embed failure, never
        # retrying forever at the lease interval (the unbounded-retry bug).
        clock = now
        for expected_attempts in range(2, MAX_ATTEMPTS + 1):
            cursor = await kb.db.execute(
                "SELECT next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?",
                (broken.id,),
            )
            row = await cursor.fetchone()
            clock = datetime.fromisoformat(row["next_attempt_at"]) + timedelta(seconds=1)

            result = await worker.drain_once(now=clock)
            assert result["claimed"] == 1

            cursor = await kb.db.execute(
                "SELECT attempts, status FROM embedding_retry_queue WHERE entry_id = ?",
                (broken.id,),
            )
            row = await cursor.fetchone()
            assert row["attempts"] == expected_attempts
            if expected_attempts < MAX_ATTEMPTS:
                assert row["status"] == "pending"
                assert result["failed"] == 1
            else:
                assert row["status"] == "exhausted"
                assert result["exhausted"] == 1

        # An exhausted row is never reclaimed — a further drain claims zero,
        # proving the row stopped retrying instead of looping forever.
        result = await worker.drain_once(now=clock + timedelta(days=1))
        assert result["claimed"] == 0
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Failure + backoff + exhaustion, all literals pinned (AC18)
# ---------------------------------------------------------------------------


async def test_backoff_literal_deltas_and_exhaustion(tmp_path, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="kb_core.embedding_retry")
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now - timedelta(seconds=61))

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=NoneEmbedder(),
        )
        clock = now
        deltas: list[int] = []
        caplog.clear()
        for i in range(MAX_ATTEMPTS):
            result = await worker.drain_once(now=clock)
            assert result["claimed"] == 1
            cursor = await kb.db.execute(
                "SELECT attempts, status, next_attempt_at FROM embedding_retry_queue "
                "WHERE entry_id = ?",
                (entry.id,),
            )
            row = await cursor.fetchone()
            assert row["status"] in ("pending", "exhausted")
            if i < MAX_ATTEMPTS - 1:
                assert row["status"] == "pending"
                observed_next = datetime.fromisoformat(row["next_attempt_at"])
                deltas.append(round((observed_next - clock).total_seconds()))
                clock = observed_next + timedelta(seconds=1)
            else:
                assert row["status"] == "exhausted"

        assert deltas == list(BACKOFF_SCHEDULE_SECONDS)

        exhaustion_errors = [
            r
            for r in caplog.records
            if r.levelname == "ERROR" and "embedding-retry-exhausted" in r.getMessage()
        ]
        assert len(exhaustion_errors) == 1

        # 7th drain claims zero — an exhausted row is never reclaimed.
        result = await worker.drain_once(now=clock + timedelta(days=1))
        assert result["claimed"] == 0
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# drain_once counter invariant (AC19)
# ---------------------------------------------------------------------------


async def test_drain_once_counter_invariant(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        good = await _create_entry(kb, title="good")
        deactivated = await _create_entry(kb, title="gone")
        await kb.knowledge_store.deactivate_entry(deactivated.id)

        now = datetime.now(UTC)
        await enqueue(kb.db, good.id, error="x", now=now - timedelta(seconds=61))
        await enqueue(kb.db, deactivated.id, error="x", now=now - timedelta(seconds=61))
        await enqueue(kb.db, "kb-99999", error="x", now=now - timedelta(seconds=61))

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=WorkingEmbedder(),
        )
        result = await worker.drain_once(now=now)
        assert result == {"claimed": 3, "embedded": 1, "failed": 0, "exhausted": 0, "skipped": 2}

        cursor = await kb.db.execute("SELECT entry_id FROM embedding_retry_queue")
        assert {r["entry_id"] for r in await cursor.fetchall()} == set()
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# queue_stats() (AC20)
# ---------------------------------------------------------------------------


async def test_queue_stats_empty_queue(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        stats = await queue_stats(kb.db, now=datetime.now(UTC))
        assert stats == {
            "pending": 0,
            "exhausted": 0,
            "oldest_pending_age_seconds": None,
            "next_due_at": None,
            "vectorless_unqueued": 0,
        }
    finally:
        await kb.close()


async def test_queue_stats_vectorless_unqueued_tripwire_count(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        await _create_entry(kb)
        stats = await queue_stats(kb.db, now=datetime.now(UTC))
        assert stats["vectorless_unqueued"] == 1
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Invariant tripwire logging (AC21)
# ---------------------------------------------------------------------------


async def test_vectorless_unqueued_tripwire_warns(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="kb_core.embedding_retry")
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        await _create_entry(kb)  # has_embedding=False, no queue row — the breach
        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=NoneEmbedder(),
        )
        caplog.clear()
        await worker.drain_once(now=datetime.now(UTC))
        breach = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "invariant breach" in r.getMessage()
        ]
        assert len(breach) == 1
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Transition-only logging (AC22)
# ---------------------------------------------------------------------------


async def test_embedder_down_then_recovered_transition_logging(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="kb_core.embedding_retry")
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="x", now=now - timedelta(seconds=61))

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=UnavailableEmbedder(),
        )
        caplog.clear()
        await worker.drain_once(now=now)
        await worker.drain_once(now=now)  # still down — must NOT log a second WARNING
        down_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "embedder unreachable" in r.getMessage()
        ]
        assert len(down_warnings) == 1

        worker._embedder = WorkingEmbedder()  # simulate recovery on the same worker
        caplog.clear()
        result = await worker.drain_once(now=now)
        assert result["claimed"] == 1
        recovered = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "embedder recovered" in r.getMessage()
        ]
        assert len(recovered) == 1
    finally:
        await kb.close()


async def test_is_available_raise_treated_as_down_drives_track_availability(
    tmp_path, caplog
) -> None:
    """``is_available()`` raising must still reach ``_track_availability(False)``.

    Before the fix, an exception from ``is_available()`` propagated straight
    out of ``drain_once`` — ``_track_availability`` was never called,
    ``_embedder_available`` stayed ``None`` forever, and the down-transition
    WARNING never logged. This pins the corrected behavior directly on the
    unit-testable seam (``drain_once``): the raise is caught, availability is
    recorded as ``False`` (so ``_run_forever`` takes the ``down_backoff_seconds``
    sleep instead of the fast ``poll_interval_seconds`` one — see
    ``test_run_loop_takes_down_backoff_sleep_when_is_available_raises`` for the
    end-to-end sleep-value proof), and the usual "embedder unreachable"
    transition log fires exactly like a clean ``False`` return would.
    """
    caplog.set_level(logging.WARNING, logger="kb_core.embedding_retry")
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=IsAvailableRaisesEmbedder(),
        )
        assert worker._embedder_available is None

        caplog.clear()
        result = await worker.drain_once(now=datetime.now(UTC))

        assert result == {"claimed": 0, "embedded": 0, "failed": 0, "exhausted": 0, "skipped": 0}
        assert worker._embedder_available is False
        down_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "embedder unreachable" in r.getMessage()
        ]
        assert len(down_warnings) == 1
    finally:
        await kb.close()


async def test_run_loop_takes_down_backoff_sleep_when_is_available_raises(
    tmp_path, monkeypatch
) -> None:
    """End-to-end: ``is_available()`` raising makes the real worker LOOP sleep
    at ``down_backoff_seconds``, not the fast ``poll_interval_seconds`` — the
    5x-too-fast-polling defect. ``asyncio.sleep`` is patched to record the
    requested duration and then block (via a never-resolving Future) so the
    test controls the loop's lifetime deterministically via ``worker.stop()``.
    """
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(
                poll_interval_seconds=1.0,
                down_backoff_seconds=999.0,
                backfill_interval_seconds=3600.0,
            ),
            embedder=IsAvailableRaisesEmbedder(),
        )

        sleeps: list[float] = []
        sleep_recorded = asyncio.Event()

        async def _fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            sleep_recorded.set()
            await asyncio.Future()  # block until the task is cancelled by stop()

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        await worker.start()
        try:
            await asyncio.wait_for(sleep_recorded.wait(), timeout=5)
        finally:
            await worker.stop()

        assert sleeps == [999.0]
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Audit trail (AC23)
# ---------------------------------------------------------------------------


async def test_successful_drain_writes_embedding_recovered_audit_event(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        entry = await _create_entry(kb)
        now = datetime.now(UTC)
        await enqueue(kb.db, entry.id, error="embed returned None", now=now - timedelta(seconds=61))

        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(batch_size=10, lease_seconds=600),
            embedder=WorkingEmbedder(),
        )
        result = await worker.drain_once(now=now)
        assert result["embedded"] == 1

        cursor = await kb.db.execute(
            "SELECT COUNT(*) AS n FROM audit_events "
            "WHERE event_type = 'embedding_recovered' AND entry_id = ?",
            (entry.id,),
        )
        assert (await cursor.fetchone())["n"] == 1
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Worker lifecycle: injection, ownership, idempotent start/stop (AC12-14)
# ---------------------------------------------------------------------------


async def test_worker_embedder_injected_verbatim_not_closed_on_stop(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        stub = WorkingEmbedder()
        closed = {"called": False}

        async def _close() -> None:
            closed["called"] = True

        stub.close = _close  # type: ignore[attr-defined]

        worker = EmbeddingRetryWorker(
            kb.db, kb.knowledge_store, EmbeddingRetryConfig(), embedder=stub
        )
        assert worker._embedder is stub
        assert worker._owned_embedder is False
        await worker.stop()  # never started — must be a safe no-op
        assert closed["called"] is False
    finally:
        await kb.close()


async def test_worker_builds_owned_embedder_with_worker_timeout(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(request_timeout=180.0),
            EmbeddingConfig(timeout=10.0),
        )
        await worker.start()
        try:
            assert worker._owned_embedder is True
            assert worker._embedder is not None
            assert worker._embedder._config.timeout == 180.0
        finally:
            await worker.stop()
    finally:
        await kb.close()


async def test_worker_no_embedder_no_config_warns_and_does_not_start(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="kb_core.embedding_retry")
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(kb.db, kb.knowledge_store, EmbeddingRetryConfig())
        await worker.start()
        assert worker.running is False
        assert any("FTS-only" in r.getMessage() for r in caplog.records)
    finally:
        await kb.close()


async def test_worker_start_idempotent_single_task(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(poll_interval_seconds=3600),
            embedder=UnavailableEmbedder(),
        )
        await worker.start()
        task1 = worker._task
        await worker.start()
        assert worker._task is task1
        await worker.stop()
    finally:
        await kb.close()


async def test_worker_stop_never_started_no_exception(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(
            kb.db, kb.knowledge_store, EmbeddingRetryConfig(), embedder=NoneEmbedder()
        )
        await worker.stop()  # must not raise
    finally:
        await kb.close()


async def test_run_loop_survives_drain_once_exception(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=NoneEmbedder())
    try:
        worker = EmbeddingRetryWorker(
            kb.db,
            kb.knowledge_store,
            EmbeddingRetryConfig(poll_interval_seconds=0.01, backfill_interval_seconds=3600),
            embedder=UnavailableEmbedder(),
        )
        calls = {"n": 0}
        real_drain_once = worker.drain_once

        async def _flaky_drain_once(*, now: datetime) -> dict[str, int]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return await real_drain_once(now=now)

        worker.drain_once = _flaky_drain_once  # type: ignore[method-assign]
        await worker.start()
        try:
            for _ in range(500):
                if calls["n"] >= 2:
                    break
                await asyncio.sleep(0.01)
            assert calls["n"] >= 2
            assert worker.running is True
        finally:
            await worker.stop()
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Migration on reopen (AC32)
# ---------------------------------------------------------------------------


async def test_reopen_existing_db_gets_embedding_retry_queue_and_index(tmp_path) -> None:
    path = tmp_path / "persist.db"
    kb1 = await create_sqlite(path, embedder=NoneEmbedder())
    entry = await _create_entry(kb1)
    await kb1.close()

    kb2 = await create_sqlite(path, embedder=NoneEmbedder())
    try:
        cursor = await kb2.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_retry_queue'"
        )
        assert await cursor.fetchone() is not None

        cursor = await kb2.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_embed_queue_due'"
        )
        assert await cursor.fetchone() is not None

        refreshed = await kb2.get(entry.id)
        assert refreshed is not None
        assert refreshed.short_title == "t"

        await kb2.db.apply_schema(embedding_dim=1024)  # must not raise
    finally:
        await kb2.close()
