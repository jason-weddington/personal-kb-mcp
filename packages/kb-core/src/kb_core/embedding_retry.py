"""Self-healing embedding retry queue + background worker.

Motivation (KB kb-03276): today, when an entry fails to embed — the
embedder is down, the request errors, or the batch path silently drops
the vector — the entry is stored with ``has_embedding = 0`` and nothing
ever retries it. It becomes permanently invisible to vector search
unless an operator notices and runs a manual rebuild. This module adds
a durable queue (``embedding_retry_queue``, see ``kb_core.db.schema``
and ``kb_core.db.postgres_backend``) plus a background worker
(:class:`EmbeddingRetryWorker`) that drains it with exponential backoff
and self-heals without any restart or operator action.

Design constraints (see the GTD spec for the full rationale):

* Portable SQL only — ``?`` placeholders, ``ON CONFLICT ... DO UPDATE`` /
  ``DO NOTHING``, no ``INSERT OR IGNORE``, no ``SELECT ... FOR UPDATE
  SKIP LOCKED``. Both SQLite and Postgres run the exact same statements.
* Every write ends with ``await db.commit()`` — required on SQLite
  (no autocommit), a documented no-op on Postgres.
* All timestamps are UTC-aware ISO-8601 strings written through
  :func:`_iso`. Every public function takes an explicit ``now: datetime``
  and rejects a naive one — callers own the clock, this module never
  calls ``datetime.now()`` for queue timing (it stays purely testable).
* ``status`` has exactly two members: ``'pending'`` and ``'exhausted'``.
  In-flight work is expressed by pushing ``next_attempt_at`` forward
  (a lease), not by a third status value. A successful row is deleted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

from kb_core.db.queries import get_entry

if TYPE_CHECKING:
    from kb_core.config import EmbeddingConfig, EmbeddingRetryConfig
    from kb_core.db.backend import Database
    from kb_core.models.entry import KnowledgeEntry
    from kb_core.search.embedder_protocol import Embedder
    from kb_core.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class _RetryCapableEmbedder(Protocol):
    """Structural narrowing of ``Embedder`` covering what the worker actually calls.

    The public constructor signature stays ``embedder: Embedder | None`` per
    the facade contract (:mod:`kb_core.search.embedder_protocol`), but the
    worker also needs ``is_available``/``embed_batch``/``store_embedding`` —
    present on the production ``EmbeddingClient`` and every test stub, but
    not part of the narrower ``Embedder``/``BatchEmbedder`` Protocols. This
    local Protocol is a typing-only cast target; it changes no behavior.
    """

    async def is_available(self) -> bool: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None: ...
    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None: ...


# Greppable markers — mirror kb_core.graph.enricher.ENRICHMENT_FAILURE_MARKER.
EMBEDDING_RETRY_MARKER = "embedding-retry"
EMBEDDING_RETRY_EXHAUSTED_MARKER = "embedding-retry-exhausted"

# Backoff schedule for attempts 1-5 (seconds). Attempt 6 exhausts the row.
BACKOFF_SCHEDULE_SECONDS = (60, 300, 900, 3600, 10800)
MAX_ATTEMPTS = 6

assert len(BACKOFF_SCHEDULE_SECONDS) == MAX_ATTEMPTS - 1  # noqa: S101 (module-level invariant)


class EmbeddingQueueStats(TypedDict):
    """Operator-facing snapshot of the embedding retry queue."""

    pending: int
    exhausted: int
    oldest_pending_age_seconds: float | None
    next_due_at: str | None
    vectorless_unqueued: int


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """Render an aware ``datetime`` as a UTC ISO-8601 string (house convention)."""
    return dt.astimezone(UTC).isoformat()


def _require_aware(now: datetime) -> None:
    """Raise ``ValueError`` if ``now`` is naive — a naive value would corrupt ordering."""
    if now.tzinfo is None:
        msg = "embedding_retry: `now` must be timezone-aware"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Audit trail (fire-and-forget, mirrors store/knowledge_store.py:_record_audit_event)
# ---------------------------------------------------------------------------


async def _record_audit_event(
    db: Database,
    event_type: str,
    entry_id: str,
    *,
    detail: str | None = None,
) -> None:
    """Record an audit event. Fire-and-forget — failures never break the caller."""
    try:
        created_at = datetime.now(UTC).isoformat()
        await db.execute(
            "INSERT INTO audit_events (event_type, entry_id, contributor, detail, created_at)"
            " VALUES (?, ?, NULL, ?, ?)",
            (event_type, entry_id, detail, created_at),
        )
        await db.commit()
    except Exception:
        logger.warning(
            "%s: failed to record audit event %s for %s",
            EMBEDDING_RETRY_MARKER,
            event_type,
            entry_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public queue operations
# ---------------------------------------------------------------------------


async def enqueue(db: Database, entry_id: str, *, error: str, now: datetime) -> None:
    """Idempotently enqueue ``entry_id`` for embedding retry.

    A fresh row is scheduled ``now + BACKOFF_SCHEDULE_SECONDS[0]`` (60s) out —
    a store-time failure is not immediately re-attempted on the next poll.
    Re-enqueuing a ``'pending'`` row updates ``last_error``/``updated_at``
    only, preserving its existing backoff position. Re-enqueuing an
    ``'exhausted'`` row revives it: ``status`` -> ``'pending'``,
    ``attempts`` -> 0, ``next_attempt_at`` -> the fresh 60s-out value.
    """
    _require_aware(now)
    ts = _iso(now)
    next_attempt_at = _iso(now + timedelta(seconds=BACKOFF_SCHEDULE_SECONDS[0]))
    await db.execute(
        "INSERT INTO embedding_retry_queue "
        "(entry_id, attempts, last_error, next_attempt_at, status, created_at, updated_at) "
        "VALUES (?, 0, ?, ?, 'pending', ?, ?) "
        "ON CONFLICT(entry_id) DO UPDATE SET "
        "last_error = excluded.last_error, "
        "updated_at = excluded.updated_at, "
        "status = CASE WHEN embedding_retry_queue.status = 'exhausted' "
        "THEN 'pending' ELSE embedding_retry_queue.status END, "
        "attempts = CASE WHEN embedding_retry_queue.status = 'exhausted' "
        "THEN 0 ELSE embedding_retry_queue.attempts END, "
        "next_attempt_at = CASE WHEN embedding_retry_queue.status = 'exhausted' "
        "THEN excluded.next_attempt_at ELSE embedding_retry_queue.next_attempt_at END",
        (entry_id, error, next_attempt_at, ts, ts),
    )
    await db.commit()


async def backfill(
    db: Database, store: KnowledgeStore, *, now: datetime, limit: int = 10000
) -> int:
    """Queue every vectorless active entry; revive any exhausted queue rows among them.

    Source of truth is the existing
    :meth:`KnowledgeStore.get_entries_without_embeddings` (``has_embedding = 0
    AND is_active = 1``). Historical rows are due immediately (``next_attempt_at
    = now``), unlike :func:`enqueue`'s 60s-out fresh-failure offset. Pending
    rows are left completely untouched; only exhausted rows are revived.
    Returns the number of ids processed.
    """
    _require_aware(now)
    ids = await store.get_entries_without_embeddings(limit)
    ts = _iso(now)
    for entry_id in ids:
        await db.execute(
            "INSERT INTO embedding_retry_queue "
            "(entry_id, attempts, last_error, next_attempt_at, status, created_at, updated_at) "
            "VALUES (?, 0, NULL, ?, 'pending', ?, ?) "
            "ON CONFLICT(entry_id) DO UPDATE SET "
            "status = 'pending', attempts = 0, last_error = excluded.last_error, "
            "next_attempt_at = excluded.next_attempt_at, updated_at = excluded.updated_at "
            "WHERE embedding_retry_queue.status = 'exhausted'",
            (entry_id, ts, ts, ts),
        )
    await db.commit()
    n = len(ids)
    logger.info(
        "%s: backfill queued %d vectorless entries (limit=%d)", EMBEDDING_RETRY_MARKER, n, limit
    )
    if n == limit:
        logger.warning("%s: backfill scan truncated at limit=%d", EMBEDDING_RETRY_MARKER, limit)
    return n


async def resolve(db: Database, entry_ids: list[str]) -> None:
    """Remove queue rows for ``entry_ids`` (a manual rebuild already fixed them). No-op on empty."""
    if not entry_ids:
        return
    placeholders = ",".join("?" for _ in entry_ids)
    await db.execute(
        "DELETE FROM embedding_retry_queue WHERE entry_id IN ("  # noqa: S608
        + placeholders
        + ")",
        tuple(entry_ids),
    )
    await db.commit()


async def _vectorless_unqueued_count(db: Database) -> int:
    cursor = await db.execute(
        "SELECT COUNT(*) AS n FROM knowledge_entries e "
        "WHERE e.has_embedding = 0 AND e.is_active = 1 "
        "AND NOT EXISTS (SELECT 1 FROM embedding_retry_queue q WHERE q.entry_id = e.id)"
    )
    row = await cursor.fetchone()
    return int(row["n"]) if row is not None else 0


async def _pending_count(db: Database) -> int:
    cursor = await db.execute(
        "SELECT COUNT(*) AS n FROM embedding_retry_queue WHERE status = 'pending'"
    )
    row = await cursor.fetchone()
    return int(row["n"]) if row is not None else 0


async def queue_stats(db: Database, *, now: datetime) -> EmbeddingQueueStats:
    """Operator-facing snapshot — pending/exhausted counts, oldest age, next due, tripwire."""
    _require_aware(now)

    cursor = await db.execute(
        "SELECT COUNT(*) AS n FROM embedding_retry_queue WHERE status = 'pending'"
    )
    row = await cursor.fetchone()
    pending = int(row["n"]) if row is not None else 0

    cursor = await db.execute(
        "SELECT COUNT(*) AS n FROM embedding_retry_queue WHERE status = 'exhausted'"
    )
    row = await cursor.fetchone()
    exhausted = int(row["n"]) if row is not None else 0

    cursor = await db.execute(
        "SELECT MIN(created_at) AS m FROM embedding_retry_queue WHERE status = 'pending'"
    )
    row = await cursor.fetchone()
    oldest_created = row["m"] if row is not None else None
    oldest_pending_age_seconds: float | None = None
    if oldest_created:
        oldest_pending_age_seconds = (now - datetime.fromisoformat(oldest_created)).total_seconds()

    cursor = await db.execute(
        "SELECT MIN(next_attempt_at) AS m FROM embedding_retry_queue WHERE status = 'pending'"
    )
    row = await cursor.fetchone()
    next_due_at = row["m"] if row is not None else None

    vectorless_unqueued = await _vectorless_unqueued_count(db)

    return EmbeddingQueueStats(
        pending=pending,
        exhausted=exhausted,
        oldest_pending_age_seconds=oldest_pending_age_seconds,
        next_due_at=next_due_at,
        vectorless_unqueued=vectorless_unqueued,
    )


# ---------------------------------------------------------------------------
# Claim (compare-and-swap)
# ---------------------------------------------------------------------------


async def _claim_due(
    db: Database, *, now: datetime, batch_size: int, lease_seconds: float
) -> list[dict[str, Any]]:
    """Select due rows and CAS-claim each by pushing ``next_attempt_at`` out by a lease.

    A row whose CAS ``UPDATE`` reports ``rowcount == 0`` was claimed by
    another process (or its due time moved) and is dropped from the batch.
    Every CAS commits before any embedding call.
    """
    ts = _iso(now)
    cursor = await db.execute(
        "SELECT entry_id, attempts, next_attempt_at FROM embedding_retry_queue "
        "WHERE status = 'pending' AND next_attempt_at <= ? "
        "ORDER BY next_attempt_at LIMIT ?",
        (ts, batch_size),
    )
    candidates = await cursor.fetchall()
    lease_until = _iso(now + timedelta(seconds=lease_seconds))
    claimed: list[dict[str, Any]] = []
    for row in candidates:
        entry_id = row["entry_id"]
        attempts = row["attempts"]
        current_next_attempt_at = row["next_attempt_at"]
        cas_cursor = await db.execute(
            "UPDATE embedding_retry_queue SET next_attempt_at = ?, updated_at = ? "
            "WHERE entry_id = ? AND next_attempt_at = ? AND status = 'pending'",
            (lease_until, ts, entry_id, current_next_attempt_at),
        )
        await db.commit()
        if cas_cursor.rowcount == 0:
            continue
        claimed.append({"entry_id": entry_id, "attempts": attempts})
    return claimed


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class EmbeddingRetryWorker:
    """Background worker that drains :func:`_claim_due` rows and re-embeds them.

    Construct directly with ``embedder=<stub>`` for hermetic unit tests —
    :meth:`drain_once` is the unit-testable seam and never needs ``start()``.
    """

    def __init__(
        self,
        db: Database,
        store: KnowledgeStore,
        config: EmbeddingRetryConfig,
        embedding_config: EmbeddingConfig | None = None,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        """Wire in dependencies. See the `embedder=` injection contract on :meth:`start`."""
        self._db = db
        self._store = store
        self._config = config
        self._embedding_config = embedding_config
        self._embedder: Embedder | None = embedder
        self._owned_embedder = False
        self._task: asyncio.Task[None] | None = None
        self._embedder_available: bool | None = None
        self._skipped_cycles = 0

    @property
    def running(self) -> bool:
        """``True`` while the background task is alive."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the background task. Idempotent; no-ops (with a WARNING) when unembeddable."""
        if self._task is not None and not self._task.done():
            return
        if self._embedder is None:
            if self._embedding_config is not None:
                from kb_core.search.embeddings import EmbeddingClient

                self._embedder = EmbeddingClient(
                    self._db,
                    config=dataclasses.replace(
                        self._embedding_config, timeout=self._config.request_timeout
                    ),
                )
                self._owned_embedder = True
            else:
                logger.warning(
                    "%s: no embedder configured — worker not started (FTS-only deployment)",
                    EMBEDDING_RETRY_MARKER,
                )
                return
        logger.info(
            "%s: starting worker poll=%ss down_backoff=%ss batch=%d lease=%ss timeout=%ss",
            EMBEDDING_RETRY_MARKER,
            self._config.poll_interval_seconds,
            self._config.down_backoff_seconds,
            self._config.batch_size,
            self._config.lease_seconds,
            self._config.request_timeout,
        )
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        """Cancel the background task (safe if never started) and close an owned embedder."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "%s: worker task raised during stop", EMBEDDING_RETRY_MARKER, exc_info=True
                )
            self._task = None
        if self._owned_embedder and self._embedder is not None:
            close_fn = getattr(self._embedder, "close", None)
            if callable(close_fn):
                try:
                    await close_fn()
                except Exception:
                    logger.warning(
                        "%s: embedder close failed", EMBEDDING_RETRY_MARKER, exc_info=True
                    )

    async def _run_forever(self) -> None:
        """Crash-proof loop: backfill first, then drain/sleep forever. Never dies on error."""
        await self._safe_backfill()
        next_backfill = time.monotonic() + self._config.backfill_interval_seconds
        while True:
            sleep_for = self._config.poll_interval_seconds
            try:
                await self.drain_once(now=datetime.now(UTC))
                if self._embedder_available is False:
                    sleep_for = self._config.down_backoff_seconds
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "%s: drain cycle raised; worker continuing", EMBEDDING_RETRY_MARKER
                )
            if time.monotonic() >= next_backfill:
                await self._safe_backfill()
                next_backfill = time.monotonic() + self._config.backfill_interval_seconds
            await asyncio.sleep(sleep_for)

    async def _safe_backfill(self) -> None:
        try:
            await backfill(self._db, self._store, now=datetime.now(UTC))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s: backfill cycle raised; worker continuing", EMBEDDING_RETRY_MARKER)

    async def _track_availability(self, available: bool) -> None:
        """Log ONLY on True/None->False and False->True transitions."""
        previous = self._embedder_available
        if available:
            if previous is False:
                logger.info(
                    "%s: embedder recovered after %d skipped cycles",
                    EMBEDDING_RETRY_MARKER,
                    self._skipped_cycles,
                )
            self._skipped_cycles = 0
        else:
            if previous is not False:
                pending = await _pending_count(self._db)
                logger.warning(
                    "%s: embedder unreachable, deferring drain (pending=%d)",
                    EMBEDDING_RETRY_MARKER,
                    pending,
                )
            self._skipped_cycles += 1
        self._embedder_available = available

    async def _check_vectorless_tripwire(self) -> None:
        n = await _vectorless_unqueued_count(self._db)
        if n > 0:
            logger.warning(
                "%s: %d active entries have no vector and no queue row (invariant breach)",
                EMBEDDING_RETRY_MARKER,
                n,
            )

    async def drain_once(self, *, now: datetime) -> dict[str, int]:
        """Claim due rows, re-embed them, and reconcile the queue. The core unit-testable seam.

        Returns ``{"claimed", "embedded", "failed", "exhausted", "skipped"}``
        with ``claimed == embedded + failed + exhausted + skipped``.
        """
        _require_aware(now)
        zero = {"claimed": 0, "embedded": 0, "failed": 0, "exhausted": 0, "skipped": 0}

        if self._embedder is None:
            await self._check_vectorless_tripwire()
            return dict(zero)
        embedder = cast("_RetryCapableEmbedder", self._embedder)

        try:
            available = await embedder.is_available()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "%s: is_available() raised; treating embedder as down",
                EMBEDDING_RETRY_MARKER,
                exc_info=True,
            )
            available = False
        await self._track_availability(available)
        if not available:
            await self._check_vectorless_tripwire()
            return dict(zero)

        started = time.monotonic()
        claimed_rows = await _claim_due(
            self._db,
            now=now,
            batch_size=self._config.batch_size,
            lease_seconds=self._config.lease_seconds,
        )
        counters = dict(zero)
        counters["claimed"] = len(claimed_rows)
        if not claimed_rows:
            await self._check_vectorless_tripwire()
            return counters

        row_by_id = {row["entry_id"]: row for row in claimed_rows}
        entries: list[KnowledgeEntry] = []
        for row in claimed_rows:
            entry_id = row["entry_id"]
            entry = await get_entry(self._db, entry_id)
            if entry is None or not entry.is_active:
                await self._db.execute(
                    "DELETE FROM embedding_retry_queue WHERE entry_id = ?", (entry_id,)
                )
                await self._db.commit()
                counters["skipped"] += 1
                continue
            entries.append(entry)

        embed_ms = 0
        if entries:
            embed_started = time.monotonic()
            batch_exc: Exception | None = None
            try:
                vectors = await embedder.embed_batch([e.embedding_text for e in entries])
            except Exception as exc:
                vectors = None
                batch_exc = exc
            embed_ms = int((time.monotonic() - embed_started) * 1000)

            if vectors is None:
                shape = "exception" if batch_exc is not None else "embed_batch-returned-none"
                last_error = (
                    repr(batch_exc)[:500] if batch_exc is not None else "embed_batch returned None"
                )
                for entry in entries:
                    outcome = await self._apply_failure(
                        entry.id,
                        attempts=row_by_id[entry.id]["attempts"],
                        last_error=last_error,
                        shape=shape,
                        now=now,
                    )
                    counters[outcome] += 1
            else:
                for entry, vector in zip(entries, vectors, strict=True):
                    outcome = await self._apply_success(
                        embedder,
                        entry,
                        vector,
                        attempts=row_by_id[entry.id]["attempts"],
                        now=now,
                    )
                    counters[outcome] += 1

        if counters["claimed"] > 0:
            logger.info(
                "%s: drain claimed=%d embedded=%d failed=%d exhausted=%d skipped=%d "
                "in %dms (embed_batch %dms)",
                EMBEDDING_RETRY_MARKER,
                counters["claimed"],
                counters["embedded"],
                counters["failed"],
                counters["exhausted"],
                counters["skipped"],
                int((time.monotonic() - started) * 1000),
                embed_ms,
            )
        await self._check_vectorless_tripwire()
        return counters

    async def _apply_success(
        self,
        embedder: _RetryCapableEmbedder,
        entry: KnowledgeEntry,
        vector: list[float],
        *,
        attempts: int,
        now: datetime,
    ) -> str:
        """Persist a successful embed.

        On a post-embed write failure, apply the SAME failure accounting as
        an embed failure (increment attempts, reschedule, exhaust at
        MAX_ATTEMPTS) — a write failure must not retry forever at the lease
        interval. Returns "embedded", "failed", or "exhausted".
        """
        step = "store_embedding"
        try:
            await embedder.store_embedding(entry.id, vector)
            step = "mark_embedding"
            await self._store.mark_embedding(entry.id, True)
            step = "audit"
            await self._record_success_audit(entry.id, now=now)
            step = "delete_queue_row"
            await self._db.execute(
                "DELETE FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
            )
            await self._db.commit()
            return "embedded"
        except Exception as exc:
            logger.error(
                "%s: post-embed write failed for entry %s at step=%s",
                EMBEDDING_RETRY_MARKER,
                entry.id,
                step,
                exc_info=True,
            )
            return await self._apply_failure(
                entry.id,
                attempts=attempts,
                last_error=f"{step}: {exc!r}",
                shape=f"post_embed_write_failure:{step}",
                now=now,
            )

    async def _record_success_audit(self, entry_id: str, *, now: datetime) -> None:
        cursor = await self._db.execute(
            "SELECT attempts, created_at FROM embedding_retry_queue WHERE entry_id = ?",
            (entry_id,),
        )
        row = await cursor.fetchone()
        attempts = row["attempts"] if row is not None else 0
        queued_seconds = 0.0
        if row is not None and row["created_at"]:
            queued_seconds = (now - datetime.fromisoformat(row["created_at"])).total_seconds()
        await _record_audit_event(
            self._db,
            "embedding_recovered",
            entry_id,
            detail=f"attempts={attempts};queued_seconds={queued_seconds}",
        )

    async def _apply_failure(
        self,
        entry_id: str,
        *,
        attempts: int,
        last_error: str,
        shape: str,
        now: datetime,
    ) -> str:
        """Increment attempts + reschedule, or flip to 'exhausted' at MAX_ATTEMPTS."""
        new_attempts = attempts + 1
        ts = _iso(now)
        truncated_error = last_error[:500]
        delay = BACKOFF_SCHEDULE_SECONDS[min(new_attempts, len(BACKOFF_SCHEDULE_SECONDS)) - 1]
        next_attempt_at = _iso(now + timedelta(seconds=delay))

        if new_attempts >= MAX_ATTEMPTS:
            await self._db.execute(
                "UPDATE embedding_retry_queue SET attempts = ?, last_error = ?, "
                "status = 'exhausted', updated_at = ? WHERE entry_id = ?",
                (new_attempts, truncated_error, ts, entry_id),
            )
            await self._db.commit()
            await _record_audit_event(
                self._db,
                "embedding_exhausted",
                entry_id,
                detail=f"attempts={new_attempts};last_error={truncated_error}",
            )
            logger.error(
                "%s: entry %s exhausted %d embedding attempts; last_error=%s",
                EMBEDDING_RETRY_EXHAUSTED_MARKER,
                entry_id,
                new_attempts,
                truncated_error,
            )
            return "exhausted"

        await self._db.execute(
            "UPDATE embedding_retry_queue SET attempts = ?, last_error = ?, "
            "next_attempt_at = ?, updated_at = ? WHERE entry_id = ?",
            (new_attempts, truncated_error, next_attempt_at, ts, entry_id),
        )
        await self._db.commit()
        logger.warning(
            "%s: entry %s attempt %d/%d failed (shape=%s) — next attempt in %ds at %s; "
            "last_error=%s",
            EMBEDDING_RETRY_MARKER,
            entry_id,
            new_attempts,
            MAX_ATTEMPTS,
            shape,
            delay,
            next_attempt_at,
            truncated_error,
        )
        return "failed"
