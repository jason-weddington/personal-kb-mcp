"""LLM call ledger: persist kb-core :class:`LLMCallRecord` rows to ``llm_calls``.

:func:`record_llm_call` is the synchronous recorder handed to every
kb-service ``AnthropicLLMClient``. It schedules the insert as a task on the
running loop so the LLM call is never slowed, and a failed insert only logs.
:func:`drain_pending_llm_calls` awaits outstanding inserts at shutdown, and
:func:`prune_llm_calls` drops rows past the retention window (run from the
surprise drain).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from kb_core.llm.provider import LLMCallRecord

from kb_service import database

if TYPE_CHECKING:
    from kb_service.db_types import DbPool

logger = logging.getLogger(__name__)

_COLUMNS = [f.name for f in dataclasses.fields(LLMCallRecord)]
_INSERT_SQL = (
    f"INSERT INTO llm_calls ({', '.join(_COLUMNS)}) "  # noqa: S608 - fixed columns
    f"VALUES ({', '.join(f'${i}' for i in range(1, len(_COLUMNS) + 1))})"
)

_PENDING: set[asyncio.Task[None]] = set()
"""Inserts scheduled by :func:`record_llm_call` that have not finished yet."""

LLM_CALLS_RETENTION_DAYS = 90
"""Days an llm_calls row is kept before :func:`prune_llm_calls` deletes it."""


async def insert_llm_call(pool: DbPool, record: LLMCallRecord) -> None:
    """Insert one record into ``llm_calls``."""
    await pool.execute(_INSERT_SQL, *dataclasses.astuple(record))


async def _write(record: LLMCallRecord) -> None:
    """Insert *record* through the open service pool; skip when none is open."""
    pool = database.current_pool()
    if pool is None:
        logger.debug("llm_calls insert skipped: no service pool")
        return
    await insert_llm_call(pool, record)


def record_llm_call(record: LLMCallRecord) -> None:
    """Schedule an insert of *record* on the running loop and return at once."""
    task = asyncio.get_running_loop().create_task(_write(record))
    _PENDING.add(task)

    def _done(t: asyncio.Task[None]) -> None:
        _PENDING.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logger.warning(
                "llm_calls insert failed purpose=%s model=%s response_id=%s: %r",
                record.purpose,
                record.model,
                record.response_id,
                t.exception(),
            )

    task.add_done_callback(_done)


async def drain_pending_llm_calls(timeout: float = 5.0) -> None:
    """Wait up to *timeout* seconds for pending inserts, then cancel the rest.

    Never raises.
    """
    try:
        loop = asyncio.get_running_loop()
        for task in [t for t in _PENDING if t.get_loop() is not loop]:
            _PENDING.discard(task)
        tasks = set(_PENDING)
        if not tasks:
            return
        _done, not_done = await asyncio.wait(tasks, timeout=timeout)
        if not_done:
            logger.warning("llm_calls drain timed out pending=%d", len(not_done))
            for task in not_done:
                task.cancel()
                _PENDING.discard(task)
    except Exception:
        logger.warning("llm_calls drain failed", exc_info=True)


def _count(status: str) -> int:
    return int(status.rsplit(" ", 1)[1])


async def prune_llm_calls(
    pool: DbPool,
    older_than_days: int = LLM_CALLS_RETENTION_DAYS,
    now: datetime | None = None,
) -> int:
    """Delete llm_calls rows older than the cutoff; return how many were deleted."""
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=older_than_days)).isoformat(
        timespec="seconds"
    )
    status = await pool.execute("DELETE FROM llm_calls WHERE ts < $1", cutoff)
    return _count(status)
