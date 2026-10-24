"""Shared re-embed engine and the vector-table dimension guard.

Env-free: callers (the stdio ``kb_maintain`` action and the ``kb-service reembed``
command) probe availability first, then hand in the embedder.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from kb_core import embedding_retry
from kb_core.db import queries

if TYPE_CHECKING:
    from collections.abc import Callable

    from kb_core.db.backend import Database
    from kb_core.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

REEMBED_MISSING_LIMIT = 1_000_000
"""Cap on entries selected by an only-missing re-embed."""

_SQLITE_DIM_RE = re.compile(r"FLOAT\[(\d+)\]")
_PG_DIM_RE = re.compile(r"vector\((\d+)\)")


class ReembedEmbedder(Protocol):
    """The embedder surface the re-embed engine needs."""

    async def is_available(self) -> bool:
        """True when the embedder can serve requests."""
        ...

    async def embed(self, text: str) -> list[float] | None:
        """Embed ``text``; None on failure."""
        ...

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        """Persist a vector for an entry."""
        ...


@dataclass(frozen=True)
class ReembedResult:
    """Outcome of a re-embed run."""

    processed: int
    succeeded: int
    failed: int
    failed_ids: list[str]


async def reembed_entries(
    db: Database,
    store: KnowledgeStore,
    embedder: ReembedEmbedder,
    *,
    only_missing: bool,
    on_progress: Callable[[int, int, int, int], None] | None = None,
    progress_every: int = 100,
) -> ReembedResult:
    """Re-embed active entries serially. Does not probe availability."""
    if only_missing:
        entry_ids = await store.get_entries_without_embeddings(limit=REEMBED_MISSING_LIMIT)
    else:
        entry_ids = await queries.get_all_active_entry_ids(db)

    total = len(entry_ids)
    succeeded_ids: list[str] = []
    failed_ids: list[str] = []
    done = 0
    for eid in entry_ids:
        try:
            entry = await queries.get_entry(db, eid)
            if entry is None:
                logger.warning("reembed failed entry_id=%s reason=%s", eid, "missing")
                failed_ids.append(eid)
            else:
                vec = await embedder.embed(entry.embedding_text)
                if vec is None:
                    logger.warning(
                        "reembed failed entry_id=%s reason=%s", eid, "embed_returned_none"
                    )
                    failed_ids.append(eid)
                else:
                    await embedder.store_embedding(eid, vec)
                    await store.mark_embedding(eid, True)
                    succeeded_ids.append(eid)
        except Exception as exc:
            logger.warning(
                "reembed failed entry_id=%s reason=%s",
                eid,
                f"exception:{type(exc).__name__}",
                exc_info=True,
            )
            failed_ids.append(eid)
        done += 1
        if on_progress is not None and done % progress_every == 0:
            on_progress(done, total, len(succeeded_ids), len(failed_ids))

    if on_progress is not None and total and done % progress_every != 0:
        on_progress(done, total, len(succeeded_ids), len(failed_ids))

    await embedding_retry.resolve(db, succeeded_ids)
    return ReembedResult(
        processed=total,
        succeeded=len(succeeded_ids),
        failed=len(failed_ids),
        failed_ids=failed_ids,
    )


def parse_vector_dim(type_sql: str) -> int | None:
    """Extract the vector dimension from a vec0 table DDL or a pgvector type string."""
    for pattern in (_SQLITE_DIM_RE, _PG_DIM_RE):
        match = pattern.search(type_sql)
        if match:
            return int(match.group(1))
    return None


async def vector_table_dim(db: Database) -> int | None:
    """Return the dimension of the ``knowledge_vec`` table, or None if unreadable."""
    from kb_core.db.sqlite_backend import SQLiteBackend

    if isinstance(db, SQLiteBackend):
        cursor = await db.execute("SELECT sql FROM sqlite_master WHERE name = 'knowledge_vec'")
        row = await cursor.fetchone()
        value = row["sql"] if row is not None else None
    else:
        cursor = await db.execute(
            "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
            "WHERE attrelid = 'knowledge_vec'::regclass AND attname = 'embedding'"
        )
        row = await cursor.fetchone()
        value = row["t"] if row is not None else None
    if not value:
        return None
    return parse_vector_dim(str(value))
