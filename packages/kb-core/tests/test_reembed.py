"""Tests for ``kb_core.search.reembed`` on a SQLite KB with an in-process fake embedder."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from kb_core import create_sqlite
from kb_core.config import EmbeddingConfig
from kb_core.embedding_retry import enqueue
from kb_core.models.entry import EntryType
from kb_core.search.embeddings import EmbeddingClient
from kb_core.search.reembed import (
    REEMBED_MISSING_LIMIT,
    parse_vector_dim,
    reembed_entries,
    vector_table_dim,
)

DIM = 8


class FakeEmbedder(EmbeddingClient):
    """Deterministic embedder; returns None for texts in ``fail_texts``."""

    def __init__(self, db: Any, fail_texts: frozenset[str] = frozenset()) -> None:
        super().__init__(db, EmbeddingConfig(dim=DIM))
        self.fail_texts = fail_texts

    async def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        if text in self.fail_texts:
            return None
        return [1.0] + [0.0] * (DIM - 1)


async def _entry(kb: Any, title: str) -> Any:
    return await kb.knowledge_store.create_entry(
        short_title=title,
        long_title=title,
        knowledge_details=title,
        entry_type=EntryType.FACTUAL_REFERENCE,
    )


async def _vec_count(kb: Any) -> int:
    cur = await kb.db.execute("SELECT COUNT(*) AS n FROM knowledge_vec")
    row = await cur.fetchone()
    return int(row["n"])


async def _flags(kb: Any) -> dict[str, int]:
    cur = await kb.db.execute("SELECT id, has_embedding FROM knowledge_entries")
    return {r["id"]: r["has_embedding"] for r in await cur.fetchall()}


@pytest.fixture
async def kb(tmp_path):
    kb = await create_sqlite(tmp_path / "kb.db", embedding_dim=DIM)
    yield kb
    await kb.close()


def test_missing_limit_constant() -> None:
    assert REEMBED_MISSING_LIMIT == 1_000_000


async def test_all_mode_reembeds_everything_active(kb) -> None:
    a = await _entry(kb, "a")
    b = await _entry(kb, "b")
    c = await _entry(kb, "c")
    embedder = FakeEmbedder(kb.db)
    await embedder.store_embedding(a.id, [0.5] + [0.0] * (DIM - 1))
    await kb.knowledge_store.mark_embedding(a.id, True)
    await kb.knowledge_store.deactivate_entry(c.id)

    result = await reembed_entries(kb.db, kb.knowledge_store, embedder, only_missing=False)

    assert (result.processed, result.succeeded, result.failed) == (2, 2, 0)
    flags = await _flags(kb)
    assert flags[a.id] == 1 and flags[b.id] == 1 and flags[c.id] == 0


async def test_only_missing(kb) -> None:
    a = await _entry(kb, "a")
    await _entry(kb, "b")
    embedder = FakeEmbedder(kb.db)
    await embedder.store_embedding(a.id, [1.0] + [0.0] * (DIM - 1))
    await kb.knowledge_store.mark_embedding(a.id, True)

    result = await reembed_entries(kb.db, kb.knowledge_store, embedder, only_missing=True)

    assert result.processed == 1
    assert await _vec_count(kb) == 2


async def test_failure_is_reported_and_logged(kb, caplog) -> None:
    a = await _entry(kb, "a")
    b = await _entry(kb, "b")
    embedder = FakeEmbedder(kb.db, frozenset({a.embedding_text}))
    with caplog.at_level(logging.WARNING, logger="kb_core.search.reembed"):
        result = await reembed_entries(kb.db, kb.knowledge_store, embedder, only_missing=False)

    assert result.failed == 1
    assert result.failed_ids == [a.id]
    assert f"reembed failed entry_id={a.id} reason=embed_returned_none" in caplog.text
    flags = await _flags(kb)
    assert flags[a.id] == 0 and flags[b.id] == 1


async def test_retry_queue_resolved_for_succeeded_only(kb) -> None:
    a = await _entry(kb, "a")
    b = await _entry(kb, "b")
    await enqueue(kb.db, a.id, error="x", now=datetime.now(UTC))
    await enqueue(kb.db, b.id, error="x", now=datetime.now(UTC))
    embedder = FakeEmbedder(kb.db, frozenset({a.embedding_text}))

    await reembed_entries(kb.db, kb.knowledge_store, embedder, only_missing=False)

    cur = await kb.db.execute("SELECT entry_id FROM embedding_retry_queue")
    assert [r["entry_id"] for r in await cur.fetchall()] == [a.id]


@pytest.mark.parametrize(("n", "expected"), [(5, [2, 4, 5]), (4, [2, 4])])
async def test_progress_callback(kb, n, expected) -> None:
    for i in range(n):
        await _entry(kb, f"e{i}")
    calls: list[int] = []
    await reembed_entries(
        kb.db,
        kb.knowledge_store,
        FakeEmbedder(kb.db),
        only_missing=False,
        on_progress=lambda done, total, ok, bad: calls.append(done),
        progress_every=2,
    )
    assert calls == expected


async def test_no_progress_on_empty(kb) -> None:
    calls: list[int] = []
    result = await reembed_entries(
        kb.db,
        kb.knowledge_store,
        FakeEmbedder(kb.db),
        only_missing=False,
        on_progress=lambda *a: calls.append(1),
    )
    assert result.processed == 0
    assert calls == []


def test_parse_vector_dim() -> None:
    ddl = (
        "CREATE VIRTUAL TABLE knowledge_vec USING vec0(entry_id TEXT PRIMARY KEY, "
        "embedding FLOAT[1024] distance_metric=cosine)"
    )
    assert parse_vector_dim(ddl) == 1024
    assert parse_vector_dim("vector(512)") == 512
    assert parse_vector_dim("text") is None


async def test_vector_table_dim_sqlite(tmp_path) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedding_dim=1024)
    try:
        assert await vector_table_dim(kb.db) == 1024
    finally:
        await kb.close()
