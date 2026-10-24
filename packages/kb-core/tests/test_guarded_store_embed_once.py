"""A guarded store reuses the near-duplicate guard's vector (embeds once)."""

from __future__ import annotations

from typing import Any

import pytest

from kb_core import create_sqlite
from kb_core.models.entry import EntryType, KnowledgeEntry

DIM = 1024


def _v(x: float) -> list[float]:
    v = [0.0] * DIM
    v[0] = 1.0
    v[1] = x
    return v


class _Counting:
    """Counting embedder; each call returns a distinct vector."""

    def __init__(self, *, fail: bool = False) -> None:
        self.embeds: list[str] = []
        self.batches: list[list[str]] = []
        self.stored: dict[str, list[float]] = {}
        self.fail = fail

    async def embed(self, text: str) -> list[float] | None:
        self.embeds.append(text)
        if self.fail:
            return None
        return _v(float(len(self.embeds)))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [_v(100.0 + i) for i in range(len(texts))]

    async def store_embedding(self, entry_id: str, vec: list[float]) -> None:
        self.stored[entry_id] = vec

    async def store_embeddings(self, pairs: list[tuple[str, list[float]]]) -> None:
        for entry_id, vec in pairs:
            self.stored[entry_id] = vec


@pytest.fixture
async def env(tmp_path: Any) -> Any:
    emb = _Counting()
    kb = await create_sqlite(tmp_path / "kb.db", embedder=emb)
    try:
        yield kb, emb
    finally:
        await kb.close()


def _fields(n: int = 0) -> dict[str, Any]:
    return {
        "short_title": f"title {n}",
        "long_title": f"long {n}",
        "knowledge_details": f"details {n}",
        "project_ref": "p",
        "enrich": False,
    }


async def _guard(kb: Any, f: dict[str, Any]) -> Any:
    return await kb.find_near_duplicates(
        short_title=f["short_title"],
        long_title=f["long_title"],
        knowledge_details=f["knowledge_details"],
        project_ref="p",
        floor=0.88,
    )


async def test_guarded_single_store_embeds_once(env: Any) -> None:
    kb, emb = env
    f = _fields()
    check = await _guard(kb, f)
    assert check.embedding is not None and check.embedded_text is not None
    entry = await kb.store(**f, precomputed_embedding=(check.embedded_text, check.embedding))
    assert len(emb.embeds) == 1
    assert emb.stored[entry.id] == check.embedding


async def test_guarded_batch_makes_no_embed_batch_call(env: Any) -> None:
    kb, emb = env
    dicts = []
    for i in range(3):
        f = _fields(i)
        f.pop("enrich")
        check = await _guard(kb, f)
        dicts.append({**f, "precomputed_embedding": (check.embedded_text, check.embedding)})
    created = await kb.store_batch(dicts, enrich=False)
    assert len(emb.embeds) == 3
    assert emb.batches == []
    assert len(created) == 3 and all(e.id in emb.stored for e in created)


async def test_batch_mixed_embeds_only_unmatched(env: Any) -> None:
    kb, emb = env
    f0, f1 = _fields(0), _fields(1)
    f0.pop("enrich")
    f1.pop("enrich")
    check = await _guard(kb, f0)
    created = await kb.store_batch(
        [{**f0, "precomputed_embedding": (check.embedded_text, check.embedding)}, f1],
        enrich=False,
    )
    assert len(emb.batches) == 1 and len(emb.batches[0]) == 1
    assert emb.stored[created[0].id] == check.embedding
    assert emb.stored[created[1].id] != check.embedding


async def test_text_mismatch_embeds_fresh(env: Any) -> None:
    kb, emb = env
    f = _fields()
    stale = [9.0] * DIM
    entry = await kb.store(**f, precomputed_embedding=("some other text", stale))
    assert len(emb.embeds) == 1
    assert emb.stored[entry.id] != stale


async def test_empty_vector_embeds_fresh(env: Any) -> None:
    kb, emb = env
    f = _fields()
    text = KnowledgeEntry.compose_embedding_text(
        f["short_title"], f["long_title"], f["knowledge_details"]
    )
    await kb.store(**f, precomputed_embedding=(text, []))
    assert len(emb.embeds) == 1


async def test_guard_skipped_store_embeds_once(env: Any) -> None:
    kb, emb = env
    f = _fields()
    f.update(entry_type=EntryType.MENTAL_MAP, knowledge_details="see kb-00001")
    await kb.store(**f)
    assert len(emb.embeds) == 1


async def test_embedder_unavailable_at_guard_then_store_retries(tmp_path: Any) -> None:
    emb = _Counting(fail=True)
    kb = await create_sqlite(tmp_path / "kb.db", embedder=emb)
    try:
        f = _fields()
        check = await _guard(kb, f)
        assert check.status == "embedder_unavailable"
        assert check.embedding is None and check.embedded_text is None
        pre = (
            (check.embedded_text, check.embedding)
            if check.embedding is not None and check.embedded_text is not None
            else None
        )
        entry = await kb.store(**f, precomputed_embedding=pre)
        assert len(emb.embeds) == 2
        cur = await kb.db.execute(
            "SELECT COUNT(*) AS n FROM embedding_retry_queue WHERE entry_id = ?", (entry.id,)
        )
        assert (await cur.fetchone())["n"] == 1
    finally:
        await kb.close()
