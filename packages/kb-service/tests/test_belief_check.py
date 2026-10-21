"""POST /api/kb/belief-check: chunking, candidates, scoring, cache, warm task, route."""

import asyncio
import logging
import math
import re
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from kb_core import create_sqlite

from kb_service import belief_check as bc
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import BeliefCheckReason
from kb_service.prevention import LoadStats, Resolution, load_resolutions
from kb_service.routes import belief_check_routes
from tests.conftest import (
    FakeKnowledgeBase,
    ScriptedEmbedder,
    ScriptedSingleEmbedder,
    fake_user,
)

PUSH = "Push feature branches to the mirror remote."
_ROUTE_LOGGER = "kb_service.routes.belief_check_routes"
_MODULE_LOGGER = "kb_service.belief_check"


@pytest.fixture(autouse=True)
async def _reset() -> AsyncIterator[None]:
    bc.reset_state()
    yield
    task = bc._warm_task
    if (
        task is not None
        and not task.done()
        and task.get_loop() is asyncio.get_running_loop()
    ):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    bc.reset_state()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "KB_SOFT_GATE_DISABLED_PROJECTS",
        "KB_DELIVER_OBSERVED_ONCE",
        "KB_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


# --- helpers -------------------------------------------------------------------


def _r(
    entry_id: str,
    wrong_belief: str = "",
    *,
    updated_at: str = "2026-10-01T00:00:00+00:00",
    capture: str | None = "deliberate",
    grounding: str | None = "observed",
    observed_sessions: int = 1,
    observed_once: bool = False,
    scope: str = "project",
) -> Resolution:
    return Resolution(
        entry_id=entry_id,
        updated_at=updated_at,
        wrong_belief=wrong_belief or f"wrong belief of {entry_id}",
        corrected_fact=f"corrected fact of {entry_id}",
        evidence="",
        cue_tool="",
        cue_target_class="",
        capture=capture,
        grounding=grounding,
        observed_sessions=observed_sessions,
        observed_once=observed_once,
        scope=scope,
    )


class Ctl(ScriptedEmbedder):
    """A ScriptedEmbedder whose wrong-belief (warm) and chunk batches differ."""

    def __init__(
        self,
        vectors: dict[str, list[float]],
        default: list[float] | None = None,
        *,
        beliefs: set[str] | None = None,
        slow: set[str] | None = None,
        warm_delay: float = 0.0,
        chunk_delay: float = 0.0,
        warm_mode: str = "ok",
        chunk_mode: str = "ok",
    ) -> None:
        super().__init__(vectors, default)
        self.beliefs = beliefs if beliefs is not None else set()
        self.slow = slow if slow is not None else self.beliefs
        self.warm_delay = warm_delay
        self.chunk_delay = chunk_delay
        self.warm_mode = warm_mode
        self.chunk_mode = chunk_mode

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        self.calls.append(list(texts))
        warm = any(t in self.beliefs for t in texts)
        if warm and any(t in self.slow for t in texts) and self.warm_delay > 0:
            await asyncio.sleep(self.warm_delay)
        if not warm and self.chunk_delay > 0:
            await asyncio.sleep(self.chunk_delay)
        mode = self.warm_mode if warm else self.chunk_mode
        if mode == "none":
            return None
        if mode == "raise":
            raise RuntimeError("boom")
        if mode == "timeout":
            raise TimeoutError
        out = [self._vector(t) for t in texts]
        if mode == "short":
            return out[:-1]
        if mode == "zero":
            return [[0.0] * len(v) for v in out]
        if mode == "nan-first":
            return [[math.nan] * len(out[0]), *out[1:]]
        return out

    def warm_calls(self) -> list[list[str]]:
        return [c for c in self.calls if any(t in self.beliefs for t in c)]


class _Kb:
    """Minimal KB: an embedder and a db handle (load_resolutions is patched)."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.db = None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        out: list[list[float]] | None = await self.embedder.embed_batch(texts)
        return out


def _patch_load(
    monkeypatch: pytest.MonkeyPatch,
    by_project: dict[str, list[Resolution]] | list[Resolution],
    delay: float = 0.0,
) -> None:
    async def fake(
        db: Any, project: str, include_observed_once: bool
    ) -> tuple[list[Resolution], LoadStats]:
        assert include_observed_once is False
        if delay:
            await asyncio.sleep(delay)
        if isinstance(by_project, list):
            return list(by_project), LoadStats()
        return list(by_project.get(project, [])), LoadStats()

    monkeypatch.setattr(bc, "load_resolutions", fake)


def _two_d(c: float) -> list[float]:
    return [c, math.sqrt(1 - c * c)]


def _five_setup() -> tuple[list[Resolution], Ctl]:
    cs = [0.95, 0.90, 0.40, 0.30, 0.20]
    cands = [_r(f"kb-0000{i}", f"belief number {i} here") for i in range(5)]
    vectors = {r.wrong_belief: _two_d(c) for r, c in zip(cands, cs, strict=True)}
    vectors[PUSH] = [1.0, 0.0]
    emb = Ctl(vectors, beliefs={r.wrong_belief for r in cands})
    return cands, emb


# --- AC3 chunking ----------------------------------------------------------------


def test_chunk_text_short_input_is_empty() -> None:
    assert bc.chunk_text("Done.", None) == []


def test_chunk_text_long_run_is_windowed() -> None:
    text = " ".join(f"w{i:03d}" for i in range(140))
    assert len(text) == 699
    assert [len(c.text) for c in bc.chunk_text(text, None)] == [299, 299, 99]


def test_chunk_text_keeps_last_sixteen() -> None:
    text = "".join(f"Sentence number {i:02d} is here. " for i in range(20))
    chunks = bc.chunk_text(text, None)
    assert len(chunks) == 16
    assert chunks[0].text == "Sentence number 04 is here."
    assert chunks[-1].text == "Sentence number 19 is here."


def test_chunk_text_truncates_input_to_its_tail() -> None:
    text = (
        "Early words that must vanish. "
        + "filler words here. " * 600
        + "Final sentence that must stay."
    )
    assert [c.text for c in bc.chunk_text(text, None)] == [
        "filler words here.",
        "Final sentence that must stay.",
    ]


def test_chunk_text_drops_short_tail_window() -> None:
    text = "a" * 299 + " " + "b" * 10
    assert [c.text for c in bc.chunk_text(text, None)] == ["a" * 299]


def test_chunk_text_hard_splits_a_word_without_spaces() -> None:
    assert [len(c.text) for c in bc.chunk_text("x" * 350, None)] == [300, 50]


def test_chunk_text_dedup_keeps_last() -> None:
    got = bc.chunk_text(
        "Same sentence repeated here.",
        "Same sentence repeated here. Reasoning only line.",
    )
    assert got == [
        bc.Chunk("reasoning", "Reasoning only line."),
        bc.Chunk("text", "Same sentence repeated here."),
    ]


def test_chunk_text_reasoning_first() -> None:
    got = bc.chunk_text(
        "Tests are green. Push feature branches to the mirror remote.",
        "The branch is ready for review, so it should go to the shared remote now.",
    )
    assert got == [
        bc.Chunk(
            "reasoning",
            "The branch is ready for review, so it should go to the shared remote now.",
        ),
        bc.Chunk("text", "Tests are green."),
        bc.Chunk("text", "Push feature branches to the mirror remote."),
    ]


def test_chunk_text_splits_on_newlines() -> None:
    got = bc.chunk_text(
        "line one is long enough\n\n  line two is long enough too", None
    )
    assert [c.text for c in got] == [
        "line one is long enough",
        "line two is long enough too",
    ]


# --- AC4 candidate rule ------------------------------------------------------------


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Any]:
    kb = await create_sqlite(
        tmp_path / "kb.db",
        embedder=None,
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
    )
    try:
        yield kb
    finally:
        await kb.close()


_COUNTER = {"n": 0}


async def _store(
    kb: Any, project: str = "p", hints_extra: dict[str, Any] | None = None, **res: Any
) -> str:
    _COUNTER["n"] += 1
    n = _COUNTER["n"]
    hints: dict[str, Any] = {"resolution": res}
    if hints_extra:
        hints.update(hints_extra)
    entry = await kb.store(
        short_title=f"Resolution {n}",
        long_title=f"Resolution number {n}",
        knowledge_details=f"details {n}",
        project_ref=project,
        hints=hints,
        enrich=False,
    )
    return str(entry.id)


def _res(wrong_belief: str = PUSH, **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "corrected_fact": "Push feature branches to origin.",
        "wrong_belief": wrong_belief,
        "provenance": {"capture": "deliberate", "grounding": "asserted"},
    }
    out.update(kw)
    return out


async def test_select_candidates_on_a_real_kb(kb: Any) -> None:
    yes_own = await _store(kb, "p", **_res())
    yes_global = await _store(
        kb,
        "q",
        **_res(
            provenance={"capture": "deliberate", "grounding": "observed"},
            scope="global",
        ),
    )
    other_project = await _store(kb, "q", **_res())
    empty = await _store(kb, "p", **_res(wrong_belief=""))
    blank = await _store(kb, "p", **_res(wrong_belief="   "))
    asserted = await _store(
        kb, "p", **_res(provenance={"capture": "autonomous", "grounding": "asserted"})
    )
    unlabelled = await _store(
        kb, "p", corrected_fact="Push to origin.", wrong_belief=PUSH
    )
    once = await _store(
        kb, "p", **_res(provenance={"capture": "autonomous", "grounding": "observed"})
    )
    twice = await _store(
        kb,
        "p",
        **_res(
            provenance={"capture": "autonomous", "grounding": "observed"},
            observed_sessions=2,
        ),
    )
    resolutions, _ = await load_resolutions(kb.db, "p", include_observed_once=False)
    kept, truncated = bc.select_candidates(resolutions)
    ids = {r.entry_id for r in kept}
    assert {yes_own, yes_global, twice} <= ids
    for no in (other_project, empty, blank, asserted, unlabelled, once):
        assert no not in ids
    assert truncated == 0


def test_select_candidates_caps_at_500() -> None:
    cands = [_r(f"kb-{i:05d}") for i in range(505)]
    kept, truncated = bc.select_candidates(cands)
    assert kept == cands[:500]
    assert truncated == 5


# --- AC5 vectors and scoring ----------------------------------------------------------


def test_score_candidates_ranks_by_cosine() -> None:
    cs = [0.95, 0.90, 0.40, 0.30, 0.20]
    cands = []
    for i, c in enumerate(cs):
        vec = bc.unit_vector(_two_d(c))
        assert vec is not None
        cands.append((_r(f"kb-{i}"), vec))
    chunk = bc.unit_vector([1.0, 0.0])
    assert chunk is not None
    scored = bc.score_candidates([chunk], list(reversed(cands)))
    assert [s.score for s in scored] == pytest.approx(cs, abs=1e-9)


def test_score_candidates_max_over_chunks() -> None:
    cand = bc.unit_vector([0.1, 0.1, 0.9, 0.1, 0.1])
    assert cand is not None
    chunks = []
    for i in range(5):
        v = bc.unit_vector([1.0 if j == i else 0.0 for j in range(5)])
        assert v is not None
        chunks.append(v)
    [s] = bc.score_candidates(chunks, [(_r("kb-1"), cand)])
    assert s.score == pytest.approx(0.9 / math.sqrt(0.85), abs=1e-9)
    assert s.chunk_index == 2
    assert isinstance(s.score, float)


def test_score_candidates_ties_by_entry_id() -> None:
    v = bc.unit_vector([1.0, 1.0])
    assert v is not None
    scored = bc.score_candidates([v], [(_r("kb-2"), v), (_r("kb-1"), v)])
    assert [s.resolution.entry_id for s in scored] == ["kb-1", "kb-2"]


def test_score_candidates_empty() -> None:
    v = bc.unit_vector([1.0])
    assert v is not None
    assert bc.score_candidates([], [(_r("kb-1"), v)]) == []
    assert bc.score_candidates([v], []) == []


@pytest.mark.parametrize(
    "raw",
    [
        [],
        [0.0, 0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [[1.0, 0.0], [0.0, 1.0]],
        ["x", 1.0],
    ],
)
def test_unit_vector_rejects_unusable(raw: Any) -> None:
    assert bc.unit_vector(raw) is None


def test_unit_vector_normalises() -> None:
    v = bc.unit_vector([3.0, 4.0])
    assert v is not None
    assert v.dtype == np.float64
    assert list(v) == pytest.approx([0.6, 0.8])


# --- AC6 cache ------------------------------------------------------------------------


def test_cached_vector_keys_on_updated_at_and_belief() -> None:
    r = _r("kb-1", "the belief")
    assert bc._store_vector(r, [1.0, 0.0])
    assert bc.cached_vector(r) is not None
    later = _r("kb-1", "the belief", updated_at="2026-10-02T00:00:00+00:00")
    assert bc.cached_vector(later) is None
    reworded = _r("kb-1", "another belief")
    assert bc.cached_vector(reworded) is None
    assert not bc._store_vector(_r("kb-2"), [0.0, 0.0])
    assert "kb-2" not in bc._VECTOR_CACHE
    bc._store_vector(later, [0.0, 1.0])
    assert len(bc._VECTOR_CACHE) == 1
    assert bc.cached_vector(r) is None


async def test_second_check_embeds_only_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    cands, emb = _five_setup()
    _patch_load(monkeypatch, cands)
    kb = _Kb(emb)
    first = await bc.check_beliefs(kb, "p", PUSH, None)
    assert first.reason == "checked"
    assert len(emb.calls) == 2
    emb.calls.clear()
    second = await bc.check_beliefs(kb, "p", PUSH, None)
    assert second.reason == "checked"
    assert second.warm == "none"
    assert emb.calls == [[PUSH]]


async def _reopen(path: Path, embedder: Any) -> Any:
    return await create_sqlite(
        path,
        embedder=embedder,
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
    )


async def test_update_reembeds_the_wrong_belief(tmp_path: Path) -> None:
    path = tmp_path / "kb.db"
    seed = await _reopen(path, None)
    entry_id = await _store(seed, "p", **_res())
    await seed.close()
    emb = ScriptedEmbedder({}, default=[1.0, 0.0])
    kb = await _reopen(path, emb)
    try:
        out = await bc.check_beliefs(kb, "p", PUSH, None)
        assert out.reason == "checked"
        assert out.matches[0].entry_id == entry_id
    finally:
        await kb.close()
    seed = await _reopen(path, None)
    await seed.update(
        entry_id,
        hints={"resolution": _res(wrong_belief="Push straight to the mirror.")},
        enrich=False,
    )
    await seed.close()
    kb = await _reopen(path, emb)
    try:
        emb.calls.clear()
        out = await bc.check_beliefs(kb, "p", PUSH, None)
        assert out.reason == "checked"
        assert ["Push straight to the mirror."] in emb.calls
    finally:
        await kb.close()


@pytest.mark.parametrize("embedder_cls", [ScriptedEmbedder, ScriptedSingleEmbedder])
async def test_real_kb_path(tmp_path: Path, embedder_cls: Any) -> None:
    path = tmp_path / "kb.db"
    seed = await _reopen(path, None)
    await _store(seed, "p", **_res(provenance={"capture": "deliberate"}))
    await seed.close()
    kb = await _reopen(path, embedder_cls({}, default=[1.0, 0.0]))
    try:
        out = await bc.check_beliefs(kb, "p", PUSH, None)
    finally:
        await kb.close()
    assert out.reason == "checked"
    assert out.scored == out.candidates == 1


# --- AC7 warm task --------------------------------------------------------------------


def _warm_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == _MODULE_LOGGER and r.getMessage().startswith("belief_check_warm ")
    ]


async def test_warm_batches(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(bc, "BELIEF_CHECK_WARM_BATCH", 2)
    todo = [_r(f"kb-{i}") for i in range(5)]
    emb = ScriptedEmbedder({}, default=[1.0, 0.0])
    await bc._warm(_Kb(emb), "p", todo)
    assert [len(c) for c in emb.calls] == [2, 2, 1]
    assert all(bc.cached_vector(r) is not None for r in todo)
    [rec] = _warm_records(caplog)
    assert rec.levelno == logging.INFO
    msg = rec.getMessage()
    assert "outcome=ok project=p requested=5 stored=5 unusable=0 batches=3" in msg
    assert msg.endswith("exc=")


@pytest.mark.parametrize(
    ("mode", "want"),
    [
        ("none", "outcome=none"),
        ("short", "outcome=length-mismatch"),
        ("raise", "outcome=error"),
    ],
)
async def test_warm_failures(
    caplog: pytest.LogCaptureFixture, mode: str, want: str
) -> None:
    todo = [_r("kb-1"), _r("kb-2")]
    emb = Ctl({}, [1.0, 0.0], beliefs={r.wrong_belief for r in todo}, warm_mode=mode)
    await bc._warm(_Kb(emb), "p", todo)
    [rec] = _warm_records(caplog)
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert want in msg
    assert "stored=0" in msg
    if mode == "raise":
        assert msg.endswith("exc=RuntimeError")
    assert bc._VECTOR_CACHE == {}


async def test_warm_skips_unusable_vectors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    todo = [_r("kb-1"), _r("kb-2")]
    emb = Ctl(
        {}, [1.0, 0.0], beliefs={r.wrong_belief for r in todo}, warm_mode="nan-first"
    )
    await bc._warm(_Kb(emb), "p", todo)
    assert bc.cached_vector(todo[0]) is None
    assert bc.cached_vector(todo[1]) is not None
    [rec] = _warm_records(caplog)
    assert "stored=1 unusable=1" in rec.getMessage()


async def test_warm_cancelled_logs_and_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    todo = [_r("kb-1")]
    emb = Ctl({}, [1.0, 0.0], beliefs={todo[0].wrong_belief}, warm_delay=1.0)
    task = asyncio.create_task(bc._warm(_Kb(emb), "p", todo))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [rec] = _warm_records(caplog)
    assert "outcome=cancelled" in rec.getMessage()


async def test_concurrent_checks_share_one_warm_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 1.0)
    a = [_r("kb-a1", "alpha belief one"), _r("kb-a2", "alpha belief two")]
    b = [_r("kb-b1", "bravo belief one"), _r("kb-b2", "bravo belief two")]
    _patch_load(monkeypatch, {"proj-a": a, "proj-b": b})
    emb = Ctl(
        {},
        [1.0, 0.0],
        beliefs={r.wrong_belief for r in a + b},
        slow={r.wrong_belief for r in a},
        warm_delay=0.2,
    )
    kb = _Kb(emb)
    task_a = asyncio.create_task(bc.check_beliefs(kb, "proj-a", PUSH, None))
    await asyncio.sleep(0.05)
    out_b = await bc.check_beliefs(kb, "proj-b", PUSH, None)
    assert (out_b.reason, out_b.detail, out_b.warm) == ("warming", "busy", "busy")
    assert out_b.candidates == 2
    assert out_b.scored == 0
    out_a = await task_a
    assert out_a.reason == "checked"
    assert out_a.scored == 2
    assert len(emb.warm_calls()) == 1
    assert not any("bravo belief one" in c for c in emb.calls)
    out_b2 = await bc.check_beliefs(kb, "proj-b", PUSH, None)
    assert out_b2.warm == "started"
    assert out_b2.reason == "checked"
    assert out_b2.scored == 2


async def test_joined_warm_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 1.0)
    cands = [_r("kb-1"), _r("kb-2")]
    _patch_load(monkeypatch, cands)
    emb = Ctl({}, [1.0, 0.0], beliefs={r.wrong_belief for r in cands}, warm_delay=0.2)
    kb = _Kb(emb)
    first = asyncio.create_task(bc.check_beliefs(kb, "p", PUSH, None))
    await asyncio.sleep(0.05)
    second = await bc.check_beliefs(kb, "p", PUSH, None)
    assert second.warm == "joined"
    assert second.reason == "checked"
    assert (await first).warm == "started"
    assert len(emb.warm_calls()) == 1


async def test_foreign_loop_warm_task_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cands = [_r("kb-1")]
    _patch_load(monkeypatch, cands)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def make() -> asyncio.Task[None]:
        return asyncio.get_running_loop().create_task(asyncio.sleep(30))

    planted = asyncio.run_coroutine_threadsafe(make(), loop).result(timeout=5)
    try:
        monkeypatch.setattr(bc, "_warm_task", planted)
        monkeypatch.setattr(bc, "_warm_ids", frozenset({"kb-1"}))
        emb = ScriptedEmbedder({}, default=[1.0, 0.0])
        out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
        assert out.warm == "started"
        assert out.reason == "checked"
        assert not planted.done()
    finally:
        loop.call_soon_threadsafe(planted.cancel)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


# --- AC8 check_beliefs ---------------------------------------------------------------


async def test_check_beliefs_top_three_includes_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cands, emb = _five_setup()
    _patch_load(monkeypatch, cands)
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert out.reason == "checked"
    assert [m.score for m in out.matches] == pytest.approx([0.95, 0.90, 0.40], abs=1e-9)
    assert out.matches[2].score < bc.BELIEF_CHECK_THRESHOLD
    assert out.matches[0].wrong_belief == cands[0].wrong_belief
    assert out.matches[0].provenance_label == "deliberate/observed"
    assert out.top_chunk_idx == 0
    assert out.top_chunk_src == "text"
    assert out.dim == 2
    assert (out.candidates, out.scored, out.chunks) == (5, 5, 1)


async def test_check_beliefs_no_embedder() -> None:
    assert (await bc.check_beliefs(_Kb(None), "p", PUSH, None)).reason == "no-embedder"


async def test_check_beliefs_no_chunks() -> None:
    emb = ScriptedEmbedder({}, default=[1.0])
    assert (await bc.check_beliefs(_Kb(emb), "p", "Done.", None)).reason == "no-chunks"


async def test_check_beliefs_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, [_r("kb-1", capture="autonomous", grounding="asserted")])
    emb = ScriptedEmbedder({}, default=[1.0])
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert out.reason == "no-candidates"


async def test_check_beliefs_load_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 0.05)
    _patch_load(monkeypatch, [_r("kb-1")], delay=0.2)
    emb = ScriptedEmbedder({}, default=[1.0])
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert (out.reason, out.detail) == ("timeout", "load")


async def test_check_beliefs_load_timeout_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError

    monkeypatch.setattr(bc, "load_resolutions", boom)
    emb = ScriptedEmbedder({}, default=[1.0])
    with pytest.raises(TimeoutError):
        await bc.check_beliefs(_Kb(emb), "p", PUSH, None)


async def test_check_beliefs_chunk_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 0.05)
    cands = [_r("kb-1")]
    _patch_load(monkeypatch, cands)
    emb = Ctl({}, [1.0], beliefs={cands[0].wrong_belief}, chunk_delay=0.5)
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert (out.reason, out.detail) == ("timeout", "chunks")


async def test_check_beliefs_warming_then_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 0.1)
    cands = [_r("kb-1"), _r("kb-2")]
    _patch_load(monkeypatch, cands)
    emb = Ctl({}, [1.0, 0.0], beliefs={r.wrong_belief for r in cands}, warm_delay=0.5)
    kb = _Kb(emb)
    out = await bc.check_beliefs(kb, "p", PUSH, None)
    assert (out.reason, out.detail, out.warm) == ("warming", "own", "started")
    assert bc._warm_task is not None
    await bc._warm_task
    emb.calls.clear()
    again = await bc.check_beliefs(kb, "p", PUSH, None)
    assert again.reason == "checked"
    assert again.scored == again.candidates == 2
    assert len(emb.calls) == 1


async def test_check_beliefs_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc, "BELIEF_CHECK_WARM_BATCH", 1)
    monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 0.45)
    cands = [_r("kb-1"), _r("kb-2"), _r("kb-3")]
    _patch_load(monkeypatch, cands)
    emb = Ctl({}, [1.0, 0.0], beliefs={r.wrong_belief for r in cands}, warm_delay=0.3)
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert out.reason == "checked"
    assert (out.candidates, out.scored) == (3, 1)


async def test_check_beliefs_partial_cache_after_warm_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cands = [_r("kb-1"), _r("kb-2")]
    _patch_load(monkeypatch, cands)
    emb = Ctl({}, [1.0, 0.0], beliefs={r.wrong_belief for r in cands})
    kb = _Kb(emb)
    assert (await bc.check_beliefs(kb, "p", PUSH, None)).scored == 2
    third = _r("kb-3")
    _patch_load(monkeypatch, [*cands, third])
    emb.beliefs.add(third.wrong_belief)
    emb.warm_mode = "raise"
    caplog.clear()
    out = await bc.check_beliefs(kb, "p", PUSH, None)
    assert out.reason == "checked"
    assert (out.scored, out.candidates) == (2, 3)
    warns = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.getMessage().startswith("belief_check_warm outcome=error")
    ]
    assert len(warns) == 1


@pytest.mark.parametrize(
    ("chunk_mode", "warm_mode", "detail"),
    [
        ("none", "ok", "chunks-none"),
        ("raise", "ok", "chunks-raised"),
        ("timeout", "ok", "chunks-raised"),
        ("short", "ok", "chunks-length"),
        ("zero", "ok", "chunks-unusable"),
        ("ok", "short", "no-cached-candidates"),
    ],
)
async def test_check_beliefs_embed_failed(
    monkeypatch: pytest.MonkeyPatch, chunk_mode: str, warm_mode: str, detail: str
) -> None:
    cands = [_r("kb-1"), _r("kb-2")]
    _patch_load(monkeypatch, cands)
    emb = Ctl(
        {},
        [1.0, 0.0],
        beliefs={r.wrong_belief for r in cands},
        chunk_mode=chunk_mode,
        warm_mode=warm_mode,
    )
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert (out.reason, out.detail) == ("embed-failed", detail)


async def test_check_beliefs_skips_odd_chunk_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cands = [_r("kb-1", "the one belief here")]
    _patch_load(monkeypatch, cands)
    first, second, third = "First chunk sentence.", "Second chunk sentence.", PUSH
    emb = Ctl(
        {
            first: [0.0, 0.0],
            second: [0.0, 1.0],
            third: [1.0, 0.0, 0.0],
            "the one belief here": [0.0, 1.0],
        },
        beliefs={"the one belief here"},
    )
    out = await bc.check_beliefs(_Kb(emb), "p", f"{first} {second} {third}", None)
    assert out.reason == "checked"
    assert out.dim == 2
    assert out.top_chunk_idx == 1
    assert out.matches[0].score == pytest.approx(1.0)


# --- AC11 tripwires ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("violator", "tripwire"),
    [
        (
            _r("kb-bad", capture="autonomous", observed_once=True),
            "tripwire=observed_once_in_candidates",
        ),
        (
            _r("kb-bad", capture="autonomous", grounding="asserted"),
            "tripwire=asserted_in_candidates",
        ),
    ],
)
async def test_candidate_tripwires(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    violator: Resolution,
    tripwire: str,
) -> None:
    good = _r("kb-good")
    _patch_load(monkeypatch, [])
    monkeypatch.setattr(bc, "select_candidates", lambda rs: ([violator, good], 0))
    emb = ScriptedEmbedder({}, default=[1.0, 0.0])
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert out.reason == "checked"
    assert [m.entry_id for m in out.matches] == ["kb-good"]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(tripwire in w and "kb-bad" in w for w in warns)


async def test_dim_mismatch_tripwire(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r = _r("kb-1")
    _patch_load(monkeypatch, [r])
    assert bc._store_vector(r, [1.0, 0.0, 0.0])
    emb = ScriptedEmbedder({}, default=[1.0, 0.0])
    out = await bc.check_beliefs(_Kb(emb), "p", PUSH, None)
    assert (out.reason, out.detail) == ("embed-failed", "no-cached-candidates")
    assert "kb-1" not in bc._VECTOR_CACHE
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "tripwire=dim_mismatch" in w and "cached_dim=3 chunk_dim=2" in w for w in warns
    )


# --- AC9/AC10 route ---------------------------------------------------------

_MARK = "Zq7MRK"
_TOKENS = (
    "project",
    "session_id",
    "turn_index",
    "harness",
    "engine",
    "reason",
    "detail",
    "warm",
    "candidates",
    "candidates_truncated",
    "uncached",
    "scored",
    "chunks",
    "pending_calls",
    "top_id",
    "top_score",
    "above_threshold",
    "match_ids",
    "match_scores",
    "top_chunk_idx",
    "top_chunk_src",
    "threshold",
    "matcher_version",
    "embed_model",
    "dim",
    "load_ms",
    "embed_ms",
    "latency_ms",
)


@pytest.fixture
def route(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    kb: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, FakeKnowledgeBase, Any]]:
    monkeypatch.setattr(fake_kb, "db", kb.db)
    app.dependency_overrides[get_current_user] = fake_user
    yield client, fake_kb, kb


def _use(monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase, emb: Any) -> None:
    monkeypatch.setattr(fake_kb, "embedder", emb)
    monkeypatch.setattr(fake_kb, "embed_batch", emb.embed_batch, raising=False)


def _body(text: str, project: str = "p", **kw: Any) -> dict[str, Any]:
    return {
        "text": text,
        "reasoning": _MARK,
        "project": project,
        "pending_calls": [{"tool": "bash", "input_summary": f"{_MARK} git push"}],
        "session_id": "sess-1",
        "turn_index": 3,
        **kw,
    }


_REASONS: list[BeliefCheckReason] = list(BeliefCheckReason.__args__)


@pytest.mark.parametrize("reason", _REASONS)
async def test_route_every_reason(
    route: tuple[TestClient, FakeKnowledgeBase, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reason: str,
) -> None:
    client, fake_kb, kb = route
    caplog.set_level(logging.INFO)
    text = f"{_MARK} {PUSH}"
    project = "p"
    emb: Any = ScriptedEmbedder({}, default=[1.0, 0.0])
    if reason != "no-candidates":
        await _store(kb, "p", **_res(provenance={"capture": "deliberate"}))
    if reason == "no-project":
        project = "   "
    elif reason == "project-disabled":
        monkeypatch.setenv("KB_SOFT_GATE_DISABLED_PROJECTS", "other, P")
    elif reason == "no-embedder":
        emb = None
    elif reason == "no-chunks":
        text = f"{_MARK} ok."
    elif reason == "timeout":
        monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 0.05)
        emb = Ctl({}, [1.0, 0.0], chunk_delay=0.3)
    elif reason == "warming":
        monkeypatch.setattr(bc, "BELIEF_CHECK_DEADLINE_S", 0.1)
        emb = Ctl({}, [1.0, 0.0], beliefs={PUSH}, warm_delay=0.3)
    elif reason == "embed-failed":
        emb = Ctl({}, [1.0, 0.0], chunk_mode="none")
    elif reason == "error":

        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"{_MARK} secret")

        monkeypatch.setattr(bc, "load_resolutions", boom)
    if emb is not None:
        _use(monkeypatch, fake_kb, emb)
    resp = client.post(
        "/api/kb/belief-check",
        json=_body(text, project),
        headers={"X-KB-Harness": "talos\nx", "X-KB-Engine": "opus"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] == reason
    assert body["threshold"] == bc.BELIEF_CHECK_THRESHOLD
    assert body["matcher_version"] == bc.BELIEF_CHECK_MATCHER_VERSION
    if reason == "checked":
        assert body["matches"][0]["score"] == pytest.approx(1.0)
        assert body["embed_model"] == "qwen3-embedding:0.6b"
    else:
        assert body["matches"] == []
    if reason == "no-embedder":
        assert body["embed_model"] == ""

    for rec in caplog.records:
        assert _MARK not in rec.getMessage()
        assert PUSH not in rec.getMessage()
    lines = [
        r
        for r in caplog.records
        if r.name == _ROUTE_LOGGER
        and r.levelno == logging.INFO
        and r.getMessage().startswith("belief_check ")
    ]
    assert len(lines) == 1
    msg = lines[0].getMessage()
    positions = [
        msg.index(f" {t}=") if i else msg.index(f"{t}=") for i, t in enumerate(_TOKENS)
    ]
    assert positions == sorted(positions)
    assert f"reason={reason} " in msg
    assert "harness=talosx " in msg
    assert "engine=opus " in msg
    assert "turn_index=3 " in msg
    assert "pending_calls=1 " in msg
    if reason in ("no-project", "project-disabled", "error"):
        assert "candidates=0 " in msg
        assert "load_ms=0 embed_ms=0 " in msg
    if reason == "checked":
        assert "above_threshold=true " in msg
        assert "top_score=1.0000 " in msg
        assert "top_chunk_src=text " in msg
        assert "dim=2 " in msg
    else:
        assert "top_id= " in msg
        assert "above_threshold=false " in msg
        assert "match_ids= match_scores= top_chunk_idx= top_chunk_src= " in msg
    if reason == "error":
        assert re.search(r"detail=test_belief_check\.py:\d+ ", msg)
        failed = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert "belief_check failed project=p exc=RuntimeError" in failed


async def test_route_sanitizes_session_id(
    route: tuple[TestClient, FakeKnowledgeBase, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _, _ = route
    caplog.set_level(logging.INFO)
    resp = client.post(
        "/api/kb/belief-check",
        json={"text": PUSH, "project": "p", "session_id": "abc\nbelief_check fake"},
    )
    assert resp.status_code == 200
    [rec] = [
        r
        for r in caplog.records
        if r.name == _ROUTE_LOGGER and r.levelno == logging.INFO
    ]
    assert "\n" not in rec.getMessage()
    assert "session_id=abcbelief_checkfake " in rec.getMessage()
    assert "turn_index= " in rec.getMessage()


def test_route_requires_auth(client: TestClient) -> None:
    resp = client.post("/api/kb/belief-check", json={"text": PUSH, "project": "p"})
    assert resp.status_code == 401


@pytest.mark.parametrize("missing", ["text", "project"])
def test_route_rejects_missing_fields(client: TestClient, missing: str) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    body = {"text": PUSH, "project": "p"}
    del body[missing]
    assert client.post("/api/kb/belief-check", json=body).status_code == 422


def test_where_without_traceback() -> None:
    assert belief_check_routes._where(RuntimeError()) == ""


def test_embed_model_on_a_broken_kb() -> None:
    class _State:
        @property
        def kb(self) -> Any:
            raise RuntimeError

    class _App:
        state = _State()

    class _Req:
        app = _App()

    assert belief_check_routes._embed_model(_Req()) == ""  # type: ignore[arg-type]
