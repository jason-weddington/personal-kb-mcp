"""Tests for surprise_worker.distill_candidates and get_distiller_llm.

Every scenario runs on a real SQLite KB plus a real SQLite service DB with a
scripted FakeLLM; the end-to-end test drives the real app through
POST /api/kb/turn, POST /api/kb/surprise/drain and GET /api/kb/prevention.
No network.
"""

import json
import logging
import sqlite3
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core import create_sqlite
from kb_core.config import IngestConfig
from kb_core.near_duplicates import NearDuplicateCheck

import kb_service.database as database
from kb_service import surprise_worker
from kb_service.db_sqlite import SqlitePool
from kb_service.main import app
from kb_service.prevention import build_gate_index, build_slice, load_resolutions
from kb_service.surprise import (
    SURPRISE_DETECTOR_SYSTEM,
    DistillResult,
    SurpriseCandidate,
)
from kb_service.surprise_distill import (
    SURPRISE_CRITIC_VERSION,
    SURPRISE_DISTILLER_SYSTEM,
    SURPRISE_DISTILLER_VERSION,
    SURPRISE_HINT_KEY,
    build_knowledge_details,
    build_resolution,
    lesson_ttl_days,
    parse_distill_response,
)
from kb_service.surprise_worker import candidate_from_row, distill_candidates
from tests.conftest import CRITIC_ACCEPT, FakeCritic, FakeLLM

LOGGER = "kb_service.surprise_worker"

# _s, _t and _w are the spec's S (shape-1), T (shape-2) and W (stored
# autonomous resolution) fixtures.

_HERMETIC_ENV = (
    "KB_SOFT_GATE_ENABLED",
    "KB_SOFT_GATE_SHADOW",
    "KB_SOFT_GATE_DISABLED_PROJECTS",
    "KB_DELIVER_OBSERVED_ONCE",
    "KB_NEAR_DUPLICATE_FLOOR",
    "KB_SKIP_SAFETY",
    "KB_SURPRISE_CAPTURE",
    "KB_SURPRISE_LESSON_TTL_DAYS",
    "KB_SURPRISE_DETECTOR_MODEL",
    "KB_SURPRISE_DISTILL_MODEL",
    "KB_SURPRISE_CRITIC_MODEL",
    "KB_SURPRISE_MIN_CONFIDENCE",
    "KB_SURPRISE_MIN_CONFIDENCE_SHAPE2",
    "KB_SURPRISE_MIN_CONFIDENCE_SHAPE3",
    "KB_QUERY_PROVIDER",
    "ANTHROPIC_API_KEY",
)

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)

_D_OBJ: dict[str, Any] = {
    "durable": True,
    "short_title": "Push to HEAD:main",
    "long_title": "Push the current branch with git push origin HEAD:main",
    "corrected_fact": "Push with git push origin HEAD:main",
    "lesson": "git push origin main is rejected as non-fast-forward here.",
    "lesson_class": "none",
}
D = json.dumps(_D_OBJ)
_D_FIELDS = {k: v for k, v in _D_OBJ.items() if k != "durable"}

_LESSONS_SQL = (
    "SELECT COUNT(*) FROM knowledge_entries WHERE entry_type = 'lesson_learned'"
    " AND project_ref = 'p' AND is_active = 1"
)

_INSERT_SQL = (
    "INSERT INTO surprise_candidates (shape, session_id, project, turn_event_ids,"
    " detector_model, detector_output, status, entry_id, created_at)"
    " VALUES ($1, $2, $3, $4, $5, $6, 'pending', NULL, $7) RETURNING id, shape,"
    " session_id, project, turn_event_ids, detector_model, detector_output,"
    " status, entry_id, created_at"
)

_CREATED_AT = "2026-10-09T00:00:00+00:00"

_DROP = object()


# --- fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _HERMETIC_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    yield tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


class _ConstEmbedder:
    async def embed(self, text: str) -> list[float] | None:
        return [1.0] + [0.0] * 1023


async def _make_kb(path: Path, **kw: Any) -> Any:
    return await create_sqlite(
        path, extraction_llm=None, query_llm=None, synthesis_llm=None, **kw
    )


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Any]:
    k = await _make_kb(tmp_path / "kb.db", embedder=None)
    try:
        yield k
    finally:
        await k.close()


@pytest.fixture
async def kb_cos(tmp_path: Path) -> AsyncIterator[Any]:
    k = await _make_kb(tmp_path / "kb_cos.db", embedder=_ConstEmbedder())
    try:
        yield k
    finally:
        await k.close()


def _raise_detector() -> Any:
    raise RuntimeError("the distiller must never use the detector getter")


@pytest.fixture(autouse=True)
def critic(monkeypatch: pytest.MonkeyPatch) -> FakeCritic:
    fake = FakeCritic()
    monkeypatch.setattr(surprise_worker, "get_critic_llm", lambda: fake)
    return fake


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: fake)
    monkeypatch.setattr(surprise_worker, "get_detector_llm", _raise_detector)
    return fake


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger=LOGGER)
    return caplog


# --- helpers ------------------------------------------------------------------


async def _insert(
    pool: SqlitePool,
    shape: int,
    session: str,
    project: str,
    turn_event_ids: list[str],
    detector_model: str,
    output: dict[str, Any],
) -> SurpriseCandidate:
    row = await pool.fetchrow(
        _INSERT_SQL,
        shape,
        session,
        project,
        json.dumps(turn_event_ids),
        detector_model,
        json.dumps(output),
        _CREATED_AT,
    )
    assert row is not None
    return candidate_from_row(row)


def _s1_output(**kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "wrong_belief": "git push origin main",
        "corrected_fact": "git push origin HEAD:main",
        "evidence_excerpt": "rejected",
        "confidence": 1.0,
    }
    out.update(kw)
    return out


async def _s(
    pool: SqlitePool, sess: str, project: str = "p", **kw: Any
) -> SurpriseCandidate:
    return await _insert(
        pool, 1, sess, project, [f"{sess}:0"], "rule:shape1", _s1_output(**kw)
    )


async def _t(pool: SqlitePool, sess: str, wb: str) -> SurpriseCandidate:
    return await _insert(
        pool,
        2,
        sess,
        "p",
        [f"{sess}:0", f"{sess}:1"],
        "FakeLLM",
        {
            "wrong_belief": wb,
            "corrected_fact": "port 8080 is taken by caddy",
            "evidence_excerpt": "8080 is taken by caddy",
            "confidence": 0.9,
        },
    )


_W_COUNTER = {"n": 0}


async def _w(kb: Any, hint: Any = None, project: str = "p", **kw: Any) -> str:
    _W_COUNTER["n"] += 1
    n = _W_COUNTER["n"]
    res: dict[str, Any] = {
        "corrected_fact": "f",
        "wrong_belief": "git push origin main",
        "cue": {
            "tool": "Bash",
            "target_class": "git push",
            "args_prefix": "origin main",
        },
        "provenance": {
            "capture": "autonomous",
            "grounding": "observed",
            "event_id": "old:0",
        },
    }
    for key, value in kw.items():
        if value is _DROP:
            res.pop(key, None)
        else:
            res[key] = value
    hints: dict[str, object] = {"resolution": res}
    if hint is not None:
        hints[SURPRISE_HINT_KEY] = hint
    entry = await kb.store(
        short_title=f"Existing resolution {n}",
        long_title=f"Existing resolution number {n}",
        knowledge_details=f"existing details {n}",
        project_ref=project,
        hints=hints,
        enrich=False,
    )
    return str(entry.id)


async def _lessons(kb: Any) -> int:
    cursor = await kb.db.execute(_LESSONS_SQL)
    row = await cursor.fetchone()
    return int(row[0])


async def _rows(pool: SqlitePool) -> list[dict[str, Any]]:
    return await pool.fetch("SELECT * FROM surprise_distillations ORDER BY id")


async def _cand_row(pool: SqlitePool, cid: int) -> dict[str, Any]:
    row = await pool.fetchrow(
        "SELECT status, entry_id FROM surprise_candidates WHERE id = $1", cid
    )
    assert row is not None
    return row


def _res_of(entry: Any) -> dict[str, Any]:
    res = entry.hints["resolution"]
    assert isinstance(res, dict)
    return res


def _summary(caplog: pytest.LogCaptureFixture) -> str:
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.getMessage().startswith("surprise_distill summary")
    ]
    assert len(lines) == 1, lines
    return lines[0]


def _warnings(caplog: pytest.LogCaptureFixture, needle: str = "") -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER
        and r.levelno == logging.WARNING
        and needle in r.getMessage()
    ]


# --- get_distiller_llm (R7) ---------------------------------------------------


@pytest.fixture
def fresh_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surprise_worker, "_DISTILLER_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_DETECTOR_LLM_CACHE", {})


def test_get_distiller_llm_default(fresh_caches: None) -> None:
    llm = surprise_worker.get_distiller_llm()
    assert type(llm).__name__ == "AnthropicLLMClient"
    assert llm._config.model == "claude-sonnet-5-5"  # type: ignore[union-attr]
    assert llm is not surprise_worker.get_detector_llm()


def test_get_distiller_llm_override_is_cached(
    fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_DISTILL_MODEL", "test-model-y")
    llm = surprise_worker.get_distiller_llm()
    assert llm._config.model == "test-model-y"  # type: ignore[union-attr]
    assert surprise_worker.get_distiller_llm() is llm
    assert surprise_worker.detector_model_name(llm) == "test-model-y"


def test_get_distiller_llm_ignores_detector_model(
    fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_DETECTOR_MODEL", "test-model-x")
    llm = surprise_worker.get_distiller_llm()
    assert llm._config.model == "claude-sonnet-5-5"  # type: ignore[union-attr]


def test_get_distiller_llm_ignores_query_provider(
    fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    llm = surprise_worker.get_distiller_llm()
    assert type(llm).__name__ == "AnthropicLLMClient"
    assert llm._config.model == "claude-sonnet-5-5"  # type: ignore[union-attr]


def test_get_distiller_llm_import_error(
    fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "kb_core.llm.anthropic", None)
    assert surprise_worker.get_distiller_llm() is None
    assert surprise_worker._DISTILLER_LLM_CACHE == {}


# --- (a)-(d): write, merge, same session, multi-merge -------------------------


async def test_write_merge_same_session_sequence(
    pool: SqlitePool, kb: Any, llm: FakeLLM, logs: pytest.LogCaptureFixture
) -> None:
    # (a) written
    c1 = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c1], "on")
    assert len(result.entries_written) == 1
    assert result.entries_merged == []
    x = result.entries_written[0]
    entry = await kb.get(x)
    assert entry.entry_type.value == "lesson_learned"
    assert entry.project_ref == "p"
    assert entry.contributor == "surprise-capture"
    assert entry.confidence_level == 0.7
    assert sorted(entry.tags) == [
        "lesson-class:none",
        "shape-1",
        "surprise-capture",
    ]
    assert entry.hints["surprise_capture"]["lesson_class"] == "none"
    assert (
        entry.source_context == f"surprise_capture candidate {c1.id} shape 1 session s1"
    )
    verdict, _ = parse_distill_response(D)
    assert verdict is not None
    assert entry.knowledge_details == build_knowledge_details(
        c1, verdict, build_resolution(c1, verdict)
    )
    assert entry.hints == {
        "resolution": {
            "corrected_fact": "Push with git push origin HEAD:main",
            "wrong_belief": "git push origin main",
            "evidence": "rejected",
            "cue": {
                "tool": "Bash",
                "target_class": "git push",
                "args_prefix": "origin main",
            },
            "provenance": {
                "capture": "autonomous",
                "grounding": "observed",
                "event_id": "s1:0",
            },
            "observed_sessions": 1,
            "scope": "project",
        },
        "surprise_capture": {
            "shape": 1,
            "lesson_class": "none",
            "sessions": ["s1"],
            "candidate_ids": [c1.id],
            "event_ids": ["s1:0"],
        },
    }
    assert await _cand_row(pool, c1.id) == {"status": "written", "entry_id": x}
    rows = await _rows(pool)
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "written"
    assert row["match_kind"] == ""
    assert row["near_duplicate_status"] == "embedder_unavailable"
    assert row["near_duplicate_floor"] == 0.88
    assert row["cue_target_class"] == "git push"
    assert row["observed_sessions_before"] is None
    assert row["observed_sessions_after"] == 1
    assert json.loads(row["verdict"]) == {
        **_D_FIELDS,
        "critic_version": SURPRISE_CRITIC_VERSION,
        "critic": json.loads(CRITIC_ACCEPT),
    }
    assert row["distiller_model"] == "FakeLLM"
    assert row["distiller_version"] == SURPRISE_DISTILLER_VERSION
    assert row["raw_response_excerpt"] == D
    assert row["prompt_chars"] > 0
    assert llm.generate_calls[0][1] == SURPRISE_DISTILLER_SYSTEM
    summary = _summary(logs)
    assert summary.startswith(
        f"surprise_distill summary distiller_version={SURPRISE_DISTILLER_VERSION}"
        " distiller_model=FakeLLM"
        " input=1 llm_calls=1 written=1 merged=0"
    )
    assert "exact_matches=0 cosine_matches=0 near_dup_unavailable=1" in summary
    assert "aborted=0" in summary
    assert "surprise_distill decision" in logs.text
    assert "outcome=written" in logs.text

    # (b) merged via exact match, no LLM call
    logs.clear()
    c2 = await _s(pool, "s2")
    result = await distill_candidates(pool, kb, [c2], "on")
    assert result == DistillResult([], [x])
    assert len(llm.generate_calls) == 1
    entry = await kb.get(x)
    assert entry.version == 2
    assert _res_of(entry)["observed_sessions"] == 2
    assert _res_of(entry)["evidence"] == "rejected"
    assert entry.hints["surprise_capture"] == {
        "shape": 1,
        "lesson_class": "none",
        "sessions": ["s1", "s2"],
        "candidate_ids": [c1.id, c2.id],
        "event_ids": ["s1:0", "s2:0"],
    }
    assert await _lessons(kb) == 1
    row = (await _rows(pool))[-1]
    assert row["outcome"] == "merged"
    assert row["match_kind"] == "exact"
    assert row["observed_sessions_before"] == 1
    assert row["observed_sessions_after"] == 2
    assert row["distiller_model"] == ""
    assert row["prompt_chars"] is None
    rs, _ = await load_resolutions(kb.db, "p", True)
    assert [c.resolution_id for c in build_gate_index(rs)[0]] == [x]
    summary = _summary(logs)
    assert "llm_calls=0" in summary
    assert "exact_matches=1" in summary
    assert "promoted=1" in summary

    # (c) same session: no write
    logs.clear()
    c3 = await _s(pool, "s1")
    result = await distill_candidates(pool, kb, [c3], "on")
    assert result == DistillResult([], [])
    entry = await kb.get(x)
    assert entry.version == 2
    assert _res_of(entry)["observed_sessions"] == 2
    assert entry.hints["surprise_capture"]["event_ids"] == ["s1:0", "s2:0"]
    assert await _cand_row(pool, c3.id) == {"status": "merged", "entry_id": x}
    row = (await _rows(pool))[-1]
    assert row["outcome"] == "same_session"
    assert row["observed_sessions_before"] == 2
    assert row["observed_sessions_after"] == 2
    assert "promoted=0" in _summary(logs)

    # (d) two new sessions in one call
    c4, c5 = await _s(pool, "s3"), await _s(pool, "s4")
    result = await distill_candidates(pool, kb, [c4, c5], "on")
    assert result.entries_merged == [x]
    entry = await kb.get(x)
    assert _res_of(entry)["observed_sessions"] == 4
    assert entry.hints["surprise_capture"]["event_ids"] == [
        "s1:0",
        "s2:0",
        "s3:0",
        "s4:0",
    ]


async def test_event_ids_cap(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    w2 = await _w(
        kb,
        hint={
            "sessions": ["old"],
            "candidate_ids": [1],
            "event_ids": [f"e{i}" for i in range(20)],
        },
    )
    c = await _s(pool, "s5")
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [w2])
    assert llm.generate_calls == []
    entry = await kb.get(w2)
    assert _res_of(entry)["observed_sessions"] == 2
    assert entry.hints["surprise_capture"] == {
        "sessions": ["old", "s5"],
        "candidate_ids": [1, c.id],
        "event_ids": [f"e{i}" for i in range(1, 20)] + ["s5:0"],
    }


# --- (e) covered --------------------------------------------------------------


@pytest.mark.parametrize(
    "provenance",
    [{"capture": "deliberate", "grounding": "asserted"}, _DROP],
)
async def test_covered_deliberate(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    logs: pytest.LogCaptureFixture,
    provenance: Any,
) -> None:
    z = await _w(kb, provenance=provenance)
    before = await kb.get(z)
    c = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [])
    after = await kb.get(z)
    assert after.version == before.version
    assert after.hints == before.hints
    assert await _lessons(kb) == 0
    assert await _cand_row(pool, c.id) == {"status": "rejected", "entry_id": None}
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("covered", "deliberate")
    assert row["matched_entry_id"] == z
    assert row["match_kind"] == "exact"
    assert llm.generate_calls == []
    assert "outcome=covered reason=deliberate" in logs.text


async def test_covered_no_resolution_cosine(
    pool: SqlitePool, kb_cos: Any, llm: FakeLLM
) -> None:
    plain = await kb_cos.store(
        short_title="Port facts",
        long_title="Which ports are used on this host",
        knowledge_details="8080 is used by caddy",
        project_ref="p",
        enrich=False,
    )
    c = await _t(pool, "s7", "port 8080 is free")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb_cos, [c], "on")
    assert result == DistillResult([], [])
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("covered", "no_resolution")
    assert row["match_kind"] == "cosine"
    assert row["similarity"] == pytest.approx(1.0)
    assert row["near_duplicate_status"] == "checked"
    assert row["matched_entry_id"] == plain.id
    assert (await kb_cos.get(plain.id)).version == 1
    assert await _lessons(kb_cos) == 0


async def test_covered_global(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    g = await _w(kb, project="q", scope="global")
    c = await _s(pool, "s1")
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [])
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("covered", "global")
    assert row["matched_entry_id"] == g
    assert llm.generate_calls == []
    assert (await kb.get(g)).version == 1


# --- (f) cosine merge, cue mismatch, same session via provenance ---------------


async def test_cosine_merge(pool: SqlitePool, kb_cos: Any, llm: FakeLLM) -> None:
    y = await _w(kb_cos, wrong_belief="port 8080 is free", cue=_DROP)
    c = await _t(pool, "s9", "the service listens on 8080")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb_cos, [c], "on")
    assert result == DistillResult([], [y])
    entry = await kb_cos.get(y)
    assert _res_of(entry)["observed_sessions"] == 2
    assert entry.hints["surprise_capture"] == {
        "sessions": ["s9"],
        "candidate_ids": [c.id],
        "event_ids": ["s9:1"],
    }
    assert (await _rows(pool))[0]["match_kind"] == "cosine"


async def test_cosine_cue_mismatch(pool: SqlitePool, kb_cos: Any, llm: FakeLLM) -> None:
    y2 = await _w(
        kb_cos,
        wrong_belief="git pull origin main",
        cue={"tool": "Bash", "target_class": "git pull"},
    )
    c = await _s(pool, "s9")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb_cos, [c], "on")
    assert result == DistillResult([], [])
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("covered", "cue_mismatch")
    assert row["match_kind"] == "cosine"
    assert (await kb_cos.get(y2)).version == 1
    assert await _lessons(kb_cos) == 0


async def test_cosine_same_session_via_provenance(
    pool: SqlitePool, kb_cos: Any, llm: FakeLLM
) -> None:
    y3 = await _w(
        kb_cos,
        wrong_belief="port 8080 is free",
        cue=_DROP,
        provenance={
            "capture": "autonomous",
            "grounding": "observed",
            "event_id": "s9:0",
        },
    )
    c = await _t(pool, "s9", "the service listens on 8080")
    llm.enqueue(D)
    await distill_candidates(pool, kb_cos, [c], "on")
    assert (await _rows(pool))[0]["outcome"] == "same_session"
    assert (await kb_cos.get(y3)).version == 1
    assert await _cand_row(pool, c.id) == {"status": "merged", "entry_id": y3}


# --- (g) mode gate (R7) -------------------------------------------------------


@pytest.mark.parametrize("mode", ["shadow", "off"])
async def test_mode_gate(
    pool: SqlitePool,
    kb: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    fake = FakeLLM()
    calls: list[int] = []

    def spy() -> FakeLLM:
        calls.append(1)
        return fake

    monkeypatch.setattr(surprise_worker, "get_distiller_llm", spy)
    monkeypatch.setattr(surprise_worker, "get_detector_llm", _raise_detector)
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    w1 = await _w(kb)
    cands = [await _s(pool, "s5"), await _t(pool, "s6", "port 8080 is free")]
    fake.enqueue(D)
    lessons = await _lessons(kb)
    caplog.clear()
    assert await distill_candidates(pool, kb, cands, mode) == DistillResult()  # type: ignore[arg-type]
    assert calls == []
    assert fake.generate_calls == []
    for c in cands:
        assert await _cand_row(pool, c.id) == {"status": "pending", "entry_id": None}
    assert (await kb.get(w1)).version == 1
    assert await _lessons(kb) == lessons
    assert await _rows(pool) == []
    assert [r for r in caplog.records if r.name == LOGGER] == []


# --- (h) parser rejects -------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "outcome", "reason", "warns"),
    [
        (
            '{"durable": false, "why": "transient network error"}',
            "not_durable",
            "transient network error",
            0,
        ),
        ("not json", "unparseable", "", 1),
        (None, "llm_error", "none", 1),
        (json.dumps({**_D_OBJ, "short_title": ""}), "invalid_fields", "", 1),
        (
            json.dumps(
                {**_D_OBJ, "corrected_fact": "use [REDACTED:Secret Assignment] here"}
            ),
            "redacted",
            "verdict",
            0,
        ),
    ],
)
async def test_rejects(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    logs: pytest.LogCaptureFixture,
    response: str | None,
    outcome: str,
    reason: str,
    warns: int,
) -> None:
    if response is not None:
        llm.enqueue(response)
    c = await _s(pool, "s1")
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [])
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == (outcome, reason)
    assert await _cand_row(pool, c.id) == {"status": "rejected", "entry_id": None}
    assert await _lessons(kb) == 0
    assert len(_warnings(logs)) == warns
    if warns:
        assert len(_warnings(logs, "surprise_distill distill_failed")) == 1
    assert f"outcome={outcome}" in logs.text


async def test_llm_exception(
    pool: SqlitePool, kb: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom(FakeLLM):
        async def generate(self, prompt: Any, *, system: Any = None) -> str | None:
            raise RuntimeError("boom")

    monkeypatch.setattr(surprise_worker, "get_distiller_llm", Boom)
    monkeypatch.setattr(surprise_worker, "get_detector_llm", _raise_detector)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("llm_error", "exception")
    assert row["distiller_model"] == "Boom"
    assert row["raw_response_excerpt"] is None


# --- (i) no LLM ---------------------------------------------------------------


async def test_no_llm(
    pool: SqlitePool,
    kb: Any,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: None)
    monkeypatch.setattr(surprise_worker, "get_detector_llm", _raise_detector)
    c = await _s(pool, "s1")
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult()
    assert await _cand_row(pool, c.id) == {"status": "pending", "entry_id": None}
    assert await _rows(pool) == []
    assert _warnings(logs) == []
    summary = _summary(logs)
    assert "no_llm=1" in summary
    assert "oldest_input_created_at=2026-10-09T00:00:00+00:00" in summary
    assert "distiller_model= " in summary

    w1 = await _w(kb)
    c6 = await _s(pool, "s6")
    assert await distill_candidates(pool, kb, [c6], "on") == DistillResult([], [w1])
    entry = await kb.get(w1)
    assert _res_of(entry)["observed_sessions"] == 2
    assert entry.hints["surprise_capture"]["event_ids"] == ["s6:0"]


# --- (j) pre-LLM rejects ------------------------------------------------------


async def test_no_project(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    c = await _s(pool, "s1", project="")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert row["outcome"] == "no_project"
    assert row["distiller_model"] == ""
    assert row["latency_ms"] is None
    assert llm.generate_calls == []
    assert (await _cand_row(pool, c.id))["status"] == "rejected"


async def test_redacted_wrong_belief(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    wb = "[REDACTED:Secret Assignment]"
    cands = [
        await _s(pool, "s1", wrong_belief=wb),
        await _s(pool, "s2", wrong_belief=wb),
    ]
    await distill_candidates(pool, kb, cands, "on")
    rows = await _rows(pool)
    assert [(r["outcome"], r["reason"]) for r in rows] == [
        ("redacted", "wrong_belief"),
        ("redacted", "wrong_belief"),
    ]
    assert await _lessons(kb) == 0
    assert llm.generate_calls == []


async def test_gate_induced(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    c = await _s(
        pool,
        "s1",
        evidence_excerpt="KB soft gate (deny once): git push origin main is rejected",
    )
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [])
    assert (await _rows(pool))[0]["outcome"] == "gate_induced"
    assert llm.generate_calls == []
    assert (await _cand_row(pool, c.id))["status"] == "rejected"


# --- (k) secrets --------------------------------------------------------------

D_SECRET = json.dumps(
    {
        **_D_OBJ,
        "lesson": "Fetch it with curl https://user:s3cretpass@example.com/x first.",
        "lesson_class": "none",
    }
)


async def test_secret_detected(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    llm.enqueue(D_SECRET)
    c = await _s(pool, "s1")
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [])
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == (
        "secret_detected",
        "Basic Auth Credentials",
    )
    assert row["verdict"] is None
    assert row["raw_response_excerpt"] == "[REDACTED:Basic Auth Credentials]"
    assert all("s3cretpass" not in str(v) for v in row.values())
    assert await _lessons(kb) == 0


async def test_secret_scan_skipped(
    pool: SqlitePool, tmp_path: Path, llm: FakeLLM
) -> None:
    kb = await _make_kb(
        tmp_path / "skip.db", embedder=None, ingest=IngestConfig(skip_safety=True)
    )
    try:
        llm.enqueue(D_SECRET)
        c = await _s(pool, "s1")
        result = await distill_candidates(pool, kb, [c], "on")
        assert len(result.entries_written) == 1
        assert (await _rows(pool))[0]["outcome"] == "written"
    finally:
        await kb.close()


# --- (l) invalid resolution ---------------------------------------------------


async def test_invalid_resolution_on_merge(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    bad = await _w(kb, provenance={"capture": "autonomous", "grounding": "observed"})
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == (
        "invalid_resolution",
        "observed_needs_event_id",
    )
    assert row["matched_entry_id"] == bad
    assert row["match_kind"] == "exact"
    assert row["observed_sessions_before"] is None
    assert (await kb.get(bad)).version == 1


async def test_invalid_resolution_none(
    pool: SqlitePool, kb: Any, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        surprise_worker, "validate_and_stamp_resolution", lambda *a, **k: None
    )
    llm.enqueue(D)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("invalid_resolution", "no_resolution")


# --- (m) kb errors ------------------------------------------------------------


async def test_kb_store_error(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    async def boom(**kw: Any) -> Any:
        raise RuntimeError("store failed")

    monkeypatch.setattr(kb, "store", boom)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult()
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("kb_error", "RuntimeError")
    assert (await _cand_row(pool, c.id))["status"] == "rejected"
    assert len(_warnings(logs, "distill_failed")) == 1


async def test_kb_error_does_not_stop_the_batch(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    def boom(content: str) -> list[str] | None:
        raise RuntimeError("scan failed")

    monkeypatch.setattr(surprise_worker, "detect_secrets_in_content", boom)
    llm.enqueue(D)
    cands = [await _s(pool, "s1"), await _s(pool, "s2", project="")]
    await distill_candidates(pool, kb, cands, "on")
    rows = await _rows(pool)
    assert [(r["outcome"], r["reason"]) for r in rows] == [
        ("kb_error", "RuntimeError"),
        ("no_project", ""),
    ]
    assert len(_warnings(logs, "distill_failed")) == 1


# --- (n) tripwires ------------------------------------------------------------


async def test_shadow_candidate_tripwire(
    pool: SqlitePool, kb: Any, llm: FakeLLM, logs: pytest.LogCaptureFixture
) -> None:
    c = await _s(pool, "s1")
    await pool.execute(
        "UPDATE surprise_candidates SET status = 'shadow' WHERE id = $1", c.id
    )
    llm.enqueue(D)
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult()
    assert llm.generate_calls == []
    assert await _lessons(kb) == 0
    assert await _rows(pool) == []
    assert len(_warnings(logs)) == 1
    assert len(_warnings(logs, "tripwire=double_distill")) == 1
    assert (await _cand_row(pool, c.id))["status"] == "shadow"
    assert "double_distill=1" in _summary(logs)


async def test_lost_claim_tripwire(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    c = await _s(pool, "s1")
    real_store = kb.store

    async def racing_store(**kw: Any) -> Any:
        await pool.execute(
            "UPDATE surprise_candidates SET status = 'rejected' WHERE id = $1", c.id
        )
        return await real_store(**kw)

    monkeypatch.setattr(kb, "store", racing_store)
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c], "on")
    assert len(result.entries_written) == 1
    assert len(_warnings(logs, "tripwire=double_distill")) == 1
    assert await _rows(pool) == []
    assert (await _cand_row(pool, c.id))["status"] == "rejected"
    summary = _summary(logs)
    assert "written=0" in summary
    assert "double_distill=1" in summary


# --- (o)-(s) floor, near-dup failure, missing entry, persistence, empty ---------


async def test_bad_floor_falls_back(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_NEAR_DUPLICATE_FLOOR", "abc")
    llm.enqueue(D)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert row["outcome"] == "written"
    assert row["near_duplicate_floor"] == 0.88
    assert len(_warnings(logs, "bad_near_duplicate_floor")) == 1


async def test_near_duplicate_search_failed(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    async def failed(**kw: Any) -> NearDuplicateCheck:
        return NearDuplicateCheck(
            status="search_failed", candidates=(), top_similarity=None
        )

    monkeypatch.setattr(kb, "find_near_duplicates", failed)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert row["outcome"] == "written"
    assert row["near_duplicate_status"] == "search_failed"
    assert "near_dup_unavailable=1" in _summary(logs)


async def test_matched_entry_missing(
    pool: SqlitePool, kb: Any, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _w(kb)

    async def none(entry_id: str) -> None:
        return None

    monkeypatch.setattr(kb, "get", none)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("covered", "missing")


async def test_persist_failure_propagates(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
) -> None:
    def broken() -> Any:
        raise RuntimeError("no connection")

    monkeypatch.setattr(pool, "acquire", broken)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    with pytest.raises(RuntimeError):
        await distill_candidates(pool, kb, [c], "on")
    cursor = await kb.db.execute(
        "SELECT id FROM knowledge_entries WHERE entry_type = 'lesson_learned'"
    )
    (new_id,) = await cursor.fetchone()
    failed = _warnings(logs, "persist_failed")
    assert len(failed) == 1
    assert f"entry_id={new_id}" in failed[0]
    assert "aborted=1" in _summary(logs)


async def test_empty_candidates(
    pool: SqlitePool, kb: Any, llm: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    assert await distill_candidates(pool, kb, [], "on") == DistillResult()
    assert [r for r in caplog.records if r.name == LOGGER] == []


# --- AC-14: end to end through the app ---------------------------------------


@pytest.fixture
def e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FakeLLM]]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "0")
    fake = FakeLLM()
    monkeypatch.setattr(surprise_worker, "get_detector_llm", lambda: fake)
    monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: fake)
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def _seed_push(client: TestClient, session: str) -> None:
    body = {
        "event_id": f"{session}:0",
        "session_id": session,
        "turn_index": 0,
        "harness": "claude-code",
        "mode": "interactive",
        "project": "p",
        "ts": "2026-10-09T00:00:00+00:00",
        "user_prompt": "push it",
        "final_message": "Pushed.",
        "truncated": False,
        "items": [
            {
                "kind": "tool_call",
                "tool_use_id": "t1",
                "tool": "Bash",
                "target": "git push origin main",
                "target_class": "git push",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "t1",
                "is_error": True,
                "excerpt": "rejected: non-fast-forward",
            },
            {
                "kind": "tool_call",
                "tool_use_id": "t2",
                "tool": "Bash",
                "target": "git push origin HEAD:main",
                "target_class": "git push",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "t2",
                "is_error": False,
                "excerpt": "ok",
            },
        ],
    }
    resp = client.post("/api/kb/turn", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["reason"] == "recorded"


def _drain(client: TestClient) -> dict[str, Any]:
    resp = client.post("/api/kb/surprise/drain")
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def _prevention(client: TestClient) -> dict[str, Any]:
    resp = client.get("/api/kb/prevention", params={"project": "p"})
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def _hints(tmp_path: Path, entry_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(tmp_path / "knowledge.db")
    try:
        row = conn.execute(
            "SELECT hints FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
    finally:
        conn.close()
    hints: dict[str, Any] = json.loads(row[0])
    return hints


def test_end_to_end_recurrence_promotion(
    e2e: tuple[TestClient, FakeLLM],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = e2e
    _seed_push(client, "s1")
    fake.enqueue('{"surprise": false}')
    fake.enqueue(D)
    body = _drain(client)
    assert len(body["entries_written"]) == 1
    x = body["entries_written"][0]
    assert body["entries_merged"] == []
    assert len(body["candidates"]) == 1
    cand = body["candidates"][0]
    assert (
        cand["shape"],
        cand["status"],
        cand["detector_model"],
        cand["entry_id"],
    ) == (
        1,
        "written",
        "rule:shape1",
        x,
    )
    prev = _prevention(client)
    labels = {s["entry_id"]: s["provenance_label"] for s in prev["slice"]}
    assert labels[x] == "autonomous/observed"
    assert [c["resolution_id"] for c in prev["index"]] == [x]
    assert [c["observed_once"] for c in prev["index"]] == [False]
    assert prev["diagnostics"]["index_excluded_observed_once"] == 0

    _seed_push(client, "s2")
    fake.enqueue('{"surprise": false}')
    body = _drain(client)
    assert body["entries_written"] == []
    assert body["entries_merged"] == [x]
    assert len(body["candidates"]) == 1
    assert (body["candidates"][0]["status"], body["candidates"][0]["entry_id"]) == (
        "merged",
        x,
    )
    hints = _hints(tmp_path, x)
    assert hints["resolution"]["observed_sessions"] == 2
    assert hints["surprise_capture"]["event_ids"] == ["s1:0", "s2:0"]
    prev = _prevention(client)
    assert [c["resolution_id"] for c in prev["index"]] == [x]
    labels = {s["entry_id"]: s["provenance_label"] for s in prev["slice"]}
    assert labels[x] == "autonomous/observed"
    assert prev["diagnostics"]["index_excluded_observed_once"] == 0

    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    _seed_push(client, "s3")
    fake.enqueue('{"surprise": false}')
    body = _drain(client)
    assert body["entries_written"] == body["entries_merged"] == []
    assert len(body["candidates"]) == 1
    assert (body["candidates"][0]["status"], body["candidates"][0]["entry_id"]) == (
        "shadow",
        None,
    )
    hints = _hints(tmp_path, x)
    assert hints["resolution"]["observed_sessions"] == 2
    assert hints["surprise_capture"]["event_ids"] == ["s1:0", "s2:0"]
    assert len(fake.generate_calls) == 4
    assert [c[1] for c in fake.generate_calls] == [
        SURPRISE_DETECTOR_SYSTEM,
        SURPRISE_DISTILLER_SYSTEM,
        SURPRISE_DETECTOR_SYSTEM,
        SURPRISE_DETECTOR_SYSTEM,
    ]
    # the shadow candidate got a dry run: an exact match from another
    # session WOULD merge into x, and nothing was written.
    conn = sqlite3.connect(database.sqlite_service_db_path())
    conn.row_factory = sqlite3.Row
    try:
        dry = conn.execute("SELECT * FROM surprise_dry_runs").fetchall()
    finally:
        conn.close()
    assert [(r["would_outcome"], r["mode"]) for r in dry] == [
        ("would_merge", "interactive")
    ]
    assert json.loads(dry[0]["payload"])["matched_entry_id"] == x


# --- failure-context lineage --------------------------------------------------

_GATE_ROW_SQL = (
    "INSERT INTO gate_decisions (decision_id, session_id, harness, mode, tool,"
    " target, target_class, decision, ts, received_ts) VALUES ($1, $2,"
    " 'claude-code', 'interactive', 'Bash', $3, 'git push', $4, $5, $5)"
)


async def _gate_row(
    pool: SqlitePool,
    session: str,
    target: str = "git push origin main",
    decision: str = "failure_context",
) -> None:
    await pool.execute(
        _GATE_ROW_SQL,
        f"cc:{session}:toolu_{target}:{decision}",
        session,
        target,
        decision,
        _CREATED_AT,
    )


@pytest.mark.parametrize("decision", ["failure_context", "failure_context_repeat"])
async def test_failure_context_lineage_suppresses(
    pool: SqlitePool, kb: Any, llm: FakeLLM, decision: str
) -> None:
    await _gate_row(pool, "s1", decision=decision)
    c = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c], "on")
    assert result == DistillResult([], [])
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("gate_induced", "failure_context")
    assert llm.generate_calls == []
    assert (await _cand_row(pool, c.id))["status"] == "rejected"


async def test_failure_context_lineage_other_target(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    await _gate_row(pool, "s1", target="git push github main")
    c = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c], "on")
    assert len(result.entries_written) == 1
    assert (await _rows(pool))[0]["outcome"] == "written"


async def test_failure_context_lineage_other_session(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    await _gate_row(pool, "s2")
    c = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c], "on")
    assert len(result.entries_written) == 1


async def test_failure_context_lineage_does_not_corroborate(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    c1 = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c1], "on")
    (x,) = result.entries_written
    await _gate_row(pool, "s2")
    c2 = await _s(pool, "s2")
    llm.enqueue(D)
    result2 = await distill_candidates(pool, kb, [c2], "on")
    assert result2 == DistillResult([], [])
    row = (await _rows(pool))[-1]
    assert (row["outcome"], row["reason"]) == ("gate_induced", "failure_context")
    entry = await kb.get(x)
    assert _res_of(entry)["observed_sessions"] == 1


# --- session mode recorded from the last turn event ---------------------------

_TURN_MODE_INSERT = (
    "INSERT INTO turn_events (event_id, session_id, harness, mode, turn_index,"
    " ts, capture_mode, received_ts) VALUES ($1, $2, 'claude-code', $3, 0, $4,"
    " 'on', $4)"
)


@pytest.mark.parametrize("mode", ["headless", "interactive", None])
async def test_new_entry_records_turn_mode(
    pool: SqlitePool, kb: Any, llm: FakeLLM, mode: str | None
) -> None:
    c1 = await _insert(
        pool, 1, "s1", "p", ["s1:0", "s1:1"], "rule:shape1", _s1_output()
    )
    if mode is not None:
        # only the LAST turn event's mode counts
        other = "interactive" if mode == "headless" else "headless"
        await pool.execute(_TURN_MODE_INSERT, "s1:0", "s1", other, _CREATED_AT)
        await pool.execute(_TURN_MODE_INSERT, "s1:1", "s1", mode, _CREATED_AT)
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c1], "on")
    (x,) = result.entries_written
    entry = await kb.get(x)
    sc = entry.hints["surprise_capture"]
    if mode is None:
        assert "mode" not in sc
    else:
        assert sc["mode"] == mode
    resolutions, _ = await load_resolutions(kb.db, "p", True)
    (r,) = [r for r in resolutions if r.entry_id == x]
    assert r.mode == mode
    assert r.observed_once is (mode != "interactive")


# --- critic pass ----------------------------------------------------------------

_REAL_GET_CRITIC_LLM = surprise_worker.get_critic_llm

_CRITIC_TURN_INSERT = (
    "INSERT INTO turn_events (event_id, session_id, harness, mode, turn_index,"
    " ts, user_prompt, items, capture_mode, received_ts) VALUES ($1, $2,"
    " 'claude-code', 'interactive', $3, $4, $5, $6, 'on', $4)"
)


def _critic_reply(**kw: Any) -> str:
    obj = json.loads(CRITIC_ACCEPT)
    obj.update(kw)
    return json.dumps(obj)


async def test_critic_rejection_prevents_write(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    critic.enqueue(
        _critic_reply(scope_ok=False, reason="stretches a narrow correction")
    )
    llm.enqueue(D)
    c = await _s(pool, "s1")
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult([], [])
    assert await _lessons(kb) == 0
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == (
        "not_durable",
        "critic: stretches a narrow correction",
    )
    verdict = json.loads(row["verdict"])
    assert verdict["critic_version"] == SURPRISE_CRITIC_VERSION
    assert verdict["critic"]["scope_ok"] is False
    assert await _cand_row(pool, c.id) == {"status": "rejected", "entry_id": None}
    assert len(critic.generate_calls) == 1


@pytest.mark.parametrize(
    "reply",
    [
        _critic_reply(supported=False),
        _critic_reply(durable=False),
        _critic_reply(misleading=True),
    ],
)
async def test_critic_any_failed_check_rejects(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic, reply: str
) -> None:
    critic.enqueue(reply)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    assert await _lessons(kb) == 0
    assert (await _rows(pool))[0]["outcome"] == "not_durable"


async def test_critic_accept_writes_as_before(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    critic.enqueue(CRITIC_ACCEPT)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    result = await distill_candidates(pool, kb, [c], "on")
    (x,) = result.entries_written
    entry = await kb.get(x)
    assert entry.short_title == _D_OBJ["short_title"]
    assert entry.long_title == _D_OBJ["long_title"]
    verdict = parse_distill_response(D)[0]
    assert verdict is not None
    res = build_resolution(c, verdict)
    assert entry.knowledge_details == build_knowledge_details(c, verdict, res)
    assert (await _rows(pool))[0]["outcome"] == "written"
    assert [call[1] for call in critic.generate_calls] == [
        surprise_worker.SURPRISE_CRITIC_SYSTEM
    ]


@pytest.mark.parametrize(
    ("reply", "outcome", "reason"),
    [
        ("not json", "unparseable", "critic: unparseable"),
        (_critic_reply(supported="yes"), "unparseable", "critic: unparseable"),
        (None, "llm_error", "critic: none"),
    ],
)
async def test_critic_unparseable_or_error_prevents_write(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    reply: str | None,
    outcome: str,
    reason: str,
) -> None:
    stub = FakeLLM()  # exhausted queue -> None
    if reply is not None:
        stub.enqueue(reply)
    monkeypatch.setattr(surprise_worker, "get_critic_llm", lambda: stub)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult([], [])
    assert await _lessons(kb) == 0
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == (outcome, reason)
    assert await _cand_row(pool, c.id) == {"status": "rejected", "entry_id": None}


async def test_critic_exception_prevents_write(
    pool: SqlitePool, kb: Any, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom(FakeLLM):
        async def generate(self, prompt: Any, *, system: Any = None) -> str | None:
            raise RuntimeError("boom")

    monkeypatch.setattr(surprise_worker, "get_critic_llm", Boom)
    llm.enqueue(D)
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    assert await _lessons(kb) == 0
    row = (await _rows(pool))[0]
    assert (row["outcome"], row["reason"]) == ("llm_error", "critic: exception")


async def test_no_critic_client_leaves_candidate_pending(
    pool: SqlitePool, kb: Any, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(surprise_worker, "get_critic_llm", lambda: None)
    c = await _s(pool, "s1")
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult()
    assert llm.generate_calls == []
    assert await _cand_row(pool, c.id) == {"status": "pending", "entry_id": None}
    assert await _rows(pool) == []


async def test_exact_match_merge_skips_critic(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    w1 = await _w(kb)
    c = await _s(pool, "s6")
    assert await distill_candidates(pool, kb, [c], "on") == DistillResult([], [w1])
    assert llm.generate_calls == []
    assert critic.generate_calls == []
    row = (await _rows(pool))[0]
    assert row["outcome"] == "merged"
    assert row["verdict"] is None


async def test_non_durable_distiller_reply_skips_critic(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    llm.enqueue('{"durable": false, "why": "one-off"}')
    c = await _s(pool, "s1")
    await distill_candidates(pool, kb, [c], "on")
    assert critic.generate_calls == []
    assert (await _rows(pool))[0]["outcome"] == "not_durable"


async def test_critic_prompt_shape2_has_user_correction_and_draft(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    await pool.execute(_CRITIC_TURN_INSERT, "s1:0", "s1", 0, _CREATED_AT, "do it", "[]")
    await pool.execute(
        _CRITIC_TURN_INSERT,
        "s1:1",
        "s1",
        1,
        _CREATED_AT,
        "no, 8080 is taken by caddy on this box",
        "[]",
    )
    c = await _t(pool, "s1", "port 8080 is free")
    llm.enqueue(D)
    await distill_candidates(pool, kb, [c], "on")
    (call,) = critic.generate_calls
    prompt = call[0]
    for text in (
        "Wrong belief: port 8080 is free",
        "Corrected fact: port 8080 is taken by caddy",
        "Evidence excerpt: 8080 is taken by caddy",
        "The human's correction: no, 8080 is taken by caddy on this box",
        f"Short title: {_D_OBJ['short_title']}",
        f"Corrected fact: {_D_OBJ['corrected_fact']}",
        f"Lesson: {_D_OBJ['lesson']}",
    ):
        assert text in prompt
    assert "do it" not in prompt


async def test_critic_prompt_shape3_has_tool_results(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    items = [
        {
            "kind": "tool_call",
            "tool_use_id": "t1",
            "tool": "Bash",
            "target": "cat config.toml",
        },
        {
            "kind": "tool_result",
            "tool_use_id": "t1",
            "is_error": False,
            "excerpt": "port = 9090",
        },
        {"kind": "assistant_text", "text": "Root cause: the port is 9090"},
    ]
    await pool.execute(
        _CRITIC_TURN_INSERT, "s3:0", "s3", 0, _CREATED_AT, None, json.dumps(items)
    )
    c = await _insert(
        pool,
        3,
        "s3",
        "p",
        ["s3:0"],
        "FakeLLM",
        {
            "wrong_belief": "the service listens on 8080",
            "corrected_fact": "the service listens on 9090",
            "evidence_excerpt": "port = 9090",
            "confidence": 0.9,
        },
    )
    llm.enqueue(D)
    await distill_candidates(pool, kb, [c], "on")
    (call,) = critic.generate_calls
    prompt = call[0]
    assert "[tool_result] port = 9090" in prompt
    assert "Wrong belief: the service listens on 8080" in prompt
    assert f"Lesson: {_D_OBJ['lesson']}" in prompt


def test_get_critic_llm_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(surprise_worker, "_CRITIC_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_DISTILLER_LLM_CACHE", {})
    monkeypatch.setenv("KB_SURPRISE_DISTILL_MODEL", "distill-model")
    llm = _REAL_GET_CRITIC_LLM()
    assert llm is not None
    assert surprise_worker.detector_model_name(llm) == "claude-sonnet-5-5"
    monkeypatch.setenv("KB_SURPRISE_CRITIC_MODEL", "critic-model")
    other = _REAL_GET_CRITIC_LLM()
    assert surprise_worker.detector_model_name(other) == "critic-model"
    assert _REAL_GET_CRITIC_LLM() is other
    assert surprise_worker._DISTILLER_LLM_CACHE == {}


def test_get_critic_llm_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surprise_worker, "_CRITIC_LLM_CACHE", {})
    monkeypatch.setitem(sys.modules, "kb_core.llm.anthropic", None)
    assert _REAL_GET_CRITIC_LLM() is None
    assert surprise_worker._CRITIC_LLM_CACHE == {}


# --- lesson expiry ------------------------------------------------------------


def _days_ahead(entry: Any) -> float:
    assert entry.expires_at is not None
    exp = entry.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=UTC)
    return (exp - datetime.now(UTC)).total_seconds() / 86400


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 30), ("7", 7), ("3650", 3650), ("0", 30), ("3651", 30), ("abc", 30)],
)
def test_lesson_ttl_env(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    if raw is None:
        monkeypatch.delenv("KB_SURPRISE_LESSON_TTL_DAYS", raising=False)
    else:
        monkeypatch.setenv("KB_SURPRISE_LESSON_TTL_DAYS", raw)
    assert lesson_ttl_days() == expected


async def test_autonomous_lesson_expiry_write_renew_permanent(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KB_SURPRISE_LESSON_TTL_DAYS", raising=False)
    c1 = await _s(pool, "s1")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [c1], "on")
    x = result.entries_written[0]
    assert 29.9 < _days_ahead(await kb.get(x)) < 30.1

    # age the expiry, then a second session renews it
    await kb.update(x, expires_at=datetime.now(UTC) + timedelta(days=1), enrich=False)
    monkeypatch.setenv("KB_SURPRISE_LESSON_TTL_DAYS", "10")
    await distill_candidates(pool, kb, [await _s(pool, "s2")], "on")
    entry = await kb.get(x)
    assert _res_of(entry)["observed_sessions"] == 2
    assert 9.9 < _days_ahead(entry) < 10.1

    # the third distinct session makes it permanent
    await distill_candidates(pool, kb, [await _s(pool, "s3")], "on")
    entry = await kb.get(x)
    assert _res_of(entry)["observed_sessions"] == 3
    assert entry.expires_at is None


async def test_lesson_ttl_env_invalid_falls_back_on_write(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_LESSON_TTL_DAYS", "nope")
    llm.enqueue(D)
    result = await distill_candidates(pool, kb, [await _s(pool, "s1")], "on")
    assert 29.9 < _days_ahead(await kb.get(result.entries_written[0])) < 30.1
    assert any("bad_lesson_ttl" in r.getMessage() for r in caplog.records)


async def test_deliberate_entry_never_given_expiry(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    x = await _w(
        kb,
        provenance={"capture": "deliberate", "grounding": "asserted"},
    )
    await distill_candidates(pool, kb, [await _s(pool, "s1")], "on")
    entry = await kb.get(x)
    assert entry.expires_at is None
    assert entry.version == 1


async def test_expired_autonomous_resolution_excluded(kb: Any) -> None:
    prov = {"capture": "autonomous", "grounding": "observed", "event_id": "e:0"}

    async def _store_res(tc: str, exp: datetime | None) -> str:
        entry = await kb.store(
            short_title=f"R {tc}",
            long_title=f"R long {tc}",
            knowledge_details="d",
            project_ref="p",
            hints={
                "resolution": {
                    "corrected_fact": f"fact {tc}",
                    "wrong_belief": "w",
                    "cue": {"tool": "Bash", "target_class": tc},
                    "provenance": prov,
                }
            },
            expires_at=exp,
            enrich=False,
        )
        return str(entry.id)

    now = datetime.now(UTC)
    live = await _store_res("git push", now + timedelta(days=5))
    permanent = await _store_res("git pull", None)
    dead = await _store_res("git fetch", now - timedelta(days=1))
    rs, _ = await load_resolutions(kb.db, "p", True)
    ids = {r.entry_id for r in rs}
    assert ids == {live, permanent}
    assert dead not in {c.resolution_id for c in build_gate_index(rs)[0]}
    assert dead not in {i.entry_id for i in build_slice(rs, [])[0]}
