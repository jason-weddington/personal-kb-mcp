"""Prevention channels: helpers on a real SQLite KB, routes, decisions, stats."""

import json
import logging
import sqlite3
import typing
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core import create_sqlite

import kb_service.database as database
from kb_service import models
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import SliceItem
from kb_service.prevention import (
    INDEX_CAP,
    Correction,
    Resolution,
    build_gate_index,
    build_slice,
    count_index_excluded_observed_once,
    load_corrections,
    load_resolutions,
    provenance_label,
    render_slice,
)
from kb_service.routes import prevention_routes
from tests.conftest import FakeKnowledgeBase, fake_user

_SWITCHES = (
    "KB_SOFT_GATE_ENABLED",
    "KB_SOFT_GATE_SHADOW",
    "KB_SOFT_GATE_DISABLED_PROJECTS",
    "KB_DELIVER_OBSERVED_ONCE",
    "KB_SURPRISE_CAPTURE",
    "KB_SOFT_GATE_MAX_DENIES_PER_TURN",
    "KB_SOFT_GATE_MAX_DENIES_PER_HOUR",
    "KB_SOFT_GATE_REARM_HOURS",
)


@pytest.fixture(autouse=True)
def _clear_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SWITCHES:
        monkeypatch.delenv(var, raising=False)


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


async def _store(kb: Any, project: str = "p", **resolution: Any) -> str:
    _COUNTER["n"] += 1
    n = _COUNTER["n"]
    entry = await kb.store(
        short_title=f"Resolution {n}",
        long_title=f"Resolution number {n}",
        knowledge_details=f"details {n}",
        project_ref=project,
        hints={"resolution": resolution},
        enrich=False,
    )
    return str(entry.id)


def _res(cue_tool: str = "Bash", cue_tc: str = "git push", **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "corrected_fact": kw.pop("corrected_fact", "push to origin"),
        "wrong_belief": kw.pop("wrong_belief", "push to github"),
        "cue": {"tool": cue_tool, "target_class": cue_tc},
        "provenance": {"capture": "deliberate", "grounding": "asserted"},
    }
    out.update(kw)
    return out


# ─── helpers on a real KB ────────────────────────────────────────────────────


async def test_load_resolutions_parsing_and_labels(kb: Any) -> None:
    auto1 = await _store(
        kb, **_res(provenance={"capture": "autonomous", "grounding": "observed"})
    )
    auto2 = await _store(
        kb,
        **_res(
            provenance={"capture": "autonomous", "grounding": "observed"},
            observed_sessions=2,
        ),
    )
    deliberate = await _store(
        kb, **_res(provenance={"capture": "deliberate", "grounding": "asserted"})
    )
    bare = await _store(kb, **{k: v for k, v in _res().items() if k != "provenance"})
    other = await _store(kb, project="q", **_res())
    glob = await _store(kb, project="q", **_res(scope="global"))

    resolutions, stats = await load_resolutions(kb.db, "p", True)
    by_id = {r.entry_id: r for r in resolutions}
    assert other not in by_id
    assert glob in by_id
    # Project-scoped first, global last.
    assert resolutions[-1].entry_id == glob
    assert by_id[auto1].observed_once is True
    assert provenance_label(by_id[auto1]) == (
        "autonomous/observed, observed once, unconfirmed"
    )
    assert by_id[auto2].observed_once is False
    assert by_id[deliberate].observed_once is False
    assert provenance_label(by_id[bare]) == "unlabelled"
    assert stats.skipped_observed_once == 0

    resolutions, stats = await load_resolutions(kb.db, "p", False)
    assert auto1 not in {r.entry_id for r in resolutions}
    assert stats.skipped_observed_once == 1


async def test_malformed_resolutions_skipped(
    kb: Any, caplog: pytest.LogCaptureFixture
) -> None:
    await _store(kb, corrected_fact=5)
    await _store(kb, **_res(scope="team"))
    await _store(kb, **_res(wrong_belief=3))
    await _store(kb, **_res(provenance={"capture": "guess"}))
    await _store(
        kb,
        **_res(cue={"tool": "Bash", "target_class": "git remote", "args_prefix": ""}),
    )
    good = await _store(kb, **_res(observed_sessions=True))
    with caplog.at_level(logging.WARNING, logger="kb_service.prevention"):
        resolutions, stats = await load_resolutions(kb.db, "p", True)
    assert [r.entry_id for r in resolutions] == [good]
    assert resolutions[0].observed_sessions == 1
    assert stats.skipped_malformed == 5
    assert any("malformed" in rec.getMessage() for rec in caplog.records)


async def test_single_malformed_counts_and_warns(
    kb: Any, caplog: pytest.LogCaptureFixture
) -> None:
    bad = await _store(kb, corrected_fact=5)
    with caplog.at_level(logging.WARNING, logger="kb_service.prevention"):
        _, stats = await load_resolutions(kb.db, "p", True)
    assert stats.skipped_malformed == 1
    assert any(bad in rec.getMessage() for rec in caplog.records)


async def test_index_admission_and_slice(kb: Any) -> None:
    push = await _store(kb, **_res())
    edit = await _store(kb, **_res(cue_tool="Edit", cue_tc="ext:py"))
    one_word = await _store(kb, **_res(cue_tc="git"))
    no_tc = await _store(kb, **_res(cue_tc=""))
    resolutions, _ = await load_resolutions(kb.db, "p", True)
    index, truncated = build_gate_index(resolutions)
    assert [c.resolution_id for c in index] == [push]
    assert truncated == 0
    items, dropped = build_slice(resolutions, [])
    assert {push, edit, one_word, no_tc} == {i.entry_id for i in items}
    assert dropped == 0


async def test_corrections_from_supersedes(kb: Any) -> None:
    old = await kb.store(
        short_title="Old belief",
        long_title="Old",
        knowledge_details="old",
        project_ref="p",
        enrich=False,
    )
    new = await kb.store(
        short_title="New fact",
        long_title="New",
        knowledge_details="new",
        project_ref="p",
        hints={"supersedes": [old.id]},
        enrich=False,
    )
    corrections = await load_corrections(kb.db, "p", 20)
    assert corrections == [
        Correction(
            entry_id=new.id, corrected_fact="New fact", wrong_belief="Old belief"
        )
    ]
    items, _ = build_slice([], corrections)
    assert items[0].provenance_label == "supersedes-edge"


def _resolution(i: int, tc: str = "git push") -> Resolution:
    return Resolution(
        entry_id=f"kb-{i:05d}",
        updated_at="2026-10-07",
        wrong_belief="w",
        corrected_fact=f"fact {i}",
        evidence="",
        cue_tool="Bash",
        cue_target_class=tc,
        capture="deliberate",
        grounding="asserted",
        observed_sessions=1,
        observed_once=False,
    )


def test_gate_admits_only_deliberate_or_observed() -> None:
    base = _resolution(1).__dict__
    deliberate = Resolution(
        **{**base, "entry_id": "kb-1", "capture": "deliberate", "grounding": "asserted"}
    )
    observed = Resolution(
        **{**base, "entry_id": "kb-2", "capture": "autonomous", "grounding": "observed"}
    )
    seeded = Resolution(
        **{**base, "entry_id": "kb-3", "capture": "autonomous", "grounding": "asserted"}
    )
    unlabelled = Resolution(
        **{**base, "entry_id": "kb-4", "capture": None, "grounding": None}
    )
    allr = [deliberate, observed, seeded, unlabelled]
    index, _ = build_gate_index(allr)
    assert [c.resolution_id for c in index] == ["kb-1", "kb-2"]
    items, _ = build_slice(allr, [])
    assert {i.entry_id for i in items} == {"kb-1", "kb-2", "kb-3", "kb-4"}


def test_caps() -> None:
    many = [_resolution(i) for i in range(205)]
    index, truncated = build_gate_index(many)
    assert len(index) == INDEX_CAP == 200
    assert truncated == 5
    items, dropped = build_slice(many[:25], [])
    assert len(items) == 20
    assert dropped == 5


def test_build_slice_dedups() -> None:
    r = _resolution(1)
    c = Correction(entry_id=r.entry_id, corrected_fact="x", wrong_belief="y")
    items, _ = build_slice([r], [c])
    assert len(items) == 1


def test_provenance_label_half() -> None:
    r = Resolution(
        **{**_resolution(1).__dict__, "capture": "deliberate", "grounding": None}
    )
    assert provenance_label(r) == "deliberate/?"


def test_render_slice() -> None:
    assert render_slice("p", []) == ""
    item = SliceItem(
        entry_id="kb-00001",
        corrected_fact="Push to origin",
        wrong_belief="Push to github",
        provenance_label="unlabelled",
    )
    assert render_slice("p", [item]) == (
        "Known gotchas for p from the KB (each replaces an earlier wrong belief):\n"
        "- Push to origin (was: Push to github) [unlabelled; kb-00001]"
    )
    no_was = item.model_copy(update={"wrong_belief": ""})
    assert " (was: " not in render_slice("p", [no_was])
    long = item.model_copy(update={"corrected_fact": "x" * 500, "wrong_belief": ""})
    assert "x" * 300 + " [" in render_slice("p", [long])
    assert "x" * 301 not in render_slice("p", [long])
    big = [
        item.model_copy(update={"corrected_fact": "y" * 400, "wrong_belief": "z" * 400})
    ] * 20
    text = render_slice("p", big)
    assert len(text) <= 4000
    assert text.startswith("Known gotchas for p from the KB")


# ─── GET /api/kb/prevention ──────────────────────────────────────────────────


@pytest.fixture
def real_client(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    kb: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(fake_kb, "db", kb.db)
    app.dependency_overrides[get_current_user] = fake_user
    yield client


async def test_prevention_gate_switches(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    push = await _store(kb, **_res())
    resp = real_client.get("/api/kb/prevention", params={"project": "p"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["gate"] == {
        "enabled": False,
        "shadow": True,
        "max_denies": 1000,
        "max_denies_per_turn": 1,
        "max_denies_per_hour": 6,
        "rearm_hours": 24,
    }
    assert body["index"] == []
    assert [s["entry_id"] for s in body["slice"]] == [push]
    assert body["slice_text"].startswith("Known gotchas for p")

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "yes")
    assert (
        real_client.get("/api/kb/prevention", params={"project": "p"}).json()["index"]
        == []
    )

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert [c["resolution_id"] for c in body["index"]] == [push]
    assert body["gate"]["enabled"] is True
    assert body["gate"]["shadow"] is True

    monkeypatch.setenv("KB_SOFT_GATE_SHADOW", "false")
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["gate"]["shadow"] is False

    monkeypatch.setenv("KB_SOFT_GATE_DISABLED_PROJECTS", " P , x")
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["gate"]["enabled"] is False
    assert body["index"] == []
    assert body["slice"]


async def test_prevention_global_scope_ignores_json_spacing(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    spaced = await _store(kb, "a", **_res(cue_tc="git push", scope="global"))
    compact = await _store(kb, "a", **_res(cue_tc="git pull", scope="global"))
    local = await _store(kb, "a", **_res(cue_tc="git fetch", scope="project"))
    cursor = await kb.db.execute(
        "SELECT hints FROM knowledge_entries WHERE id = ?", (compact,)
    )
    row = await cursor.fetchone()
    await kb.db.execute(
        "UPDATE knowledge_entries SET hints = ? WHERE id = ?",
        (json.dumps(json.loads(row[0]), separators=(",", ":")), compact),
    )
    await kb.db.commit()
    assert '"scope":"global"' in row[0] or '"scope": "global"' in row[0]
    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    body = real_client.get("/api/kb/prevention", params={"project": "b"}).json()
    ids = {c["resolution_id"] for c in body["index"]}
    assert ids == {spaced, compact}
    assert local not in ids


async def test_prevention_observed_once_flag(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    auto = await _store(
        kb, **_res(provenance={"capture": "autonomous", "grounding": "observed"})
    )
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    labels = {s["entry_id"]: s["provenance_label"] for s in body["slice"]}
    assert labels[auto].endswith(", observed once, unconfirmed")
    monkeypatch.setenv("KB_DELIVER_OBSERVED_ONCE", "FALSE")
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["slice"] == []
    assert body["diagnostics"]["skipped_observed_once"] == 1


async def test_prevention_cwd_resolution_and_empty(
    kb: Any, real_client: TestClient
) -> None:
    await _store(kb, **_res())
    body = real_client.get("/api/kb/prevention", params={"cwd": "/home/j/git/p"}).json()
    assert body["project"] == "p"
    assert body["slice"]
    body = real_client.get("/api/kb/prevention").json()
    assert body["project"] == ""
    assert body["slice"] == []


async def test_prevention_db_failure_is_inert(
    real_client: TestClient, fake_kb: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    monkeypatch.setattr(fake_kb, "db", Boom())
    resp = real_client.get("/api/kb/prevention", params={"project": "p"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["gate"]["enabled"] is False
    assert body["index"] == []
    assert body["slice"] == []
    assert body["slice_text"] == ""


def test_prevention_surprise_capture_default_off(real_client: TestClient) -> None:
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["surprise_capture"] == "off"


@pytest.mark.parametrize(
    ("value", "expected"), [("shadow", "shadow"), ("bogus", "off")]
)
def test_prevention_surprise_capture_values(
    real_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: str,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", value)
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["surprise_capture"] == expected


def test_prevention_surprise_capture_no_project(
    real_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    caplog.set_level(logging.INFO)
    body = real_client.get("/api/kb/prevention").json()
    assert body["project"] == ""
    assert body["surprise_capture"] == "on"
    assert "no_project=true" in caplog.text


def test_prevention_inert_reports_surprise_capture(
    real_client: TestClient, fake_kb: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    monkeypatch.setattr(fake_kb, "db", Boom())
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["gate"]["enabled"] is False
    assert body["surprise_capture"] == "shadow"


def test_prevention_surprise_capture_ignores_listener_switch(
    real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert body["surprise_capture"] == "shadow"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/kb/prevention"),
        ("post", "/api/kb/prevention/decisions"),
        ("get", "/api/kb/prevention/stats"),
    ],
)
def test_routes_require_auth(client: TestClient, method: str, path: str) -> None:
    if method == "post":
        resp = client.post(path, json={"rows": []})
    else:
        resp = client.get(path)
    assert resp.status_code == 401


# ─── decisions + stats on the real SQLite service DB ─────────────────────────

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)


@pytest.fixture
def local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The real app in local no-auth mode, service DB in *tmp_path*."""
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _query(sql: str, *args: Any) -> list[sqlite3.Row]:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _row(i: int, **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "decision_id": f"cc:s1:toolu_{i}:denied",
        "session_id": "s1",
        "tool": "Bash",
        "target": "git push origin main",
        "target_class": "git push",
        "resolution_id": "kb-00001",
        "decision": "denied",
        "ts": "2026-10-07T12:00:00+00:00",
    }
    out.update(kw)
    return out


def test_decisions_insert_and_duplicates(local_client: TestClient) -> None:
    rows = [_row(i) for i in range(3)]
    resp = local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    assert resp.json() == {"inserted": 3, "duplicates": 0}
    resp = local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    assert resp.json() == {"inserted": 0, "duplicates": 3}


def test_decisions_batch_cap(local_client: TestClient) -> None:
    rows = [_row(i) for i in range(501)]
    resp = local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    assert resp.status_code == 422


def test_decisions_value_mapping(local_client: TestClient) -> None:
    rows = [
        _row(
            1,
            decision="retry",
            decision_id="cc:s1:toolu_1:retry",
            ts="2026-10-07T12:00:00.123Z",
            retry_changed_command=None,
            target="x" * 600,
            reason_excerpt="r" * 1200,
            shadow=True,
        ),
        _row(2, ts="not a timestamp", retry_changed_command=False),
        _row(3, ts="2026-10-07T12:00:00", retry_changed_command=True),
    ]
    local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    stored = _query("SELECT * FROM gate_decisions ORDER BY id")
    assert stored[0]["ts"] == "2026-10-07T12:00:00+00:00"
    assert stored[0]["retry_changed_command"] is None
    assert len(stored[0]["target"]) == 500
    assert len(stored[0]["reason_excerpt"]) == 1000
    assert stored[0]["shadow"] == 1
    assert stored[1]["ts"] == stored[1]["received_ts"]
    assert stored[1]["retry_changed_command"] == 0
    assert stored[2]["ts"] == "2026-10-07T12:00:00+00:00"
    assert stored[2]["retry_changed_command"] == 1


def _rows_at(*stamps: str, session_id: str = "s1") -> list[dict[str, Any]]:
    return [
        _row(
            i,
            session_id=session_id,
            decision_id=f"cc:{session_id}:toolu_{i}:denied",
            resolution_id=f"kb-{i:05d}",
            ts=ts,
        )
        for i, ts in enumerate(stamps)
    ]


def _invariant_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return any("gate_invariant_violation" in r.getMessage() for r in caplog.records)


def test_decisions_invariant_warning_per_hour(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    # Six denies inside one hour: at the default cap, no warning.
    six = [f"2026-10-07T12:{m:02d}:00+00:00" for m in range(0, 60, 10)]
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        local_client.post("/api/kb/prevention/decisions", json={"rows": _rows_at(*six)})
    assert not _invariant_logged(caplog)
    # A seventh at 12:59 puts seven in [12:00, 13:00): over the cap.
    seventh = _row(
        99, decision_id="cc:s1:toolu_99:denied", ts="2026-10-07T12:59:00+00:00"
    )
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        local_client.post("/api/kb/prevention/decisions", json={"rows": [seventh]})
    assert _invariant_logged(caplog)


def test_decisions_many_denies_spread_over_hours_ok(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    # A weeks-long session: 20 denies an hour apart never trips the check.
    stamps = [f"2026-10-{7 + h // 24:02d}T{h % 24:02d}:00:00+00:00" for h in range(20)]
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        local_client.post(
            "/api/kb/prevention/decisions", json={"rows": _rows_at(*stamps)}
        )
    assert not _invariant_logged(caplog)


def test_decisions_invariant_uses_env_limit(
    local_client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_SOFT_GATE_MAX_DENIES_PER_HOUR", "2")
    stamps = ("2026-10-07T12:00:00+00:00", "2026-10-07T12:10:00+00:00")
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        local_client.post(
            "/api/kb/prevention/decisions", json={"rows": _rows_at(*stamps)}
        )
    assert not _invariant_logged(caplog)
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        local_client.post(
            "/api/kb/prevention/decisions",
            json={"rows": [_row(50, decision_id="w50", decision="would_deny")]},
        )
    assert _invariant_logged(caplog)


def test_gate_settings_from_env(
    real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SOFT_GATE_MAX_DENIES_PER_TURN", " 3 ")
    monkeypatch.setenv("KB_SOFT_GATE_MAX_DENIES_PER_HOUR", "1000")
    monkeypatch.setenv("KB_SOFT_GATE_REARM_HOURS", "1")
    gate = real_client.get("/api/kb/prevention", params={"project": ""}).json()["gate"]
    assert gate["max_denies_per_turn"] == 3
    assert gate["max_denies_per_hour"] == 1000
    assert gate["rearm_hours"] == 1
    assert gate["max_denies"] == prevention_routes.LEGACY_MAX_DENIES == 1000


@pytest.mark.parametrize("bad", ["0", "1001", "-5", "abc", "2.5"])
def test_gate_settings_env_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    for var in (
        "KB_SOFT_GATE_MAX_DENIES_PER_TURN",
        "KB_SOFT_GATE_MAX_DENIES_PER_HOUR",
        "KB_SOFT_GATE_REARM_HOURS",
    ):
        monkeypatch.setenv(var, bad)
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        gate = prevention_routes.gate_settings("p")
    assert (gate.max_denies_per_turn, gate.max_denies_per_hour, gate.rearm_hours) == (
        1,
        6,
        24,
    )
    assert gate.max_denies == 1000
    warned = [
        r.getMessage()
        for r in caplog.records
        if "invalid KB_SOFT_GATE" in r.getMessage()
    ]
    assert len(warned) == 3


def test_gate_settings_unset_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        gate = prevention_routes.gate_settings("p")
    assert (gate.max_denies_per_turn, gate.max_denies_per_hour, gate.rearm_hours) == (
        1,
        6,
        24,
    )
    assert not caplog.records


def test_inert_response_carries_new_settings(
    real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = prevention_routes._inert("p", tool_map={}).gate
    assert gate.enabled is False
    assert gate.max_denies == 1000
    assert gate.rearm_hours == 24


def test_rearmed_and_overridden_rows_ingest(local_client: TestClient) -> None:
    rows = [
        _row(
            1,
            decision="rearmed",
            decision_id="cc:s1:rearmed:1",
            tool="SessionStart",
            resolution_id="",
            source="compact",
        ),
        _row(2, decision="overridden", decision_id="cc:s1:toolu_2:overridden"),
        _row(
            3,
            decision="skipped_cap",
            decision_id="cc:s1:toolu_3:skipped_cap",
            reason="per_hour",
        ),
    ]
    assert local_client.post(
        "/api/kb/prevention/decisions", json={"rows": rows}
    ).json() == {"inserted": 3, "duplicates": 0}
    stored = _query("SELECT decision, reason_excerpt FROM gate_decisions ORDER BY id")
    assert [(r["decision"], r["reason_excerpt"]) for r in stored] == [
        ("rearmed", "compact"),
        ("overridden", None),
        ("skipped_cap", "per_hour"),
    ]
    counts = local_client.get("/api/kb/prevention/stats").json()["counts"]
    assert counts["rearmed"] == 1
    assert counts["overridden"] == 1


def test_stats_repeat_deny_respects_rearm(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    def deny(i: int, ts: str, sid: str) -> dict[str, Any]:
        return _row(i, session_id=sid, decision_id=f"d{i}", ts=ts)

    rows = [
        # sA: same lesson again after 25h — re-armed by time, not a repeat
        deny(1, "2026-10-01T00:00:00+00:00", "sA"),
        deny(2, "2026-10-02T01:00:00+00:00", "sA"),
        # sB: again after 2h but a compaction re-arm in between — not a repeat
        deny(3, "2026-10-01T00:00:00+00:00", "sB"),
        _row(
            4,
            session_id="sB",
            decision="rearmed",
            decision_id="r4",
            tool="SessionStart",
            resolution_id="",
            ts="2026-10-01T01:00:00+00:00",
        ),
        deny(5, "2026-10-01T02:00:00+00:00", "sB"),
        # sC: again after 2h, no re-arm — a repeat
        deny(6, "2026-10-01T00:00:00+00:00", "sC"),
        deny(7, "2026-10-01T02:00:00+00:00", "sC"),
    ]
    _post(local_client, *rows)
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        body = local_client.get("/api/kb/prevention/stats").json()
    assert body["invariant_violations"]["repeat_deny_pairs"] == 1
    assert body["invariant_violations"]["over_cap_sessions"] == 0
    assert _invariant_logged(caplog)


def test_max_in_hour_window() -> None:
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 10, 7, tzinfo=UTC)
    mins = [0, 10, 59, 60, 61, 200]
    stamps = [base + timedelta(minutes=m) for m in mins]
    assert prevention_routes._max_in_hour([*stamps, None]) == 4  # 10, 59, 60, 61
    assert prevention_routes._max_in_hour(stamps[:4]) == 3  # 60 is outside [0, 60)
    assert prevention_routes._max_in_hour([]) == 0
    assert prevention_routes._parse_ts("garbage") is None
    assert prevention_routes._parse_ts(None) is None
    assert prevention_routes._parse_ts(base.replace(tzinfo=None)) == base
    assert prevention_routes._parse_ts("2026-10-07T00:00:00") == base


def test_decisions_db_failure(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom() -> Any:
        raise RuntimeError("down")

    monkeypatch.setattr(prevention_routes, "get_db", _boom)
    resp = local_client.post("/api/kb/prevention/decisions", json={"rows": [_row(1)]})
    assert resp.status_code == 200
    assert resp.json() == {"inserted": 0, "duplicates": 0}
    resp = local_client.get("/api/kb/prevention/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["retry_changed_share"] is None
    assert body["by_host"] == []
    assert set(body["counts"]) == set(prevention_routes._DECISIONS)


def test_stats_empty(local_client: TestClient) -> None:
    body = local_client.get("/api/kb/prevention/stats").json()
    assert body["counts"] == dict.fromkeys(prevention_routes._DECISIONS, 0)
    assert body["retry_changed_share"] is None
    assert body["would_deny_precision"] is None
    assert (
        local_client.get("/api/kb/prevention/stats", params={"hours": 0}).status_code
        == 422
    )


def _insert_failure(session_id: str, tool: str, tc: str, ts: str) -> None:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    try:
        conn.execute(
            "INSERT INTO failure_events (event_id, cue_key, normalizer_version,"
            " session_id, harness, mode, host_class, project, project_source, tool,"
            " target, target_class, normalized_error, error_rule, raw_error_excerpt,"
            " is_interrupt, ts, received_ts) VALUES (?, 'k', 1, ?, 'claude-code',"
            " 'interactive', 'linux', 'p', 'kb_project', ?, '', ?, 'e', 'keyword',"
            " 'e', 0, ?, ?)",
            (f"ev-{session_id}-{ts}", session_id, tool, tc, ts, ts),
        )
        conn.commit()
    finally:
        conn.close()


def test_stats_aggregates(
    local_client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_SOFT_GATE_MAX_DENIES_PER_HOUR", "2")
    rows = [
        # s1: 3 denied rows (over cap), two for the same resolution (repeat pair)
        _row(1),
        _row(2),
        _row(3, resolution_id="kb-00002"),
        _row(4, decision="retry", decision_id="r1", retry_changed_command=True),
        _row(5, decision="retry", decision_id="r2", retry_changed_command=False),
        _row(6, decision="retry", decision_id="r3", retry_changed_command=None),
        _row(
            7,
            session_id="s2",
            decision="would_deny",
            decision_id="w1",
            resolution_id="kb-00003",
            shadow=True,
        ),
        _row(
            8,
            session_id="s2",
            decision="armed",
            decision_id="a1",
            tool="SessionStart",
            resolution_id="",
            host="h1",
            hook_version="1.0",
        ),
        _row(
            9,
            session_id="s3",
            decision="armed",
            decision_id="a2",
            tool="SessionStart",
            resolution_id="",
            host="h1",
        ),
        _row(
            10,
            decision="summary",
            decision_id="sum1",
            tool="Stop",
            resolution_id="",
            pre_tool_errors=2,
        ),
    ]
    resp = local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    assert resp.json()["inserted"] == 10
    _insert_failure("s2", "Bash", "git push", "2026-10-07T12:00:05+00:00")

    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        body = local_client.get("/api/kb/prevention/stats").json()
    assert body["counts"]["denied"] == 3
    assert body["counts"]["skipped_cap"] == 0
    assert body["armed_sessions"] == 2
    assert body["retries"] == 2
    assert body["retries_changed"] == 1
    assert body["retries_abandoned"] == 1
    assert body["retry_changed_share"] == 0.5
    assert body["would_deny_precision"] == 1.0
    assert body["pre_tool_errors_total"] == 2
    by_res = {r["resolution_id"]: r for r in body["by_resolution"]}
    assert by_res["kb-00001"]["denied"] == 2
    assert by_res["kb-00001"]["retries"] == 2
    assert by_res["kb-00001"]["retries_changed"] == 1
    assert by_res["kb-00003"]["would_deny"] == 1
    assert by_res["kb-00003"]["followed_by_failure"] == 1
    assert [r["resolution_id"] for r in body["by_resolution"]] == sorted(by_res)
    assert body["by_host"] == [
        {
            "harness": "claude-code",
            "mode": "interactive",
            "host": "h1",
            "armed": 2,
            "last_ts": body["by_host"][0]["last_ts"],
            "hook_version": "1.0",
        }
    ]
    assert body["invariant_violations"] == {
        "over_cap_sessions": 1,
        "repeat_deny_pairs": 1,
        "repeat_failure_context_pairs": 0,
    }
    assert any("gate_invariant_violation" in r.getMessage() for r in caplog.records)


def test_stats_failure_before_deny_not_counted(local_client: TestClient) -> None:
    rows = [_row(1, session_id="s9", decision="would_deny", decision_id="w9")]
    local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    _insert_failure("s9", "Bash", "git push", "2026-10-07T11:00:00+00:00")
    body = local_client.get("/api/kb/prevention/stats").json()
    assert body["would_deny_precision"] == 0.0


async def test_args_prefix_carried_to_index(kb: Any) -> None:
    rid = await _store(
        kb,
        **_res(
            cue={"tool": "Bash", "target_class": "git remote", "args_prefix": "add"}
        ),
    )
    resolutions, stats = await load_resolutions(kb.db, "p", True)
    assert stats.skipped_malformed == 0
    index, _ = build_gate_index(resolutions)
    assert [(c.resolution_id, c.args_prefix) for c in index] == [(rid, "add")]


# ─── D7: observed-once resolutions stay out of the gate ─────────────────────


def _once(entry_id: str, sessions: int = 1) -> Resolution:
    base = _resolution(1).__dict__
    return Resolution(
        **{
            **base,
            "entry_id": entry_id,
            "capture": "autonomous",
            "grounding": "observed",
            "observed_sessions": sessions,
            "observed_once": sessions < 2,
        }
    )


def test_observed_once_excluded_from_index_and_last_in_slice() -> None:
    a = _once("kb-A")
    b = _once("kb-B", sessions=2)
    c = Resolution(
        **{
            **_resolution(1).__dict__,
            "entry_id": "kb-C",
            "capture": "deliberate",
            "grounding": "asserted",
        }
    )
    index, _ = build_gate_index([a, b, c])
    assert [i.resolution_id for i in index] == ["kb-B", "kb-C"]
    assert count_index_excluded_observed_once([a, b, c]) == 1
    items, _ = build_slice([a, b, c], [])
    assert [i.entry_id for i in items] == ["kb-B", "kb-C", "kb-A"]


def test_slice_observed_once_cap() -> None:
    once = [_once(f"kb-O{i}") for i in range(25)]
    d = Resolution(**{**_resolution(1).__dict__, "entry_id": "kb-D"})
    corr = Correction(entry_id="kb-corr", corrected_fact="x", wrong_belief="y")
    items, dropped = build_slice([*once, d], [corr])
    assert [i.entry_id for i in items] == [
        "kb-D",
        "kb-corr",
        "kb-O0",
        "kb-O1",
        "kb-O2",
        "kb-O3",
        "kb-O4",
    ]
    assert dropped == 20


async def test_prevention_observed_once_index_exclusion(
    kb: Any,
    real_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    caplog.set_level(logging.INFO, logger=prevention_routes.logger.name)
    prov = {"capture": "autonomous", "grounding": "observed", "event_id": "s1:0"}
    o1 = await _store(kb, **_res(provenance=prov))
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert o1 not in {c["resolution_id"] for c in body["index"]}
    assert o1 in {s["entry_id"] for s in body["slice"]}
    assert body["diagnostics"]["index_excluded_observed_once"] == 1

    o2 = await _store(kb, **_res(provenance=prov, observed_sessions=2))
    caplog.clear()
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert o2 in {c["resolution_id"] for c in body["index"]}
    fetch = [r for r in caplog.records if "prevention_fetch project=p" in r.message]
    assert len(fetch) == 1
    index_ids = fetch[0].message.split("index_ids=", 1)[1]
    assert o2 in index_ids
    assert o1 not in index_ids

    monkeypatch.setenv("KB_DELIVER_OBSERVED_ONCE", "FALSE")
    body = real_client.get("/api/kb/prevention", params={"project": "p"}).json()
    assert o1 not in {c["resolution_id"] for c in body["index"]}
    assert o1 not in {s["entry_id"] for s in body["slice"]}
    assert body["diagnostics"]["index_excluded_observed_once"] == 0


async def test_prevention_observed_once_in_index_tripwire(
    kb: Any,
    real_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import kb_service.prevention as prevention

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    monkeypatch.setattr(prevention, "_gate_trusted", lambda r: True)
    o1 = await _store(
        kb,
        **_res(
            provenance={
                "capture": "autonomous",
                "grounding": "observed",
                "event_id": "s1:0",
            }
        ),
    )
    with caplog.at_level(logging.WARNING, logger=prevention_routes.logger.name):
        real_client.get("/api/kb/prevention", params={"project": "p"})
    warnings = [
        r.message
        for r in caplog.records
        if "tripwire=observed_once_in_index" in r.message
    ]
    assert len(warnings) == 1
    assert o1 in warnings[0]


async def test_prevention_asserted_in_index_tripwire(
    kb: Any,
    real_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import kb_service.prevention as prevention

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    monkeypatch.setattr(prevention, "_gate_trusted", lambda r: True)
    a1 = await _store(
        kb,
        **_res(
            provenance={"capture": "autonomous", "grounding": "asserted"},
            observed_sessions=2,
        ),
    )
    d1 = await _store(kb, **_res())
    with caplog.at_level(logging.WARNING, logger=prevention_routes.logger.name):
        real_client.get("/api/kb/prevention", params={"project": "p"})
    assert not [
        r for r in caplog.records if "tripwire=observed_once_in_index" in r.message
    ]
    warnings = [
        r.message for r in caplog.records if "tripwire=asserted_in_index" in r.message
    ]
    assert len(warnings) == 1
    assert a1 in warnings[0]
    assert d1 not in warnings[0]


async def test_autonomous_asserted_never_gates(kb: Any) -> None:
    ids: list[str] = []
    cases = [
        (1, "interactive"),
        (1, "headless"),
        (2, "headless"),
        (2, "interactive"),
        (3, "interactive"),
    ]
    for shape, mode in cases:
        for n in (1, 2, 3):
            entry = await kb.store(
                short_title=f"asserted {shape} {mode} {n}",
                long_title=f"asserted case {shape} {mode} {n}",
                knowledge_details="d",
                project_ref="p",
                hints={
                    "resolution": _res(
                        corrected_fact=f"fact {shape} {mode} {n}",
                        provenance={"capture": "autonomous", "grounding": "asserted"},
                        observed_sessions=n,
                    ),
                    "surprise_capture": {"shape": shape, "mode": mode},
                },
                enrich=False,
            )
            ids.append(str(entry.id))
    control_entry = await kb.store(
        short_title="control",
        long_title="control case",
        knowledge_details="d",
        project_ref="p",
        hints={
            "resolution": _res(
                corrected_fact="control fact",
                provenance={
                    "capture": "autonomous",
                    "grounding": "observed",
                    "event_id": "c:0",
                },
                observed_sessions=1,
            ),
            "surprise_capture": {"shape": 2, "mode": "headless"},
        },
        enrich=False,
    )
    control = str(control_entry.id)
    rs, _ = await load_resolutions(kb.db, "p", True)
    assert [c.resolution_id for c in build_gate_index(rs)[0]] == [control]
    assert {i.entry_id for i in build_slice(rs, [])[0]} == {*ids, control}


# ─── first-sighting admission by shape ──────────────────────────────────────


@pytest.mark.parametrize(
    ("shape", "mode", "admitted"),
    [
        (1, "interactive", True),
        (1, "headless", False),
        (1, None, False),
        (1, "batch", False),
        (2, None, True),
        (2, "headless", True),
        (2, "interactive", True),
        (3, "interactive", False),
        (None, "interactive", False),
        ("x", None, False),
        (4, None, False),
        (4, "headless", False),
    ],
)
async def test_observed_once_gate_admission_by_shape(
    kb: Any, shape: Any, mode: Any, admitted: bool
) -> None:
    prov = {"capture": "autonomous", "grounding": "observed"}
    hints: dict[str, Any] = {"resolution": _res(provenance=prov)}
    if shape is not None:
        hints["surprise_capture"] = {"shape": shape}
        if mode is not None:
            hints["surprise_capture"]["mode"] = mode
    entry = await kb.store(
        short_title="s",
        long_title="shape case",
        knowledge_details="d",
        project_ref="p",
        hints=hints,
        enrich=False,
    )
    resolutions, _ = await load_resolutions(kb.db, "p", True)
    (r,) = [x for x in resolutions if x.entry_id == str(entry.id)]
    assert r.observed_once is (not admitted)
    assert r.shape == (shape if shape in (1, 2, 3) else None)
    expected_mode = mode if mode in ("interactive", "headless") else None
    assert r.mode == (expected_mode if shape is not None else None)
    index, _ = build_gate_index(resolutions)
    assert (str(entry.id) in [c.resolution_id for c in index]) is admitted
    assert count_index_excluded_observed_once(resolutions) == (0 if admitted else 1)
    label = provenance_label(r)
    assert ("observed once" not in label) is admitted
    entries = [c for c in index if c.resolution_id == str(entry.id)]
    assert [c.observed_once for c in entries] == ([False] if admitted else [])
    items, _ = build_slice(resolutions, [])
    text = render_slice("p", items)
    assert ("observed once" not in text) is admitted


@pytest.mark.parametrize("mode", ["headless", None])
async def test_headless_shape1_promoted_at_second_sighting(
    kb: Any, mode: str | None
) -> None:
    prov = {"capture": "autonomous", "grounding": "observed"}
    sc: dict[str, Any] = {"shape": 1}
    if mode is not None:
        sc["mode"] = mode
    entry = await kb.store(
        short_title="s",
        long_title="headless shape 1 seen twice",
        knowledge_details="d",
        project_ref="p",
        hints={
            "resolution": _res(provenance=prov, observed_sessions=2),
            "surprise_capture": sc,
        },
        enrich=False,
    )
    resolutions, _ = await load_resolutions(kb.db, "p", True)
    (r,) = [x for x in resolutions if x.entry_id == str(entry.id)]
    assert r.observed_once is False
    index, _ = build_gate_index(resolutions)
    assert str(entry.id) in [c.resolution_id for c in index]
    assert "observed once" not in provenance_label(r)


async def test_shape_does_not_change_other_resolutions(kb: Any) -> None:
    once2 = {"capture": "autonomous", "grounding": "observed"}
    deliberate = await _store(kb, **_res())
    two = await _store(kb, **_res(provenance=once2, observed_sessions=2))
    resolutions, _ = await load_resolutions(kb.db, "p", True)
    index, _ = build_gate_index(resolutions)
    assert {c.resolution_id for c in index} == {deliberate, two}


# ─── PostToolUseFailure failure-context decisions ───────────────────────────


def _stmt(prefix: str) -> str:
    return next(s for s in database._SCHEMA_STATEMENTS if s.startswith(prefix))


_CREATE = "CREATE TABLE IF NOT EXISTS gate_decisions ("
_DROP = (
    "ALTER TABLE gate_decisions DROP CONSTRAINT IF EXISTS gate_decisions_decision_check"
)
_ADD = "ALTER TABLE gate_decisions ADD CONSTRAINT gate_decisions_decision_check"


def test_gate_decisions_constraint_statement_order() -> None:
    stmts = database._SCHEMA_STATEMENTS
    create, drop, add = (stmts.index(_stmt(p)) for p in (_CREATE, _DROP, _ADD))
    assert create < drop < add


def test_gate_decision_enum_drift() -> None:
    members = typing.get_args(models.GateDecision)
    assert tuple(members) == prevention_routes._DECISIONS
    create, add = _stmt(_CREATE), _stmt(_ADD)
    for member in members:
        assert f"'{member}'" in create
        assert f"'{member}'" in add
    marker = f"'{members[-1]}'"
    assert marker == database._GATE_DECISIONS_CHECK_MARKER


def _fc_row(n: int, decision: str = "failure_context", **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "decision_id": f"cc:s1:toolu_f{n}:{decision}",
        "session_id": "s1",
        "tool": "Bash",
        "target_class": "git push",
        "resolution_id": "kb-00001",
        "decision": decision,
        "reason_excerpt": "KB: x",
        "ts": "2026-10-07T12:00:00+00:00",
    }
    out.update(kw)
    return out


def _post(client: TestClient, *rows: dict[str, Any]) -> dict[str, Any]:
    resp = client.post("/api/kb/prevention/decisions", json={"rows": list(rows)})
    assert resp.status_code == 200
    result: dict[str, Any] = resp.json()
    return result


def test_failure_context_decisions_ingest_and_stats(local_client: TestClient) -> None:
    row = _fc_row(1)
    assert _post(local_client, row) == {"inserted": 1, "duplicates": 0}
    assert _post(local_client, row) == {"inserted": 0, "duplicates": 1}
    repeat = _fc_row(2, "failure_context_repeat")
    repeat.pop("reason_excerpt")
    assert _post(local_client, repeat)["inserted"] == 1
    error = _fc_row(
        3,
        "failure_context_error",
        resolution_id="",
        last_error_type="RuntimeError",
        reason_excerpt=None,
    )
    assert _post(local_client, error)["inserted"] == 1
    body = local_client.get("/api/kb/prevention/stats").json()
    assert body["counts"]["failure_context"] == 1
    assert body["counts"]["failure_context_repeat"] == 1
    assert body["counts"]["failure_context_error"] == 1
    assert body["by_resolution"] == []


def test_repeat_failure_context_pairs(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    rows = [
        _fc_row(
            i,
            session_id="s9",
            decision_id=f"cc:s9:t{i}:failure_context",
            resolution_id=rid,
        )
        for i, rid in enumerate(("kb-00001", "kb-00002", "kb-00003"))
    ]
    _post(local_client, *rows)
    body = local_client.get("/api/kb/prevention/stats").json()
    assert body["invariant_violations"] == {
        "over_cap_sessions": 0,
        "repeat_deny_pairs": 0,
        "repeat_failure_context_pairs": 0,
    }
    _post(
        local_client,
        _fc_row(4, session_id="s9", decision_id="cc:s9:t4:failure_context"),
    )
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        body = local_client.get("/api/kb/prevention/stats").json()
    assert body["invariant_violations"]["repeat_failure_context_pairs"] == 1
    assert body["invariant_violations"]["over_cap_sessions"] == 0
    assert body["invariant_violations"]["repeat_deny_pairs"] == 0
    assert any("gate_invariant_violation" in r.getMessage() for r in caplog.records)


async def test_expired_correction_excluded(kb: Any) -> None:
    from datetime import UTC, datetime, timedelta

    old = await kb.store(
        short_title="Old belief",
        long_title="Old",
        knowledge_details="old",
        project_ref="p",
        enrich=False,
    )
    await kb.store(
        short_title="Expired fact",
        long_title="New",
        knowledge_details="new",
        project_ref="p",
        hints={"supersedes": [old.id]},
        expires_at=datetime.now(UTC) - timedelta(days=1),
        enrich=False,
    )
    assert await load_corrections(kb.db, "p", 20) == []


_TALOS_MAP = {
    "bash": "Bash",
    "edit_file": "Edit",
    "read_file": "Read",
    "list_files": "LS",
    "run_checks": "run_checks",
    "finish": "finish",
}


async def test_prevention_tool_map_talos(
    kb: Any,
    real_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    push = await _store(kb, **_res())
    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    params = {"project": "p", "harness": "talos"}
    body = real_client.get("/api/kb/prevention", params=params).json()
    assert body["gate"]["enabled"] is True
    assert [(c["resolution_id"], c["tool"]) for c in body["index"]] == [(push, "Bash")]
    assert body["tool_map"] == _TALOS_MAP
    assert "harness=talos tool_map_len=6" in caplog.text
    caplog.clear()
    for q in ({"project": "p"}, {"project": "p", "harness": "claude-code"}):
        assert real_client.get("/api/kb/prevention", params=q).json()["tool_map"] == {}
    assert "harness=None tool_map_len=0" in caplog.text
    body = real_client.get("/api/kb/prevention", params={"harness": "talos"}).json()
    assert body["project"] == ""
    assert body["tool_map"] == _TALOS_MAP


def test_prevention_tool_map_on_inert_path(
    real_client: TestClient, fake_kb: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    monkeypatch.setattr(fake_kb, "db", Boom())
    params = {"project": "p", "harness": "talos"}
    body = real_client.get("/api/kb/prevention", params=params).json()
    assert body["gate"]["enabled"] is False
    assert body["tool_map"] == _TALOS_MAP


async def test_prevention_mapped_harness_index_without_gate_switch(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KB_SOFT_GATE_ENABLED", raising=False)
    monkeypatch.delenv("KB_SOFT_GATE_DISABLED_PROJECTS", raising=False)
    await _store(kb, **_res())
    get = real_client.get
    body = get("/api/kb/prevention", params={"project": "p", "harness": "talos"}).json()
    assert len(body["index"]) == 1
    assert body["gate"]["enabled"] is False
    for q in ({"project": "p"}, {"project": "p", "harness": "claude-code"}):
        assert get("/api/kb/prevention", params=q).json()["index"] == []
    monkeypatch.setenv("KB_SOFT_GATE_DISABLED_PROJECTS", "p")
    body = get("/api/kb/prevention", params={"project": "p", "harness": "talos"}).json()
    assert body["index"] == []


async def test_prevention_mapped_harness_excludes_observed_once(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KB_SOFT_GATE_ENABLED", raising=False)
    monkeypatch.delenv("KB_SOFT_GATE_DISABLED_PROJECTS", raising=False)
    await _store(
        kb,
        **_res(
            provenance={
                "capture": "autonomous",
                "grounding": "observed",
                "shape": 1,
                "sessions": ["s1"],
            }
        ),
    )
    body = real_client.get(
        "/api/kb/prevention", params={"project": "p", "harness": "talos"}
    ).json()
    assert body["index"] == []


def test_decisions_canonicalize_tool_only(local_client: TestClient) -> None:
    rows = [
        _row(
            1,
            decision_id="talos:s1:toolu_1:denied",
            harness="talos",
            tool="bash",
            target="ls && git push github main",
            target_class="git push",
        ),
        _row(
            2, decision_id="talos:s1:toolu_2:denied", harness="talos", tool="web_fetch"
        ),
        _row(3, decision_id="cc:s1:toolu_3:denied", tool="bash"),
    ]
    resp = local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    assert resp.json() == {"inserted": 3, "duplicates": 0}
    got = [
        tuple(r)
        for r in _query(
            "SELECT tool, target, target_class FROM gate_decisions ORDER BY id"
        )
    ]
    assert got == [
        ("Bash", "ls && git push github main", "git push"),
        ("web_fetch", "git push origin main", "git push"),
        ("bash", "git push origin main", "git push"),
    ]
