"""Prevention channels: helpers on a real SQLite KB, routes, decisions, stats."""

import logging
import sqlite3
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core import create_sqlite

import kb_service.database as database
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import SliceItem
from kb_service.prevention import (
    INDEX_CAP,
    Correction,
    Resolution,
    build_gate_index,
    build_slice,
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
    bare = await _store(kb, **_res())
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
        capture=None,
        grounding=None,
        observed_sessions=1,
        observed_once=False,
    )


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
    r = Resolution(**{**_resolution(1).__dict__, "capture": "deliberate"})
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
    assert body["gate"] == {"enabled": False, "shadow": True, "max_denies": 2}
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


def test_decisions_invariant_warning(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    rows = [_row(i, resolution_id=f"kb-0000{i}") for i in range(3)]
    with caplog.at_level(logging.WARNING, logger="kb_service.routes.prevention_routes"):
        local_client.post("/api/kb/prevention/decisions", json={"rows": rows})
    assert any("gate_invariant_violation" in r.getMessage() for r in caplog.records)


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
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
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
