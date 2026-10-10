"""Tests for hints.resolution validation, stamping and route wiring."""

import json
import logging
import sqlite3
from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType

from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.resolution_hint import (
    ResolutionHintError,
    validate_and_stamp_resolution,
)
from tests.conftest import FakeKnowledgeBase, fake_user, make_entry
from tests.test_map_lint_routes import (
    MACHINE_EMAIL,
    _override_user,
    _seed_app_config,
    fake_machine_user,
)

CUE = {"tool": "Bash", "target_class": "git remote"}


def _v(res: Any, **kw: Any) -> dict[str, Any] | None:
    kw.setdefault("is_machine", False)
    kw.setdefault("entry_type", "factual_reference")
    return validate_and_stamp_resolution({"resolution": res}, **kw)


def _fails(
    res: Any, reason: str, contains: str = "hints.resolution", **kw: Any
) -> None:
    with pytest.raises(ResolutionHintError) as ei:
        _v(res, **kw)
    assert ei.value.reason == reason
    assert str(ei.value).startswith("hints.resolution")
    assert contains in str(ei.value)


# ─── pure function ───────────────────────────────────────────────────────────


def test_identity_without_resolution() -> None:
    assert validate_and_stamp_resolution(None, is_machine=False, entry_type="x") is None
    h = {"related_entities": ["x"]}
    out = validate_and_stamp_resolution(h, is_machine=True, entry_type="x")
    assert out is h


def test_does_not_mutate_input() -> None:
    h = {"resolution": {"corrected_fact": "x", "cue": dict(CUE)}}
    before = deepcopy(h)
    out = validate_and_stamp_resolution(h, is_machine=False, entry_type="x")
    assert h == before
    assert out is not h
    assert out is not None
    assert out["resolution"]["provenance"] == {
        "capture": "deliberate",
        "grounding": "asserted",
    }
    assert "observed_sessions" not in out["resolution"]


def test_shape_errors() -> None:
    _fails("nope", "invalid_shape", "must be an object")
    _fails(None, "invalid_shape")
    _fails(
        {"corrected_fact": "x", "corrected_facts": 1},
        "unknown_keys",
        "['corrected_facts']",
    )
    _fails(
        {"corrected_fact": "x", "cue": {**CUE, "z": 1}},
        "unknown_keys",
        "hints.resolution.cue",
    )
    _fails(
        {"corrected_fact": "x", "provenance": {"z": 1}},
        "unknown_keys",
        "hints.resolution.provenance",
    )
    _fails({}, "corrected_fact", "corrected_fact")
    _fails({"corrected_fact": 3}, "corrected_fact")
    _fails({"corrected_fact": "  "}, "corrected_fact")
    _fails({"corrected_fact": "x" * 1001}, "corrected_fact")
    assert _v({"corrected_fact": "x" * 1000}) is not None
    _fails({"corrected_fact": "x", "wrong_belief": 1}, "type_error")
    _fails({"corrected_fact": "x", "evidence": 1}, "type_error")
    for cue in (
        {},
        None,
        "x",
        {"tool": "Bash"},
        {"target_class": "x"},
        {"tool": "Bash", "target_class": ""},
    ):
        _fails(
            {"corrected_fact": "x", "cue": cue}, "type_error", "tool and target_class"
        )
    _fails({"corrected_fact": "x", "cue": {**CUE, "args_prefix": ""}}, "type_error")
    _fails({"corrected_fact": "x", "cue": {**CUE, "args_prefix": 1}}, "type_error")
    _fails({"corrected_fact": "x", "provenance": []}, "type_error")
    _fails({"corrected_fact": "x", "provenance": {"capture": "x"}}, "type_error")
    _fails({"corrected_fact": "x", "provenance": {"grounding": "x"}}, "type_error")
    _fails({"corrected_fact": "x", "provenance": {"event_id": ""}}, "type_error")
    for obs in (True, 0, "1", 1.5):
        _fails({"corrected_fact": "x", "observed_sessions": obs}, "type_error")
    _fails({"corrected_fact": "x", "scope": "team"}, "type_error")
    _fails({"corrected_fact": "x"}, "mental_map", "mental_map", entry_type="mental_map")


def test_text_stored_unstripped() -> None:
    out = _v({"corrected_fact": " x ", "wrong_belief": " w "})
    assert out is not None
    assert out["resolution"]["corrected_fact"] == " x "
    assert out["resolution"]["wrong_belief"] == " w "


@pytest.mark.parametrize("tc", ["git remote", "git push", "ls"])
def test_bash_class_fixed_point_ok(tc: str) -> None:
    assert _v({"corrected_fact": "x", "cue": {"tool": "Bash", "target_class": tc}})


@pytest.mark.parametrize(
    ("tc", "norm"),
    [("git remote add", "'git remote'"), ("sudo git push", "'git push'")],
)
def test_bash_class_not_normalized(tc: str, norm: str) -> None:
    _fails(
        {"corrected_fact": "x", "cue": {"tool": "Bash", "target_class": tc}},
        "not_normalized",
        norm,
    )


def test_non_bash_cue_unchecked_and_args_prefix_ok() -> None:
    assert _v(
        {"corrected_fact": "x", "cue": {"tool": "Edit", "target_class": "ext:py"}}
    )
    assert _v({"corrected_fact": "x", "cue": {**CUE, "args_prefix": "add"}})


def test_stamping_rules(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        out = _v(
            {"corrected_fact": "x", "provenance": {"capture": "deliberate"}},
            is_machine=True,
        )
    assert out is not None
    assert out["resolution"]["provenance"]["capture"] == "autonomous"
    assert any(r.getMessage() == "resolution_capture_forced" for r in caplog.records)
    out = _v({"corrected_fact": "x", "provenance": {"capture": "autonomous"}})
    assert out is not None
    assert out["resolution"]["provenance"]["capture"] == "autonomous"
    _fails(
        {"corrected_fact": "x", "provenance": {"grounding": "observed"}},
        "observed_needs_event_id",
    )
    out = _v(
        {
            "corrected_fact": "x",
            "provenance": {"grounding": "observed", "event_id": "e"},
        }
    )
    assert out is not None
    assert out["resolution"]["provenance"]["event_id"] == "e"


def _existing(capture: str | None) -> dict[str, Any]:
    res: dict[str, Any] = {"corrected_fact": "old"}
    if capture:
        res["provenance"] = {"capture": capture}
    return {"resolution": res}


def test_deliberate_protection() -> None:
    new = {"corrected_fact": "n"}
    # machine asserted over deliberate / missing provenance: rejected
    for ex in (_existing("deliberate"), _existing(None)):
        _fails(new, "deliberate_protected", is_machine=True, existing_hints=ex)
    # allowed combos
    assert _v(new, is_machine=False, existing_hints=_existing("autonomous"))
    assert _v(new, is_machine=False, existing_hints=_existing("deliberate"))
    assert _v(new, is_machine=True, existing_hints=_existing("autonomous"))
    assert _v(
        {**new, "provenance": {"grounding": "observed", "event_id": "e"}},
        is_machine=True,
        existing_hints=_existing("deliberate"),
    )
    assert _v(new, is_machine=True, existing_hints={"other": 1})
    assert _v(new, is_machine=True, existing_hints=None)


# ─── routes ──────────────────────────────────────────────────────────────────

BASE = {
    "short_title": "Test entry",
    "long_title": "A full test entry",
    "knowledge_details": "Some valid knowledge details.",
}
RES = {"corrected_fact": "x", "cue": dict(CUE), "scope": "global"}


def _post(client: TestClient, hints: Any, **extra: Any) -> Any:
    return client.post("/api/kb/store", json={**BASE, "hints": hints, **extra})


@pytest.fixture
def user(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_user


@pytest.fixture
def machine(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})


def _kb() -> FakeKnowledgeBase:
    kb: FakeKnowledgeBase = app.state.kb
    return kb


def _route_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "resolution-route" in r.getMessage()
    ]


@pytest.mark.usefixtures("user")
def test_create_stamps_deliberate(client: TestClient) -> None:
    resp = _post(client, {"resolution": RES})
    assert resp.status_code == 200
    stored = _kb().store_calls[-1]["hints"]["resolution"]
    assert stored["provenance"] == {"capture": "deliberate", "grounding": "asserted"}
    assert "observed_sessions" not in stored


@pytest.mark.usefixtures("machine")
def test_create_machine_forced_autonomous(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        resp = _post(
            client,
            {"resolution": {**RES, "provenance": {"capture": "deliberate"}}},
        )
    assert resp.status_code == 200
    stored = _kb().store_calls[-1]["hints"]["resolution"]
    assert stored["provenance"]["capture"] == "autonomous"
    assert any(r.getMessage() == "resolution_capture_forced" for r in caplog.records)


@pytest.mark.usefixtures("user")
def test_create_rejections(client: TestClient) -> None:
    r = _post(client, {"resolution": {"corrected_fact": ""}})
    assert r.status_code == 422 and "corrected_fact" in r.json()["detail"]
    assert _post(client, {"resolution": None}).status_code == 422
    r = _post(client, {"resolution": {"corrected_facts": "x"}})
    assert r.status_code == 422 and "['corrected_facts']" in r.json()["detail"]
    r = _post(
        client,
        {
            "resolution": {
                "corrected_fact": "x",
                "provenance": {"grounding": "observed"},
            }
        },
    )
    assert r.status_code == 422
    r = _post(
        client,
        {
            "resolution": {
                "corrected_fact": "x",
                "provenance": {"grounding": "observed", "event_id": "evt-1"},
            }
        },
    )
    assert r.status_code == 200
    assert _kb().store_calls == [_kb().store_calls[-1]]


@pytest.mark.usefixtures("user")
def test_mental_map_rejected(client: TestClient) -> None:
    r = _post(
        client,
        {"resolution": RES, "related_entities": ["kb-00001"]},
        entry_type="mental_map",
    )
    assert r.status_code == 422
    assert "mental_map" in r.json()["detail"]
    batch = client.post(
        "/api/kb/store_batch",
        json={
            "entries": [
                {**BASE, "entry_type": "mental_map", "hints": {"resolution": RES}}
            ]
        },
    )
    assert batch.status_code == 422
    assert batch.json()["detail"].startswith("entry 0: ")


@pytest.mark.usefixtures("user")
def test_batch_all_or_nothing(client: TestClient) -> None:
    resp = client.post(
        "/api/kb/store_batch",
        json={
            "entries": [
                {**BASE, "hints": {"resolution": RES}},
                {**BASE, "hints": {"resolution": {"corrected_fact": ""}}},
            ]
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("entry 1: ")
    assert _kb().store_batch_calls == []


@pytest.mark.usefixtures("user")
def test_batch_stamps_and_keeps_supersedes(client: TestClient) -> None:
    resp = client.post(
        "/api/kb/store_batch",
        json={
            "entries": [
                {
                    **BASE,
                    "hints": {"resolution": RES, "supersedes": ["kb-00002"]},
                }
            ]
        },
    )
    assert resp.status_code == 200
    hints = _kb().store_batch_calls[-1][0][0]["hints"]
    assert hints["resolution"]["provenance"]["capture"] == "deliberate"
    assert hints["supersedes"] == ["kb-00002"]


@pytest.mark.usefixtures("user")
def test_create_keeps_supersedes(client: TestClient) -> None:
    resp = _post(client, {"resolution": RES, "supersedes": ["kb-00002"]})
    assert resp.status_code == 200
    hints = _kb().store_calls[-1]["hints"]
    assert hints["supersedes"] == ["kb-00002"] and "resolution" in hints


@pytest.mark.usefixtures("user")
def test_malformed_beats_near_duplicate(client: TestClient) -> None:
    from kb_core.near_duplicates import NearDuplicateCandidate, NearDuplicateCheck

    kb = _kb()
    kb.near_duplicate_check = NearDuplicateCheck(
        status="checked",
        candidates=(
            NearDuplicateCandidate(
                id="kb-00009",
                short_title="t",
                entry_type="factual_reference",
                similarity=0.99,
                updated_at=None,
            ),
        ),
        top_similarity=0.99,
    )
    r = _post(client, {"resolution": {"corrected_fact": ""}})
    assert r.status_code == 422


def _put_existing(entry_id: str, hints: dict[str, Any]) -> None:
    e = make_entry(entry_id)
    e.hints = hints
    _kb().entries[entry_id] = e


UPD = {"update_entry_id": "kb-00007", "change_reason": "because"}


@pytest.mark.usefixtures("machine")
def test_update_deliberate_protected_and_forced(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _put_existing(
        "kb-00007",
        {
            "resolution": {
                "corrected_fact": "o",
                "provenance": {"capture": "deliberate"},
            }
        },
    )
    with caplog.at_level(logging.INFO):
        r = client.post("/api/kb/store", json={**UPD, "hints": {"resolution": RES}})
    assert r.status_code == 422
    lines = _route_lines(caplog)
    assert len(lines) == 1 and "reason=deliberate_protected" in lines[0]
    assert _kb().update_calls == []


@pytest.mark.usefixtures("machine")
def test_update_machine_over_autonomous(client: TestClient) -> None:
    _put_existing(
        "kb-00007",
        {
            "resolution": {
                "corrected_fact": "o",
                "provenance": {"capture": "autonomous"},
            }
        },
    )
    res = {**RES, "provenance": {"capture": "deliberate"}}
    r = client.post("/api/kb/store", json={**UPD, "hints": {"resolution": res}})
    assert r.status_code == 200
    stored = _kb().update_calls[-1][1]["hints"]["resolution"]
    assert stored["provenance"]["capture"] == "autonomous"


@pytest.mark.usefixtures("user")
def test_update_without_resolution_passthrough(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _put_existing("kb-00007", {"resolution": {"corrected_fact": "o"}})
    with caplog.at_level(logging.INFO):
        r = client.post("/api/kb/store", json={**UPD, "hints": {"tags_note": 1}})
    assert r.status_code == 200
    assert _kb().update_calls[-1][1]["hints"] == {"tags_note": 1}
    assert _route_lines(caplog) == []


@pytest.mark.usefixtures("user")
def test_create_without_resolution_untouched(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    hints = {"related_entities": ["x"]}
    with caplog.at_level(logging.INFO):
        assert _post(client, hints).status_code == 200
    assert _kb().store_calls[-1]["hints"] == hints
    assert _route_lines(caplog) == []


@pytest.mark.usefixtures("user")
def test_route_trail_lines(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        _post(client, {"resolution": RES})
    lines = _route_lines(caplog)
    assert len(lines) == 1
    assert "outcome=accepted reason=ok" in lines[0] and "op=store" in lines[0]
    caplog.clear()
    bad = {
        "corrected_fact": "x",
        "cue": {"tool": "Bash", "target_class": "git remote add"},
    }
    with caplog.at_level(logging.INFO):
        _post(client, {"resolution": bad})
    lines = _route_lines(caplog)
    assert len(lines) == 1 and "reason=not_normalized" in lines[0]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        client.post(
            "/api/kb/store_batch",
            json={"entries": [{**BASE, "hints": {"resolution": bad}}]},
        )
    lines = _route_lines(caplog)
    assert len(lines) == 1 and "op=store_batch outcome=rejected" in lines[0]


@pytest.mark.usefixtures("user")
def test_supersede_drop_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _put_existing("kb-00002", {"resolution": {"corrected_fact": "o"}})
    with caplog.at_level(logging.WARNING):
        r = _post(client, {"supersedes": ["kb-00002"]})
    assert r.status_code == 200
    assert any(
        "resolution_dropped_by_supersede" in rec.getMessage()
        and "kb-00002" in rec.getMessage()
        for rec in caplog.records
    )


# ─── round trip with the reader ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_round_trip_load_resolutions() -> None:
    from kb_service.prevention import load_resolutions

    stamped = validate_and_stamp_resolution(
        {
            "resolution": {
                "corrected_fact": "use the script",
                "cue": {**CUE, "args_prefix": "add"},
                "provenance": {"grounding": "observed", "event_id": "evt-1"},
                "scope": "global",
            }
        },
        is_machine=True,
        entry_type=EntryType.PATTERN_CONVENTION.value,
    )
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE knowledge_entries (id TEXT, updated_at TEXT, hints TEXT,"
        " project_ref TEXT, is_active INTEGER, superseded_by TEXT, entry_type TEXT,"
        " expires_at TEXT)"
    )
    conn.execute(
        "INSERT INTO knowledge_entries VALUES (?,?,?,?,?,?,?,?)",
        (
            "kb-00001",
            "2026-01-01",
            json.dumps(stamped),
            "p",
            1,
            None,
            "pattern_convention",
            None,
        ),
    )

    class _Db:
        async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
            cur = conn.execute(sql, params)

            class _C:
                async def fetchall(self_inner) -> list[Any]:  # noqa: N805
                    return cur.fetchall()

            return _C()

    resolutions, stats = await load_resolutions(
        _Db(), "other", include_observed_once=True
    )
    assert stats.skipped_malformed == 0
    assert len(resolutions) == 1
    assert resolutions[0].grounding == "observed"
    assert resolutions[0].cue_args_prefix == "add"
