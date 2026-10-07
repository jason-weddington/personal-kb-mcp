"""Hermetic tests for the near-duplicate guard on kb_store create."""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType
from kb_core.near_duplicates import NearDuplicateCandidate, NearDuplicateCheck

from kb_service.auth import get_current_user
from kb_service.config import NEAR_DUPLICATE_FLOOR_DEFAULT, get_near_duplicate_floor
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user, make_entry

A = "kb-00010"
B = "kb-00011"
MARKER = "near-duplicate-guard op="

_BODY: dict[str, Any] = {
    "short_title": "New fact",
    "long_title": "A brand new fact",
    "knowledge_details": "Details of the new fact.",
    "project_ref": "proj",
}


def _cand(entry_id: str, sim: float, title: str = "Existing") -> NearDuplicateCandidate:
    return NearDuplicateCandidate(
        id=entry_id,
        short_title=title,
        entry_type="factual_reference",
        similarity=sim,
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _setup(
    client: TestClient, *cands: NearDuplicateCandidate, status: Any = "checked"
) -> FakeKnowledgeBase:
    app.dependency_overrides[get_current_user] = fake_user
    kb: FakeKnowledgeBase = app.state.kb
    kb.near_duplicate_check = NearDuplicateCheck(
        status=status,
        candidates=tuple(cands),
        top_similarity=max((c.similarity for c in cands), default=None),
        raw_hits=3,
        eligible_count=len(cands),
        embed_ms=4,
        search_ms=5,
    )
    for c in cands:
        kb.entries[c.id] = make_entry(c.id)
    return kb


def _guard_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if MARKER in r.getMessage()]


def _expected_message(
    floor: float, project: str, cands: list[NearDuplicateCandidate], prefix: str = ""
) -> str:
    return (
        prefix + f"near-duplicate: this new entry is at or above cosine {floor:.2f} "
        f"to existing entries in project {project}: "
        + "; ".join(f'{c.id} "{c.short_title}" ({c.similarity:.3f})' for c in cands)
        + ". Cover EVERY listed id with one of: update_entry_id=<id> (same fact: "
        "update that entry, with change_reason); supersedes=[<id>] (the new entry "
        "replaces it; older clients: hints={'supersedes': [<id>]}); "
        "distinct_from=[<id>] (genuinely different facts; older clients: "
        "hints={'distinct_from': [<id>]})."
    )


# ── config ───────────────────────────────────────────────────────────────


def test_floor_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_NEAR_DUPLICATE_FLOOR", "0.9")
    assert get_near_duplicate_floor() == 0.9
    monkeypatch.delenv("KB_NEAR_DUPLICATE_FLOOR")
    assert get_near_duplicate_floor() == NEAR_DUPLICATE_FLOOR_DEFAULT == 0.88
    monkeypatch.setenv("KB_NEAR_DUPLICATE_FLOOR", "abc")
    with pytest.raises(ValueError, match="KB_NEAR_DUPLICATE_FLOOR"):
        get_near_duplicate_floor()


# ── store: create ────────────────────────────────────────────────────────


def test_a_conflict_409(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    cand = _cand(A, 0.93)
    kb = _setup(client, cand)
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json=_BODY)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "near_duplicate"
    assert detail["candidates"][0] == {
        "id": A,
        "short_title": "Existing",
        "entry_type": "factual_reference",
        "similarity": 0.93,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    assert detail["message"] == _expected_message(0.88, "proj", [cand])
    assert "hints" in detail["message"]
    assert kb.store_calls == []
    recs = _guard_records(caplog)
    assert len(recs) == 1 and "outcome=conflict" in recs[0].getMessage()
    assert len(kb.audit_events) == 1
    ev = kb.audit_events[0]
    assert ev[0] == "near_duplicate_checked" and ev[1] is None


def test_b_distinct_from_resolves(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _setup(client, _cand(A, 0.93))
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": [A]})
    assert resp.status_code == 200
    assert kb.store_calls[0]["hints"]["distinct_from"] == [A]
    recs = _guard_records(caplog)
    assert len(recs) == 1
    msg = recs[0].getMessage()
    assert "outcome=resolved" in msg and "{'kb-00010': 'distinct_from'}" in msg
    assert "near-duplicate-guard-stored" in caplog.text
    assert kb.audit_events[0][1] == "kb-00001"
    assert kb.audit_events[0][3]["resolved_by"] == {A: "distinct_from"}


def test_b2_hints_none(client: TestClient) -> None:
    kb = _setup(client, _cand(A, 0.93))
    resp = client.post(
        "/api/kb/store", json={**_BODY, "hints": None, "distinct_from": [A]}
    )
    assert resp.status_code == 200
    assert kb.store_calls[0]["hints"] == {"distinct_from": [A]}


def test_b3_unused_distinct_from(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _setup(client)
    kb.entries[A] = make_entry(A)
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": [A]})
    assert resp.status_code == 200
    msg = _guard_records(caplog)[0].getMessage()
    assert "outcome=clear" in msg and "distinct_from_unused=['kb-00010']" in msg


def test_b4_retry_same_text_sha(client: TestClient) -> None:
    kb = _setup(client, _cand(A, 0.93))
    assert client.post("/api/kb/store", json=_BODY).status_code == 409
    assert (
        client.post("/api/kb/store", json={**_BODY, "distinct_from": [A]}).status_code
        == 200
    )
    assert len(kb.audit_events) == 2
    assert kb.audit_events[0][3]["text_sha"] == kb.audit_events[1][3]["text_sha"]


def test_c_supersedes_resolves(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _setup(client, _cand(A, 0.93))
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json={**_BODY, "supersedes": [A]})
    assert resp.status_code == 200
    assert "distinct_from" not in kb.store_calls[0]["hints"]
    assert kb.audit_events[0][3]["resolved_by"] == {A: "supersedes"}


def test_d_partial_cover_lists_only_uncovered(client: TestClient) -> None:
    _setup(client, _cand(A, 0.95), _cand(B, 0.91, "Other"))
    resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": [A]})
    assert resp.status_code == 409
    assert [c["id"] for c in resp.json()["detail"]["candidates"]] == [B]


def test_d2_partial_cover_records_resolved_by(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _setup(client, _cand(A, 0.95), _cand(B, 0.91, "Other"))
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json={**_BODY, "supersedes": [A]})
    assert resp.status_code == 409
    assert [c["id"] for c in resp.json()["detail"]["candidates"]] == [B]
    recs = _guard_records(caplog)
    assert len(recs) == 1
    assert f"resolved_by={{'{A}': 'supersedes'}}" in recs[0].getMessage()
    assert kb.audit_events[0][3]["resolved_by"] == {A: "supersedes"}


def test_e_legacy_scalar_hint(client: TestClient) -> None:
    kb = _setup(client, _cand(A, 0.93))
    resp = client.post("/api/kb/store", json={**_BODY, "hints": {"distinct_from": A}})
    assert resp.status_code == 200
    assert kb.store_calls[0]["hints"]["distinct_from"] == [A]


def test_f_mental_map_exempt(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _setup(client, _cand(A, 0.99))
    body = {
        **_BODY,
        "entry_type": EntryType.MENTAL_MAP.value,
        "knowledge_details": "Map pointing at kb-00010.",
    }
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json=body)
    assert resp.status_code == 200
    assert kb.find_near_duplicates_calls == []
    assert "outcome=exempt_mental_map" in _guard_records(caplog)[0].getMessage()


def test_g_no_project_skipped(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _setup(client, _cand(A, 0.99))
    body = {k: v for k, v in _BODY.items() if k != "project_ref"}
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json=body)
    assert resp.status_code == 200
    assert kb.find_near_duplicates_calls == []
    assert "outcome=skipped_no_project" in _guard_records(caplog)[0].getMessage()


@pytest.mark.parametrize("status", ["embedder_unavailable", "search_failed"])
def test_h_fail_open(
    client: TestClient, caplog: pytest.LogCaptureFixture, status: str
) -> None:
    _setup(client, status=status)
    with caplog.at_level(logging.INFO):
        resp = client.post("/api/kb/store", json=_BODY)
    assert resp.status_code == 200
    recs = _guard_records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.WARNING
    assert f"outcome={status}" in recs[0].getMessage()


def test_i_missing_distinct_from(client: TestClient) -> None:
    kb = _setup(client)
    resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": ["kb-99999"]})
    assert resp.status_code == 422
    assert "distinct_from rejected: kb-99999 not found" in resp.json()["detail"]
    assert kb.store_calls == []


def test_j_both_superseded_and_distinct(client: TestClient) -> None:
    _setup(client, _cand(A, 0.93))
    resp = client.post(
        "/api/kb/store", json={**_BODY, "supersedes": [A], "distinct_from": [A]}
    )
    assert resp.status_code == 422
    assert "cannot be both superseded and distinct" in resp.json()["detail"]


def test_k_update_rejects_distinct_from(client: TestClient) -> None:
    kb = _setup(client)
    body = {"update_entry_id": A, "change_reason": "x", "distinct_from": [B]}
    resp = client.post("/api/kb/store", json=body)
    assert resp.status_code == 422
    assert "distinct_from applies only when creating" in resp.json()["detail"]
    body2 = {
        "update_entry_id": A,
        "change_reason": "x",
        "hints": {"distinct_from": "kb-00001"},
    }
    resp = client.post("/api/kb/store", json=body2)
    assert resp.status_code == 422
    assert kb.update_calls == []


def test_m_floor_default_and_env(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = _setup(client)
    client.post("/api/kb/store", json=_BODY)
    assert kb.find_near_duplicates_calls[-1]["floor"] == 0.88
    monkeypatch.setenv("KB_NEAR_DUPLICATE_FLOOR", "0.9")
    client.post("/api/kb/store", json=_BODY)
    assert kb.find_near_duplicates_calls[-1]["floor"] == 0.9


def test_n_texts_passed(client: TestClient) -> None:
    kb = _setup(client)
    client.post("/api/kb/store", json=_BODY)
    call = kb.find_near_duplicates_calls[0]
    assert call["short_title"] == _BODY["short_title"]
    assert call["long_title"] == _BODY["long_title"]
    assert call["knowledge_details"] == _BODY["knowledge_details"]
    assert call["project_ref"] == "proj"


def test_o_invalid_id(client: TestClient) -> None:
    _setup(client)
    resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": ["kb-1"]})
    assert resp.status_code == 422
    assert "kb-1 is not a valid entry id" in resp.json()["detail"]


def test_p_inactive(client: TestClient) -> None:
    kb = _setup(client)
    kb.entries[A] = make_entry(A).model_copy(update={"is_active": False})
    resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": [A]})
    assert resp.status_code == 422
    assert f"{A} is inactive" in resp.json()["detail"]


def test_q_mental_map_target(client: TestClient) -> None:
    kb = _setup(client)
    kb.entries[A] = make_entry(A).model_copy(
        update={"entry_type": EntryType.MENTAL_MAP}
    )
    resp = client.post("/api/kb/store", json={**_BODY, "distinct_from": [A]})
    assert resp.status_code == 422
    assert f"{A} is a mental_map" in resp.json()["detail"]


def test_r_non_str_hint(client: TestClient) -> None:
    _setup(client)
    resp = client.post("/api/kb/store", json={**_BODY, "hints": {"distinct_from": [5]}})
    assert resp.status_code == 422
    assert "must be a kb-id string or a list" in resp.json()["detail"]


def test_s_log_numeric_fields(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _setup(client, _cand(A, 0.93))
    with caplog.at_level(logging.INFO):
        client.post("/api/kb/store", json=_BODY)
    msg = _guard_records(caplog)[0].getMessage()
    for field in (
        "raw_hits=3",
        "eligible=1",
        "embed_ms=4",
        "search_ms=5",
        "floor=0.8800",
    ):
        assert field in msg


# ── store_batch ──────────────────────────────────────────────────────────


_BATCH = {
    "short_title": "Batch",
    "long_title": "Batch long",
    "knowledge_details": "Batch details.",
    "project_ref": "proj",
}


def test_l_batch_conflict(client: TestClient) -> None:
    kb = _setup(client)
    # entry 0 clear, entry 1 conflicts: the fake returns one check for all calls,
    # so make it conflict only for the second call.
    cand = _cand(A, 0.93)
    clear = kb.near_duplicate_check
    conflict = NearDuplicateCheck(
        status="checked", candidates=(cand,), top_similarity=0.93
    )
    checks = iter([clear, conflict])
    original = kb.find_near_duplicates

    async def scripted(**kw: Any) -> NearDuplicateCheck:
        await original(**kw)
        return next(checks)

    kb.find_near_duplicates = scripted  # type: ignore[method-assign]
    resp = client.post("/api/kb/store_batch", json={"entries": [_BATCH, _BATCH]})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["entry_index"] == 1
    assert detail["message"] == _expected_message(
        0.88, "proj", [cand], prefix="entry 1: "
    )
    assert kb.store_batch_calls == []


def test_l2_batch_distinct_from(client: TestClient) -> None:
    kb = _setup(client, _cand(A, 0.93))
    resp = client.post(
        "/api/kb/store_batch",
        json={"entries": [{**_BATCH, "distinct_from": [A]}]},
    )
    assert resp.status_code == 200
    assert kb.store_batch_calls[0][0][0]["hints"]["distinct_from"] == [A]
    assert kb.audit_events and kb.audit_events[0][1] == "kb-00001"


def test_batch_distinct_from_invalid(client: TestClient) -> None:
    _setup(client)
    resp = client.post(
        "/api/kb/store_batch",
        json={"entries": [{**_BATCH, "distinct_from": ["kb-99999"]}]},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("entry 0: distinct_from rejected")
