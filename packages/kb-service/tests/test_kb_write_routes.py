"""Hermetic tests for the KB write endpoints.

Covers: POST /api/kb/store, /store_batch, /entries/{id}/deactivate,
/entries/{id}/reactivate, /bulk_update, /feedback.

All tests are hermetic (no live Postgres / Ollama / network).  The ``client``
fixture from conftest.py already patches ``create_postgres`` / ``init_db`` /
``get_db``, so these tests only need to override the auth dependency and
(optionally) inject errors or monkeypatch module-level names.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry

import kb_service.routes.kb_write_routes as kb_write_routes_module
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_admin_user, fake_user, make_entry

# ─── helpers ─────────────────────────────────────────────────────────────────

_STORE_VALID = {
    "short_title": "Test entry",
    "long_title": "A full test entry",
    "knowledge_details": "Some valid knowledge details.",
}

_BATCH_ENTRY: dict[str, Any] = {
    "short_title": "Batch entry",
    "long_title": "A batch knowledge entry",
    "knowledge_details": "Batch knowledge details.",
}


def _authed(client: TestClient, *, admin: bool = False) -> None:
    """Override the auth dependency on the global app."""
    app.dependency_overrides[get_current_user] = fake_admin_user if admin else fake_user


# ─── POST /api/kb/store — create path ────────────────────────────────────────


def test_store_create_happy_path(client: TestClient) -> None:
    """Happy path: create stores via facade, returns action=created + entry."""
    _authed(client)
    resp = client.post("/api/kb/store", json=_STORE_VALID)
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "created"
    assert "entry" in body
    assert "id" in body["entry"]

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.store_calls, "store() was not called"
    call = kb.store_calls[-1]
    assert call["contributor"] == "tester@example.com"
    assert call["short_title"] == "Test entry"
    # team is None because the StatefulFakeDbPool has no team stored
    assert call["team"] is None


def test_store_create_requires_auth(client: TestClient) -> None:
    """POST /api/kb/store with no credentials returns 401."""
    resp = client.post("/api/kb/store", json=_STORE_VALID)
    assert resp.status_code == 401


def test_store_create_empty_required_field(client: TestClient) -> None:
    """Missing required field on create path → 422 with string detail."""
    _authed(client)
    resp = client.post(
        "/api/kb/store",
        json={"short_title": "", "long_title": "x", "knowledge_details": "x"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "required" in detail.lower()


def test_store_create_secret_scan_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkeypatched detect_secrets_in_content returning findings → 422."""
    _authed(client)
    monkeypatch.setattr(
        kb_write_routes_module,
        "detect_secrets_in_content",
        lambda _content: ["Secret Keyword"],
    )
    resp = client.post("/api/kb/store", json=_STORE_VALID)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "Secret Keyword" in detail


def test_store_create_ttl_zero_422(client: TestClient) -> None:
    """TTL of '0d' raises ValueError from compute_expires_at → 422."""
    _authed(client)
    resp = client.post("/api/kb/store", json={**_STORE_VALID, "ttl": "0d"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)


def test_store_create_orphan_mental_map_422(client: TestClient) -> None:
    """Creating a mental_map entry with no pointers → 422 with pinned message."""
    _authed(client)
    resp = client.post(
        "/api/kb/store",
        json={
            **_STORE_VALID,
            "knowledge_details": "No pointers here.",
            "entry_type": "mental_map",
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "A mental_map entry requires at least one outbound pointer" in detail


def test_store_create_mental_map_with_kb_ref_passes(client: TestClient) -> None:
    """mental_map with a kb-XXXXX ref in knowledge_details is accepted."""
    _authed(client)
    resp = client.post(
        "/api/kb/store",
        json={
            **_STORE_VALID,
            "knowledge_details": "See kb-00042 for details.",
            "entry_type": "mental_map",
        },
    )
    assert resp.status_code == 200


# ─── POST /api/kb/store — update path ────────────────────────────────────────


def test_store_update_happy_path(client: TestClient) -> None:
    """Update path calls kb.update, returns action=updated + entry."""
    _authed(client)
    kb_setup: FakeKnowledgeBase = app.state.kb
    kb_setup.entries["kb-00001"] = make_entry("kb-00001")
    resp = client.post(
        "/api/kb/store",
        json={**_STORE_VALID, "update_entry_id": "kb-00001"},
    )
    assert resp.status_code == 200
    assert resp.json()["action"] == "updated"

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.update_calls
    entry_id, kwargs = kb.update_calls[-1]
    assert entry_id == "kb-00001"
    assert kwargs["updated_by"] == "tester@example.com"


def test_store_update_not_found_404(client: TestClient) -> None:
    """ValueError containing 'not found' from kb.update → 404."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb._update_raises = ValueError("Entry kb-99999 not found")
    resp = client.post(
        "/api/kb/store",
        json={**_STORE_VALID, "update_entry_id": "kb-99999"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_store_update_inactive_409(client: TestClient) -> None:
    """ValueError 'is inactive and cannot be updated' → 409."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    # The pre-update orphan check now fetches the existing entry first — it
    # must exist for kb.update to even be reached.
    kb.entries["kb-00001"] = make_entry("kb-00001")
    kb._update_raises = ValueError("Entry kb-00001 is inactive and cannot be updated")
    resp = client.post(
        "/api/kb/store",
        json={**_STORE_VALID, "update_entry_id": "kb-00001"},
    )
    assert resp.status_code == 409
    assert "inactive" in resp.json()["detail"]


def test_store_update_confidence_level_omitted_passes_none(
    client: TestClient,
) -> None:
    """Omitting confidence_level on update passes None to kb.update (no-change).

    This locks in the deliberate divergence from the MCP tool, which always
    resets to 0.9.  The service treats an omitted field as no-change.
    """
    _authed(client)
    kb_setup: FakeKnowledgeBase = app.state.kb
    kb_setup.entries["kb-00001"] = make_entry("kb-00001")
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001"},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.update_calls
    _entry_id, kwargs = kb.update_calls[-1]
    assert kwargs["confidence_level"] is None


def _map_entry(
    entry_id: str = "kb-00001",
    knowledge_details: str = "See kb-00042 for details.",
    hints: dict[str, Any] | None = None,
) -> KnowledgeEntry:
    """A stored mental_map entry with an outbound pointer, for update tests."""
    return KnowledgeEntry(
        id=entry_id,
        short_title="A map",
        long_title="A mental map entry",
        knowledge_details=knowledge_details,
        entry_type=EntryType.MENTAL_MAP,
        hints=hints or {},
    )


def test_store_update_metadata_only_map_untouched_succeeds(client: TestClient) -> None:
    """THE TRAP: a tags-only update to a map whose STORED body has a pointer
    must succeed — knowledge_details is None on the request (metadata-only),
    so the orphan check must fall back to the entry's current stored body
    rather than 422 on the absent request body."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = _map_entry("kb-00001")
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001", "tags": ["reorg"]},
    )
    assert resp.status_code == 200
    assert kb.update_calls
    _entry_id, kwargs = kb.update_calls[-1]
    assert kwargs["knowledge_details"] is None
    assert kwargs["tags"] == ["reorg"]


def test_store_update_hints_only_map_untouched_succeeds(client: TestClient) -> None:
    """A hints-only update (no knowledge_details) to a map whose stored body
    has a pointer must also succeed — same trap, different field."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = _map_entry("kb-00001")
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001", "hints": {"tag": "reorg"}},
    )
    assert resp.status_code == 200


def test_store_update_map_orphaned_via_effective_body_422(client: TestClient) -> None:
    """Clearing a map's knowledge_details pointer with no replacement hint
    must 422 — the EFFECTIVE post-update body (request details, since present)
    has no pointer."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = _map_entry("kb-00001")
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001", "knowledge_details": "No pointers now."},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "A mental_map entry requires at least one outbound pointer" in detail
    assert not kb.update_calls


def test_store_update_map_becomes_orphan_when_stored_body_has_no_pointer_422(
    client: TestClient,
) -> None:
    """A metadata-only update on a map whose STORED body already has zero
    pointers (pre-existing orphan, or entry_type flipped to mental_map on
    this same request with no pointer) must still 422."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = _map_entry(
        "kb-00001", knowledge_details="No pointers stored."
    )
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001", "tags": ["x"]},
    )
    assert resp.status_code == 422
    assert not kb.update_calls


def test_store_update_map_pointer_via_merged_hint_succeeds(client: TestClient) -> None:
    """A request hint merges with the entry's EXISTING hints (not replaces
    them) — mirrors kb-core's update_entry merge. A request-supplied
    supersedes hint satisfies the check even though the stored body/hints
    have no pointer of their own."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = _map_entry(
        "kb-00001", knowledge_details="No pointers stored.", hints={}
    )
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001", "hints": {"supersedes": "kb-00099"}},
    )
    assert resp.status_code == 200


def test_store_update_entry_type_flip_to_map_checks_effective_body(
    client: TestClient,
) -> None:
    """Flipping entry_type to mental_map on an entry whose stored body has no
    pointer, with no new pointer supplied, must 422 (effective type is the
    REQUEST's entry_type when present)."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = make_entry("kb-00001")  # factual_reference, no pointer
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": "kb-00001", "entry_type": "mental_map"},
    )
    assert resp.status_code == 422


def test_store_update_not_found_uses_direct_404(client: TestClient) -> None:
    """Update on an entry the orphan-check pre-fetch can't find → 404, without
    ever reaching kb.update."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    resp = client.post(
        "/api/kb/store",
        json={**_STORE_VALID, "update_entry_id": "kb-99999"},
    )
    assert resp.status_code == 404
    assert not kb.update_calls


# ─── POST /api/kb/store_batch ────────────────────────────────────────────────


def test_store_batch_happy_path(client: TestClient) -> None:
    """Batch create calls facade with contributor/team injected, returns shape."""
    _authed(client)
    resp = client.post(
        "/api/kb/store_batch",
        json={"entries": [_BATCH_ENTRY, {**_BATCH_ENTRY, "short_title": "B2"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["requested"] == 2
    assert isinstance(body["created"], list)
    assert len(body["created"]) == 2

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.store_batch_calls
    entry_dicts, enrich = kb.store_batch_calls[-1]
    assert enrich is True
    for d in entry_dicts:
        assert d["contributor"] == "tester@example.com"
        assert d["team"] is None


def test_store_batch_requires_auth(client: TestClient) -> None:
    """POST /api/kb/store_batch with no credentials returns 401."""
    resp = client.post("/api/kb/store_batch", json={"entries": [_BATCH_ENTRY]})
    assert resp.status_code == 401


def test_store_batch_too_many_entries_422(client: TestClient) -> None:
    """11 entries in store_batch → 422 (Pydantic max_length=10 validation)."""
    _authed(client)
    entries = [{**_BATCH_ENTRY, "short_title": f"E{i}"} for i in range(11)]
    resp = client.post("/api/kb/store_batch", json={"entries": entries})
    assert resp.status_code == 422


def test_store_batch_secret_scan_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secret in batch entry → whole-batch 422 with entry index in detail."""
    _authed(client)
    monkeypatch.setattr(
        kb_write_routes_module,
        "detect_secrets_in_content",
        lambda _content: ["Secret Keyword"],
    )
    resp = client.post(
        "/api/kb/store_batch",
        json={"entries": [_BATCH_ENTRY, {**_BATCH_ENTRY, "short_title": "B2"}]},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    # First entry (index 0) triggers the rejection
    assert "entry 0" in detail
    assert "Secret Keyword" in detail


def test_store_batch_orphan_mental_map_422(client: TestClient) -> None:
    """store_batch with orphan mental_map entry → whole-batch 422 with index."""
    _authed(client)
    resp = client.post(
        "/api/kb/store_batch",
        json={
            "entries": [
                {
                    **_BATCH_ENTRY,
                    "knowledge_details": "No pointers.",
                    "entry_type": "mental_map",
                }
            ]
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "entry 0" in detail
    assert "mental_map" in detail


# ─── POST /api/kb/entries/{id}/deactivate ────────────────────────────────────


def test_deactivate_happy_path(client: TestClient) -> None:
    """Deactivate calls kb.deactivate, then records graph-edge cleanup + commit."""
    _authed(client)
    resp = client.post("/api/kb/entries/kb-00001/deactivate")
    assert resp.status_code == 200
    body = resp.json()
    assert "entry" in body

    kb: FakeKnowledgeBase = app.state.kb
    # facade call
    assert kb.deactivate_calls
    entry_id, contributor = kb.deactivate_calls[-1]
    assert entry_id == "kb-00001"
    assert contributor == "tester@example.com"
    # graph cleanup — scoped to exclude mental_map sources (defense in depth,
    # see the guard test below for the primary block on map entries).
    assert kb.db.calls, "db.execute was not called"
    sql, params = kb.db.calls[-1]
    assert "DELETE FROM graph_edges WHERE source = ?" in sql
    assert "mental_map" in sql
    assert params == ("kb-00001",)
    assert kb.db.committed >= 1


def test_deactivate_requires_auth(client: TestClient) -> None:
    """POST /api/kb/entries/{id}/deactivate with no credentials returns 401."""
    resp = client.post("/api/kb/entries/kb-00001/deactivate")
    assert resp.status_code == 401


def test_deactivate_not_found_404(client: TestClient) -> None:
    """ValueError 'not found' from kb.deactivate → 404."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb._deactivate_raises = ValueError("Entry kb-00099 not found")
    resp = client.post("/api/kb/entries/kb-00099/deactivate")
    assert resp.status_code == 404


def test_deactivate_already_inactive_409(client: TestClient) -> None:
    """ValueError 'already inactive' from kb.deactivate → 409."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb._deactivate_raises = ValueError("Entry kb-00001 is already inactive")
    resp = client.post("/api/kb/entries/kb-00001/deactivate")
    assert resp.status_code == 409


def test_deactivate_mental_map_422(client: TestClient) -> None:
    """Deactivating a mental_map entry is rejected with a clear 4xx message,
    and never reaches kb.deactivate or the graph-edge delete."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = _map_entry("kb-00001")
    resp = client.post("/api/kb/entries/kb-00001/deactivate")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "mental_map" in detail
    assert not kb.deactivate_calls
    assert not kb.db.calls


def test_deactivate_non_map_entry_unaffected(client: TestClient) -> None:
    """A non-map entry_type deactivates normally — the guard only blocks maps."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    kb.entries["kb-00001"] = make_entry("kb-00001")  # factual_reference
    resp = client.post("/api/kb/entries/kb-00001/deactivate")
    assert resp.status_code == 200
    assert kb.deactivate_calls


async def test_map_references_edges_survive_update_and_attempted_deactivate(
    client: TestClient,
) -> None:
    """The listener's detail -> owning-map reverse lookup depends on a map's
    outbound 'references' edges. Prove they survive: a metadata-only update
    doesn't touch graph_edges at all, and a blocked deactivate attempt never
    issues the delete — so a direct graph_edges query for edge_type=
    'references' still resolves the map's detail entries afterward."""
    _authed(client)
    kb: FakeKnowledgeBase = app.state.kb
    map_id = "kb-00001"
    kb.entries[map_id] = _map_entry(map_id)
    references_row = (map_id, "kb-00042", "references")
    kb.db.rows_for["edge_type = 'references'"] = [references_row]

    async def _references_for(source: str) -> list[Any]:
        cursor = await kb.db.execute(
            "SELECT source, target, edge_type FROM graph_edges"
            " WHERE source = ? AND edge_type = 'references'",
            (source,),
        )
        return await cursor.fetchall()

    # Metadata-only update: no delete of any kind is issued.
    resp = client.post(
        "/api/kb/store",
        json={"update_entry_id": map_id, "tags": ["reorg"]},
    )
    assert resp.status_code == 200
    assert not any("DELETE" in sql for sql, _ in kb.db.calls)
    assert await _references_for(map_id) == [references_row]

    # Attempted deactivate is blocked before any delete is issued.
    resp = client.post(f"/api/kb/entries/{map_id}/deactivate")
    assert resp.status_code == 422
    assert not any("DELETE" in sql for sql, _ in kb.db.calls)
    assert await _references_for(map_id) == [references_row]


# ─── POST /api/kb/entries/{id}/reactivate ────────────────────────────────────


def test_reactivate_happy_path(client: TestClient) -> None:
    """Admin reactivate calls facade and triggers best-effort graph rebuild."""
    _authed(client, admin=True)
    resp = client.post("/api/kb/entries/kb-00001/reactivate")
    assert resp.status_code == 200
    assert "entry" in resp.json()

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.reactivate_calls
    entry_id, contributor = kb.reactivate_calls[-1]
    assert entry_id == "kb-00001"
    assert contributor == "admin@example.com"

    # graph_builder.build_for_entry should have been called
    assert kb.graph_builder.build_calls, "build_for_entry was not called"
    # graph_enricher is None — endpoint must tolerate it silently
    assert kb.graph_enricher is None


def test_reactivate_requires_auth(client: TestClient) -> None:
    """POST /api/kb/entries/{id}/reactivate with no credentials returns 401."""
    resp = client.post("/api/kb/entries/kb-00001/reactivate")
    assert resp.status_code == 401


def test_reactivate_non_admin_403(client: TestClient) -> None:
    """Non-admin user accessing reactivate → 403."""
    _authed(client, admin=False)
    resp = client.post("/api/kb/entries/kb-00001/reactivate")
    assert resp.status_code == 403


def test_reactivate_not_found_404(client: TestClient) -> None:
    """ValueError 'not found' from kb.reactivate → 404."""
    _authed(client, admin=True)
    kb: FakeKnowledgeBase = app.state.kb
    kb._reactivate_raises = ValueError("Entry kb-00099 not found")
    resp = client.post("/api/kb/entries/kb-00099/reactivate")
    assert resp.status_code == 404


def test_reactivate_already_active_409(client: TestClient) -> None:
    """ValueError 'already active' from kb.reactivate → 409."""
    _authed(client, admin=True)
    kb: FakeKnowledgeBase = app.state.kb
    kb._reactivate_raises = ValueError("Entry kb-00001 is already active")
    resp = client.post("/api/kb/entries/kb-00001/reactivate")
    assert resp.status_code == 409


# ─── POST /api/kb/bulk_update ────────────────────────────────────────────────


def test_bulk_update_happy_path(client: TestClient) -> None:
    """Admin bulk_update with valid filters/updates returns shape + recorded call."""
    _authed(client, admin=True)
    resp = client.post(
        "/api/kb/bulk_update",
        json={
            "filters": {"contributor": "alice@example.com"},
            "updates": {"project_ref": "proj-x"},
            "dry_run": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is False
    assert isinstance(body["count"], int)
    assert isinstance(body["results"], list)


def test_bulk_update_requires_auth(client: TestClient) -> None:
    """POST /api/kb/bulk_update with no credentials returns 401."""
    resp = client.post(
        "/api/kb/bulk_update",
        json={
            "filters": {"contributor": "x"},
            "updates": {"project_ref": "y"},
        },
    )
    assert resp.status_code == 401


def test_bulk_update_non_admin_403(client: TestClient) -> None:
    """Non-admin user accessing bulk_update → 403."""
    _authed(client, admin=False)
    resp = client.post(
        "/api/kb/bulk_update",
        json={
            "filters": {"contributor": "x"},
            "updates": {"project_ref": "y"},
        },
    )
    assert resp.status_code == 403


def test_bulk_update_empty_filters_422(client: TestClient) -> None:
    """filters={} → 422 (mass-update guard)."""
    _authed(client, admin=True)
    resp = client.post(
        "/api/kb/bulk_update",
        json={"filters": {}, "updates": {"project_ref": "y"}},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)


def test_bulk_update_unknown_filter_key_422(client: TestClient) -> None:
    """Disallowed filter key 'short_title' → 422."""
    _authed(client, admin=True)
    resp = client.post(
        "/api/kb/bulk_update",
        json={
            "filters": {"short_title": "x"},
            "updates": {"project_ref": "y"},
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "short_title" in detail


def test_bulk_update_dry_run_defaults_true(client: TestClient) -> None:
    """Omitting dry_run → kb.bulk_update recorded dry_run=True."""
    _authed(client, admin=True)
    resp = client.post(
        "/api/kb/bulk_update",
        json={
            "filters": {"contributor": "alice@example.com"},
            "updates": {"project_ref": "proj-x"},
        },
    )
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    assert kb.bulk_update_calls
    last = kb.bulk_update_calls[-1]
    assert last["dry_run"] is True


# ─── POST /api/kb/feedback ───────────────────────────────────────────────────


def test_feedback_happy_path(client: TestClient) -> None:
    """Feedback endpoint writes to kb.db and returns status=recorded."""
    _authed(client)
    resp = client.post(
        "/api/kb/feedback",
        json={
            "feedback_type": "friction",
            "tool_name": "kb_search",
            "query_or_params": "some query",
            "detail": "Results were unhelpful",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert body["feedback_type"] == "friction"

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.db.calls, "db.execute was not called"
    sql, params = kb.db.calls[-1]
    assert "INSERT INTO agent_feedback" in sql
    assert isinstance(params, tuple)
    assert params[0] == "friction"  # feedback_type
    assert params[4] == "tester@example.com"  # contributor
    assert kb.db.committed >= 1


def test_feedback_requires_auth(client: TestClient) -> None:
    """POST /api/kb/feedback with no credentials returns 401."""
    resp = client.post("/api/kb/feedback", json={"feedback_type": "missing"})
    assert resp.status_code == 401


def test_feedback_bogus_type_422(client: TestClient) -> None:
    """Invalid feedback_type value → Pydantic 422."""
    _authed(client)
    resp = client.post("/api/kb/feedback", json={"feedback_type": "bogus"})
    assert resp.status_code == 422


# ─── router mounting sanity check ────────────────────────────────────────────


def test_write_routes_mounted() -> None:
    """All six write endpoints are registered on the app."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/store" in paths
    assert "/api/kb/store_batch" in paths
    assert "/api/kb/entries/{entry_id}/deactivate" in paths
    assert "/api/kb/entries/{entry_id}/reactivate" in paths
    assert "/api/kb/bulk_update" in paths
    assert "/api/kb/feedback" in paths
