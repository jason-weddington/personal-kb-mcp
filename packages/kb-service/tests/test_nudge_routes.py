"""Hermetic tests for POST /api/kb/pointer-candidates (nudge_routes.py)."""

from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry

from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import fake_user

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    entry_id: str,
    *,
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    project_ref: str | None = "proj-a",
    has_embedding: bool = True,
    short_title: str = "T",
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title="T long",
        knowledge_details="D",
        entry_type=entry_type,
        project_ref=project_ref,
        has_embedding=has_embedding,
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_pointer_candidates_requires_auth(client: TestClient) -> None:
    """POST /api/kb/pointer-candidates with no Authorization header returns 401."""
    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 401


def test_pointer_candidates_reachable_through_real_app(client: TestClient) -> None:
    """The route is registered on the real app object — an unregistered router 404s."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001", has_embedding=False)
    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    # Reachable at all (not 404-unregistered) and returns a well-formed body.
    assert resp.status_code == 200
    assert resp.json() == {"has_owning_map": False, "candidates": []}


# ---------------------------------------------------------------------------
# Missing entry -> 404
# ---------------------------------------------------------------------------


def test_pointer_candidates_missing_entry_404(client: TestClient) -> None:
    """A genuinely-missing entry_id is the one degenerate case that 404s."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-99999"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# has_owning_map — both ways
# ---------------------------------------------------------------------------


def test_pointer_candidates_has_owning_map_true(client: TestClient) -> None:
    """An ACTIVE mental_map with a references edge -> has_owning_map True, no kNN."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001")
    app.state.kb.db.has_owning_map = True
    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_owning_map"] is True
    assert body["candidates"] == []
    # No kNN was run once an owning map was found.
    assert app.state.kb.db.vector_search_calls == []


def test_pointer_candidates_has_owning_map_false_with_candidates(
    client: TestClient,
) -> None:
    """No owning map + an embedded entry with a project_ref -> candidates from kNN."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001", project_ref="proj-a")
    app.state.kb.entries["kb-00050"] = _entry(
        "kb-00050",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="proj-a",
        short_title="Proj A map",
    )
    app.state.kb.db.has_owning_map = False
    app.state.kb.db.own_embedding = [0.1, 0.2, 0.3]
    app.state.kb.db.vector_search_result = [("kb-00050", 0.05)]

    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_owning_map"] is False
    assert body["candidates"] == [
        {
            "map_id": "kb-00050",
            "short_title": "Proj A map",
            "project_ref": "proj-a",
            "distance": 0.05,
        }
    ]

    # vector_search was scoped to the entry's own project_ref and mental_map.
    embedding, kwargs = app.state.kb.db.vector_search_calls[0]
    assert embedding == [0.1, 0.2, 0.3]
    assert kwargs["project_ref"] == "proj-a"
    assert kwargs["entry_type"] == "mental_map"
    assert kwargs["limit"] == 5


def test_pointer_candidates_excludes_self_from_candidates(client: TestClient) -> None:
    """A kNN hit that IS the queried entry itself (edge case) is filtered out."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001", project_ref="proj-a")
    app.state.kb.db.own_embedding = [0.1, 0.2]
    app.state.kb.db.vector_search_result = [("kb-00001", 0.0)]

    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 200
    assert resp.json()["candidates"] == []


# ---------------------------------------------------------------------------
# Graceful empty — every degenerate case is a 200, not an error
# ---------------------------------------------------------------------------


def test_pointer_candidates_no_embedding_yet_is_graceful_empty(
    client: TestClient,
) -> None:
    """No embedding yet is the COMMON transient state right after a store.

    Not an error.

    The retry queue enqueues on embed failure, so ``has_embedding=False`` is
    normal immediately after ``kb_store``, before the embed worker runs.
    """
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001", has_embedding=False)
    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 200
    assert resp.json() == {"has_owning_map": False, "candidates": []}
    # No kNN was attempted — there's no vector to search from.
    assert app.state.kb.db.vector_search_calls == []


def test_pointer_candidates_no_project_ref_is_graceful_empty(
    client: TestClient,
) -> None:
    """An entry with no project_ref can't be scoped to a per-project kNN."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001", project_ref=None)
    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 200
    assert resp.json() == {"has_owning_map": False, "candidates": []}
    assert app.state.kb.db.vector_search_calls == []


def test_pointer_candidates_no_maps_in_project_is_graceful_empty(
    client: TestClient,
) -> None:
    """An embedded entry whose project has zero mental_maps -> empty candidates."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.entries["kb-00001"] = _entry("kb-00001", project_ref="proj-a")
    app.state.kb.db.own_embedding = [0.1, 0.2]
    app.state.kb.db.vector_search_result = []

    resp = client.post("/api/kb/pointer-candidates", json={"entry_id": "kb-00001"})
    assert resp.status_code == 200
    assert resp.json() == {"has_owning_map": False, "candidates": []}
