"""Hermetic tests for kb_read_routes: get, graph, preflight, and list endpoints."""

from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry

import kb_service.routes.kb_read_routes as kb_read_routes
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    entry_id: str,
    *,
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    is_active: bool = True,
    superseded_by: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title="T",
        long_title="T",
        knowledge_details="D",
        entry_type=entry_type,
        is_active=is_active,
        superseded_by=superseded_by,
    )


# ---------------------------------------------------------------------------
# 401 gate — all new endpoints must reject unauthenticated requests
# ---------------------------------------------------------------------------


ALL_ENDPOINTS = [
    ("POST", "/api/kb/get", {"ids": ["kb-00001"]}),
    ("GET", "/api/kb/graph/neighbors", None),
    ("GET", "/api/kb/graph/bfs", None),
    ("GET", "/api/kb/graph/path", None),
    ("GET", "/api/kb/graph/supersedes-chain", None),
    ("GET", "/api/kb/graph/scope-entries", None),
    ("GET", "/api/kb/graph/vocabulary", None),
    ("GET", "/api/kb/graph/full", None),
    ("GET", "/api/kb/preflight", None),
    ("GET", "/api/kb/projects", None),
    ("GET", "/api/kb/contributors", None),
    ("GET", "/api/kb/teams", None),
]


@pytest.mark.parametrize("method,path,body", ALL_ENDPOINTS)
def test_endpoints_require_auth(
    client: TestClient, method: str, path: str, body: Any
) -> None:
    resp = client.post(path, json=body) if method == "POST" else client.get(path)
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/kb/get
# ---------------------------------------------------------------------------


def test_get_happy_path_order_preserved(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.entries["kb-00001"] = _entry("kb-00001")
    fake_kb.entries["kb-00002"] = _entry("kb-00002")

    resp = client.post("/api/kb/get", json={"ids": ["kb-00002", "kb-00001"]})
    assert resp.status_code == 200
    data = resp.json()
    ids = [r["id"] for r in data["results"]]
    assert ids == ["kb-00002", "kb-00001"]
    assert all(r["found"] for r in data["results"])


def test_get_unknown_id_not_found(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/get", json={"ids": ["kb-99999"]})
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["found"] is False
    assert result["entry"] is None
    assert result["pointer_rot"] == []


def test_get_inactive_entry_treated_as_not_found(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.entries["kb-00001"] = _entry("kb-00001", is_active=False)
    resp = client.post("/api/kb/get", json={"ids": ["kb-00001"]})
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["found"] is False
    assert result["entry"] is None


def test_get_empty_ids_422(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/get", json={"ids": []})
    assert resp.status_code == 422


def test_get_too_many_ids_422(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/get", json={"ids": [f"kb-{i:05d}" for i in range(21)]})
    assert resp.status_code == 422


def test_get_mental_map_pointer_rot(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Mental-map entry with one superseded and one deactivated neighbor."""
    app.dependency_overrides[get_current_user] = fake_user

    map_entry = _entry("kb-00001", entry_type=EntryType.MENTAL_MAP)
    fake_kb.entries["kb-00001"] = map_entry

    # Neighbors: two kb-id targets + one non-kb node (filtered out)
    fake_kb.graph.neighbors_result = [
        ("kb-00010", "relates_to", "outgoing"),
        ("kb-00005", "relates_to", "outgoing"),
        ("project:foo", "tagged_by", "outgoing"),  # filtered by regex
    ]
    # kb-00005: superseded (rot signal; sorted first)
    fake_kb.entries["kb-00005"] = _entry(
        "kb-00005", superseded_by="kb-00006", is_active=False
    )
    # kb-00010: deactivated (rot signal)
    fake_kb.entries["kb-00010"] = _entry("kb-00010", is_active=False)

    resp = client.post("/api/kb/get", json={"ids": ["kb-00001"]})
    assert resp.status_code == 200
    rot = resp.json()["results"][0]["pointer_rot"]
    # sorted ascending by target_id: kb-00005 < kb-00010
    assert len(rot) == 2
    assert rot[0] == {"target_id": "kb-00005", "superseded_by": "kb-00006"}
    assert rot[1] == {"target_id": "kb-00010", "superseded_by": None}


def test_get_non_mental_map_no_pointer_rot(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.entries["kb-00001"] = _entry("kb-00001", entry_type=EntryType.DECISION)
    resp = client.post("/api/kb/get", json={"ids": ["kb-00001"]})
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["pointer_rot"] == []
    # No graph calls for non-mental-map entries
    assert fake_kb.graph.calls == []


# ---------------------------------------------------------------------------
# touch_accessed monkeypatching
# ---------------------------------------------------------------------------


def test_touch_accessed_called_with_found_ids(
    client: TestClient, fake_kb: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.entries["kb-00001"] = _entry("kb-00001")
    fake_kb.entries["kb-00002"] = _entry("kb-00002")

    touched: list[Any] = []

    async def fake_touch(db: Any, ids: list[str]) -> None:
        touched.extend(ids)

    monkeypatch.setattr(kb_read_routes, "touch_accessed", fake_touch)

    resp = client.post(
        "/api/kb/get", json={"ids": ["kb-00001", "kb-99999", "kb-00002"]}
    )
    assert resp.status_code == 200
    # Only found ids; request order preserved
    assert touched == ["kb-00001", "kb-00002"]


def test_touch_accessed_not_called_when_nothing_found(
    client: TestClient, fake_kb: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    touched: list[Any] = []

    async def fake_touch(db: Any, ids: list[str]) -> None:
        touched.extend(ids)

    monkeypatch.setattr(kb_read_routes, "touch_accessed", fake_touch)

    resp = client.post("/api/kb/get", json={"ids": ["kb-99999"]})
    assert resp.status_code == 200
    assert touched == []


# ---------------------------------------------------------------------------
# GET /api/kb/preflight
# ---------------------------------------------------------------------------


def test_preflight_happy_path(client: TestClient, fake_kb: FakeKnowledgeBase) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.preflight_result = "## Projects\n..."

    resp = client.get(
        "/api/kb/preflight", params={"project_ref": "personal-kb", "since": "7d"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["project_ref"] == "personal-kb"
    assert data["context"] == "## Projects\n..."

    # Confirm a timedelta was passed as since
    assert fake_kb.preflight_calls
    pr, since_val = fake_kb.preflight_calls[-1]
    assert pr == "personal-kb"
    assert isinstance(since_val, timedelta)
    assert since_val == timedelta(days=7)


def test_preflight_no_since(client: TestClient, fake_kb: FakeKnowledgeBase) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/kb/preflight", params={"project_ref": "personal-kb"})
    assert resp.status_code == 200
    _pr, since_val = fake_kb.preflight_calls[-1]
    assert since_val is None


@pytest.mark.parametrize("bad_since", ["0d", "xyz", "0h", "abc123", "7x"])
def test_preflight_invalid_since_422(client: TestClient, bad_since: str) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get(
        "/api/kb/preflight", params={"project_ref": "personal-kb", "since": bad_since}
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/kb/graph/neighbors
# ---------------------------------------------------------------------------


def test_graph_neighbors_happy_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.neighbors_result = [
        ("kb-00002", "relates_to", "outgoing"),
        ("kb-00003", "supersedes", "incoming"),
    ]
    resp = client.get("/api/kb/graph/neighbors", params={"node_id": "kb-00001"})
    assert resp.status_code == 200
    neighbors = resp.json()["neighbors"]
    assert len(neighbors) == 2
    assert neighbors[0] == {
        "neighbor_id": "kb-00002",
        "edge_type": "relates_to",
        "direction": "outgoing",
    }


def test_graph_neighbors_invalid_direction_422(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get(
        "/api/kb/graph/neighbors",
        params={"node_id": "kb-00001", "direction": "sideways"},
    )
    assert resp.status_code == 422


def test_graph_neighbors_edge_types_list(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get(
        "/api/kb/graph/neighbors?node_id=kb-00001&edge_types=relates_to&edge_types=supersedes"
    )
    assert resp.status_code == 200
    call = next(c for c in fake_kb.graph.calls if c[0] == "neighbors")
    assert call[2]["edge_types"] == ["relates_to", "supersedes"]


# ---------------------------------------------------------------------------
# GET /api/kb/graph/bfs
# ---------------------------------------------------------------------------


def test_graph_bfs_happy_path(client: TestClient, fake_kb: FakeKnowledgeBase) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.bfs_result = [
        ("kb-00002", 1, ["kb-00001", "kb-00002"]),
        ("kb-00003", 2, ["kb-00001", "kb-00002", "kb-00003"]),
    ]
    resp = client.get("/api/kb/graph/bfs", params={"start_node": "kb-00001"})
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 2
    assert entries[0]["entry_id"] == "kb-00002"
    assert entries[0]["depth"] == 1
    assert entries[0]["path"] == ["kb-00001", "kb-00002"]


# ---------------------------------------------------------------------------
# GET /api/kb/graph/path
# ---------------------------------------------------------------------------


def test_graph_path_happy_path(client: TestClient, fake_kb: FakeKnowledgeBase) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.find_path_result = [("kb-00001", "relates_to", "kb-00002")]
    resp = client.get(
        "/api/kb/graph/path", params={"source": "kb-00001", "target": "kb-00002"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert len(data["hops"]) == 1
    assert data["hops"][0] == {
        "source": "kb-00001",
        "edge_type": "relates_to",
        "target": "kb-00002",
    }


def test_graph_path_none_not_found(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.find_path_result = None
    resp = client.get(
        "/api/kb/graph/path", params={"source": "kb-00001", "target": "kb-00002"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False
    assert data["hops"] == []


def test_graph_path_source_equals_target(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.find_path_result = []  # source == target
    resp = client.get(
        "/api/kb/graph/path", params={"source": "kb-00001", "target": "kb-00001"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["hops"] == []


# ---------------------------------------------------------------------------
# GET /api/kb/graph/supersedes-chain
# ---------------------------------------------------------------------------


def test_graph_supersedes_chain_happy_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.supersedes_chain_result = ["kb-00001", "kb-00002", "kb-00003"]
    resp = client.get("/api/kb/graph/supersedes-chain", params={"entry_id": "kb-00003"})
    assert resp.status_code == 200
    assert resp.json()["chain"] == ["kb-00001", "kb-00002", "kb-00003"]


# ---------------------------------------------------------------------------
# GET /api/kb/graph/scope-entries
# ---------------------------------------------------------------------------


def test_graph_scope_entries_happy_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.entries_for_scope_result = ["kb-00001", "kb-00002"]
    resp = client.get(
        "/api/kb/graph/scope-entries", params={"scope": "project:personal-kb"}
    )
    assert resp.status_code == 200
    assert resp.json()["entry_ids"] == ["kb-00001", "kb-00002"]


# ---------------------------------------------------------------------------
# GET /api/kb/graph/vocabulary
# ---------------------------------------------------------------------------


def test_graph_vocabulary_happy_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.graph.vocabulary_result = {
        "project": ["personal-kb", "team-kb"],
        "tag": ["python"],
    }
    resp = client.get("/api/kb/graph/vocabulary")
    assert resp.status_code == 200
    data = resp.json()["nodes"]
    assert data["project"] == ["personal-kb", "team-kb"]
    assert data["tag"] == ["python"]


# ---------------------------------------------------------------------------
# List endpoints: /api/kb/projects, /contributors, /teams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,sql_fragment",
    [
        ("/api/kb/projects", "project_ref"),
        ("/api/kb/contributors", "contributor"),
        ("/api/kb/teams", "team"),
    ],
)
def test_list_endpoint_maps_rows(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    path: str,
    sql_fragment: str,
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.db.rows = [("personal-kb", 42)]
    resp = client.get(path)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items == [{"name": "personal-kb", "entry_count": 42}]


@pytest.mark.parametrize(
    "path",
    [
        "/api/kb/projects",
        "/api/kb/contributors",
        "/api/kb/teams",
    ],
)
def test_list_endpoint_empty_rows_200(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    path: str,
) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.db.rows = []
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


# ---------------------------------------------------------------------------
# GET /api/kb/graph/full
# ---------------------------------------------------------------------------


def test_graph_full_happy_path(client: TestClient, fake_kb: FakeKnowledgeBase) -> None:
    """Full-graph dump applies inactive/orphan/edge filter rules correctly.

    Seed:
    - kb-00001: active entry node (conn_count=2), with entry metadata
    - kb-00002: inactive entry node — must be excluded
    - tool:sqlite: entity node (conn_count=1) — included; label derived from node_id
    - orphan:x: entity node (conn_count=0) — excluded as orphan
    Edges:
    - kb-00001 -> tool:sqlite (valid)
    - kb-00002 -> kb-00001 (touches excluded node — must be excluded)

    rows_for insertion order (COLLISION TRAP — see conftest AC):
      'is_active = 0'  → query 1 (inactive IDs)
      'conn_count'     → query 2 (graph_nodes — contains 'FROM graph_edges' too)
      'FROM graph_edges' → query 3 (edges only)
      'is_active = 1'  → query 4 (entry metadata)
    """
    app.dependency_overrides[get_current_user] = fake_user

    # Must insert in this EXACT order — first-match-wins on substring lookup
    fake_kb.db.rows_for["is_active = 0"] = [("kb-00002",)]
    fake_kb.db.rows_for["conn_count"] = [
        ("kb-00001", "entry", None, 2),  # active entry node
        ("kb-00002", "entry", None, 1),  # inactive → excluded
        ("tool:sqlite", "tool", None, 1),  # entity with connections → included
        ("orphan:x", "other", None, 0),  # orphan → excluded
    ]
    fake_kb.db.rows_for["FROM graph_edges"] = [
        ("kb-00001", "tool:sqlite", "uses", None),  # valid
        ("kb-00002", "kb-00001", "related", None),  # touches inactive → excluded
    ]
    fake_kb.db.rows_for["is_active = 1"] = [
        (
            "kb-00001",
            "SQLite Entry",
            "Using SQLite for storage",
            "factual_reference",
            None,
            0.9,
            "me",
            "my-project",
        ),
    ]

    resp = client.get("/api/kb/graph/full")
    assert resp.status_code == 200
    data = resp.json()

    # ── nodes ────────────────────────────────────────────────────────────────
    nodes_by_id = {n["id"]: n for n in data["nodes"]}

    # Inactive entry excluded
    assert "kb-00002" not in nodes_by_id
    # Orphan entity excluded
    assert "orphan:x" not in nodes_by_id

    # Active entry: label == short_title; val == conn_count; properties include meta
    entry_node = nodes_by_id["kb-00001"]
    assert entry_node["label"] == "SQLite Entry"
    assert entry_node["val"] == 2
    assert entry_node["properties"]["project_ref"] == "my-project"

    # Entity node: label derived from node_id split on ':'
    tool_node = nodes_by_id["tool:sqlite"]
    assert tool_node["label"] == "sqlite"
    assert tool_node["val"] == 1

    # ── edges ────────────────────────────────────────────────────────────────
    edges = data["edges"]
    edge_keys = [(e["source"], e["target"]) for e in edges]

    # Valid edge present
    assert ("kb-00001", "tool:sqlite") in edge_keys
    # Edge touching excluded node absent
    assert ("kb-00002", "kb-00001") not in edge_keys

    # ── stats ─────────────────────────────────────────────────────────────────
    assert data["stats"]["node_count"] == len(data["nodes"])
    assert data["stats"]["edge_count"] == len(data["edges"])
    assert data["stats"]["node_count"] == 2
    assert data["stats"]["edge_count"] == 1
