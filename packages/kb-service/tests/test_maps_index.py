"""Hermetic tests for GET /api/kb/maps-index."""

from fastapi.testclient import TestClient

from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import MapRef, MapsIndexResponse, ProjectMaps
from tests.conftest import fake_user


def test_maps_index_requires_auth(client: TestClient) -> None:
    """GET /api/kb/maps-index with no Authorization header returns exactly 401."""
    resp = client.get("/api/kb/maps-index")
    assert resp.status_code == 401


def test_maps_index_empty(client: TestClient) -> None:
    """No mental_map projects configured returns 200 with an empty projects list."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/kb/maps-index")
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


def test_maps_index_sorted(client: TestClient) -> None:
    """Projects returned sorted ascending by project_ref regardless of config order."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.maps_projects = {
        "proj-b": [{"id": "kb-2", "short_title": "B map", "long_title": "B long"}],
        "proj-a": [{"id": "kb-1", "short_title": "A map", "long_title": "A long"}],
    }
    resp = client.get("/api/kb/maps-index")
    assert resp.status_code == 200
    body = resp.json()
    assert [p["project_ref"] for p in body["projects"]] == ["proj-a", "proj-b"]
    assert body["projects"][0]["maps"] == [
        {"id": "kb-1", "short_title": "A map", "long_title": "A long"}
    ]
    assert body["projects"][1]["maps"] == [
        {"id": "kb-2", "short_title": "B map", "long_title": "B long"}
    ]


def test_maps_index_empty_project_omitted(client: TestClient) -> None:
    """A project_ref mapping to an empty maps list is omitted from the response."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.maps_projects = {
        "proj-with-maps": [{"id": "kb-1", "short_title": "A", "long_title": "AA"}],
        "proj-empty": [],
    }
    resp = client.get("/api/kb/maps-index")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["projects"]) == 1
    assert body["projects"][0]["project_ref"] == "proj-with-maps"


def test_maps_index_filters_none_and_empty_refs(client: TestClient) -> None:
    """None and empty-string project refs in discovery rows are filtered out."""
    app.dependency_overrides[get_current_user] = fake_user
    app.state.kb.maps_rows = [(None,), ("",), ("proj-a",)]
    app.state.kb.maps_projects = {
        "proj-a": [{"id": "kb-1", "short_title": "A", "long_title": "AA"}],
    }
    resp = client.get("/api/kb/maps-index")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["projects"]) == 1
    assert body["projects"][0]["project_ref"] == "proj-a"


def test_maps_index_round_trip_contract() -> None:
    """ProjectMaps.model_dump() equals the legacy JSONL record format exactly."""
    result = ProjectMaps(
        project_ref="x",
        maps=[MapRef(id="kb-1", short_title="s", long_title="l")],
    ).model_dump()
    assert result == {
        "project_ref": "x",
        "maps": [{"id": "kb-1", "short_title": "s", "long_title": "l"}],
    }


def test_maps_index_response_model_parsed() -> None:
    """MapsIndexResponse validates and round-trips correctly."""
    response = MapsIndexResponse(
        projects=[
            ProjectMaps(
                project_ref="demo",
                maps=[MapRef(id="kb-99", short_title="Demo", long_title="Demo map")],
            )
        ]
    )
    dumped = response.model_dump()
    assert dumped == {
        "projects": [
            {
                "project_ref": "demo",
                "maps": [
                    {"id": "kb-99", "short_title": "Demo", "long_title": "Demo map"}
                ],
            }
        ]
    }
