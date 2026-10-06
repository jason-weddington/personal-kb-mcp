"""Hermetic tests for the KB ingest endpoints.

Covers: POST /api/kb/ingest/text, /url, /file.

All tests are hermetic (no live Postgres / Ollama / network).  The ``client``
fixture from conftest.py patches ``create_postgres`` / ``init_db`` / ``get_db``,
so these tests only need to override the auth dependency and optionally inject
errors or monkeypatch module-level names.
"""

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core import Attribution

import kb_service.routes.ingest_routes as ingest_routes_module
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user

# ─── helpers ─────────────────────────────────────────────────────────────────


def _authed(client: TestClient) -> None:
    """Override the auth dependency on the global app."""
    app.dependency_overrides[get_current_user] = fake_user


def _make_attribution(
    *, contributor: str = "tester@example.com", team: str | None = None
) -> Attribution:
    """Build an Attribution for monkeypatching resolve_attribution."""
    return Attribution(contributor=contributor, team=team)


def _patch_attribution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    team: str | None = "team-x",
    contributor: str = "tester@example.com",
) -> None:
    """Monkeypatch resolve_attribution in ingest_routes to return a fixed value."""
    attr = _make_attribution(contributor=contributor, team=team)

    async def _fake_resolve(user: Any) -> Attribution:
        return attr

    monkeypatch.setattr(ingest_routes_module, "resolve_attribution", _fake_resolve)


# ─── 401 — unauthenticated ────────────────────────────────────────────────────


def test_text_requires_auth(client: TestClient) -> None:
    """POST /api/kb/ingest/text without credentials returns 401."""
    resp = client.post(
        "/api/kb/ingest/text", json={"content": "x", "source_name": "x.md"}
    )
    assert resp.status_code == 401


def test_url_requires_auth(client: TestClient) -> None:
    """POST /api/kb/ingest/url without credentials returns 401."""
    resp = client.post("/api/kb/ingest/url", json={"url": "https://example.com"})
    assert resp.status_code == 401


def test_file_requires_auth(client: TestClient) -> None:
    """POST /api/kb/ingest/file without credentials returns 401."""
    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("test.md", b"hello", "text/plain")},
    )
    assert resp.status_code == 401


# ─── /text happy paths ────────────────────────────────────────────────────────


def test_text_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: ingest_text called with content+source_name and contributor."""
    _authed(client)
    _patch_attribution(monkeypatch, team="team-x")

    resp = client.post(
        "/api/kb/ingest/text",
        json={"content": "hello world", "source_name": "note.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "doc.md"
    assert body["action"] == "ingested"

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.ingest_text_calls, "ingest_text was not called"
    args, kwargs = kb.ingest_text_calls[-1]
    assert args == ("hello world", "note.md")
    assert kwargs["contributor"] == "tester@example.com"
    assert kwargs["team"] == "team-x"


def test_text_with_project_ref_and_dry_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """project_ref and dry_run are forwarded to ingest_text."""
    _authed(client)
    _patch_attribution(monkeypatch, team="team-x")

    resp = client.post(
        "/api/kb/ingest/text",
        json={
            "content": "some text",
            "source_name": "doc.txt",
            "project_ref": "proj-x",
            "dry_run": True,
        },
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    _, kwargs = kb.ingest_text_calls[-1]
    assert kwargs["project_ref"] == "proj-x"
    assert kwargs["dry_run"] is True


def test_text_team_none_forwarded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When resolve_attribution returns team=None, team=None is explicitly forwarded."""
    _authed(client)
    _patch_attribution(monkeypatch, team=None)

    resp = client.post(
        "/api/kb/ingest/text",
        json={"content": "data", "source_name": "x.md"},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    _, kwargs = kb.ingest_text_calls[-1]
    assert "team" in kwargs
    assert kwargs["team"] is None


def test_text_response_matches_file_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Response JSON field-for-field equals the configured file_result."""
    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    resp = client.post(
        "/api/kb/ingest/text",
        json={"content": "data", "source_name": "x.md"},
    )
    assert resp.status_code == 200
    assert resp.json() == dataclasses.asdict(kb.file_result)


# ─── /url happy paths ─────────────────────────────────────────────────────────


def test_url_without_content_calls_ingest_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """URL-only mode calls ingest_url with project_ref/dry_run/contributor/team."""
    _authed(client)
    _patch_attribution(monkeypatch, team="team-x")

    resp = client.post(
        "/api/kb/ingest/url",
        json={"url": "https://example.com", "project_ref": "proj-x", "dry_run": True},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.ingest_url_calls, "ingest_url was not called"
    args, kwargs = kb.ingest_url_calls[-1]
    assert args == ("https://example.com",)
    assert kwargs["project_ref"] == "proj-x"
    assert kwargs["dry_run"] is True
    assert kwargs["contributor"] == "tester@example.com"
    assert kwargs["team"] == "team-x"


def test_url_with_content_calls_ingest_url_content(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With content, ingest_url_content is called with (content, url) positionally."""
    _authed(client)
    _patch_attribution(monkeypatch, team="team-x")

    resp = client.post(
        "/api/kb/ingest/url",
        json={
            "url": "https://example.com/page",
            "content": "pre-fetched text",
            "project_ref": "proj-x",
            "dry_run": False,
        },
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.ingest_url_content_calls, "ingest_url_content was not called"
    assert not kb.ingest_url_calls, "ingest_url must not be called with content"

    args, kwargs = kb.ingest_url_content_calls[-1]
    assert args == ("pre-fetched text", "https://example.com/page")
    assert kwargs["project_ref"] == "proj-x"
    assert kwargs["dry_run"] is False
    assert kwargs["contributor"] == "tester@example.com"
    assert kwargs["team"] == "team-x"


def test_url_error_action_passes_through_as_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """action='error' from kb-core passes through as HTTP 200 — no error mapping."""
    from kb_core.ingest.ingester import FileResult

    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    kb.file_result = FileResult(
        path="https://example.com",
        action="error",
        reason="Failed to fetch: connection refused",
    )

    resp = client.post(
        "/api/kb/ingest/url",
        json={"url": "https://example.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "error"
    assert "Failed to fetch" in body["reason"]


# ─── /file happy paths ────────────────────────────────────────────────────────


def test_file_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """File upload calls ingest_file with path ending in original filename."""
    _authed(client)
    _patch_attribution(monkeypatch, team="team-x")

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("upload.md", b"# Hello\n\nWorld", "text/markdown")},
        data={"project_ref": "proj-x", "dry_run": "true"},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.ingest_file_calls, "ingest_file was not called"
    args, kwargs = kb.ingest_file_calls[-1]
    path_arg = args[0]
    assert str(path_arg).endswith("upload.md"), f"path ending wrong: {path_arg}"
    assert kwargs["project_ref"] == "proj-x"
    assert kwargs["dry_run"] is True
    assert kwargs["contributor"] == "tester@example.com"
    assert kwargs["team"] == "team-x"


def test_file_temp_dir_cleaned_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the /file response, the temp directory no longer exists."""
    _authed(client)
    _patch_attribution(monkeypatch)

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("doc.txt", b"content", "text/plain")},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    args, _ = kb.ingest_file_calls[-1]
    tmp_path = Path(str(args[0]))
    assert not tmp_path.parent.exists(), "Temp directory should be cleaned up"


def test_file_missing_filename_422(client: TestClient) -> None:
    """A file upload without a filename returns 422."""
    _authed(client)

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("", b"data", "application/octet-stream")},
    )
    assert resp.status_code == 422


# ─── upload size cap ─────────────────────────────────────────────────────────


def test_file_oversize_413(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 100-byte upload with KB_INGEST_MAX_FILE_SIZE=10 returns 413."""
    _authed(client)
    monkeypatch.setenv("KB_INGEST_MAX_FILE_SIZE", "10")

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("big.md", b"x" * 100, "text/markdown")},
    )
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "10" in detail


def test_file_oversize_413_no_ingest_called(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On 413, ingest_file is never called."""
    _authed(client)
    monkeypatch.setenv("KB_INGEST_MAX_FILE_SIZE", "10")

    client.post(
        "/api/kb/ingest/file",
        files={"file": ("big.md", b"x" * 100, "text/markdown")},
    )
    kb: FakeKnowledgeBase = app.state.kb
    assert not kb.ingest_file_calls, "ingest_file should NOT be called on 413"


def test_file_exactly_at_limit_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 10-byte upload with KB_INGEST_MAX_FILE_SIZE=10 proceeds (at-limit allowed)."""
    _authed(client)
    _patch_attribution(monkeypatch)
    monkeypatch.setenv("KB_INGEST_MAX_FILE_SIZE", "10")

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("exact.md", b"x" * 10, "text/markdown")},
    )
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    assert kb.ingest_file_calls, "ingest_file must be called for at-limit upload"


# ─── safety gate ─────────────────────────────────────────────────────────────


def test_safety_503_when_deps_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503 when safety deps missing and KB_SKIP_SAFETY not set."""
    _authed(client)
    monkeypatch.setattr(
        ingest_routes_module,
        "_missing_safety_deps",
        lambda: ["detect-secrets", "scrubadub"],
    )
    monkeypatch.delenv("KB_SKIP_SAFETY", raising=False)

    resp = client.post(
        "/api/kb/ingest/text",
        json={"content": "x", "source_name": "x.md"},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "detect-secrets" in detail
    assert "scrubadub" in detail
    assert "KB_SKIP_SAFETY" in detail


def test_safety_503_no_facade_called(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Facade ingest methods are NOT called when 503 safety gate fires."""
    _authed(client)
    monkeypatch.setattr(
        ingest_routes_module,
        "_missing_safety_deps",
        lambda: ["detect-secrets", "scrubadub"],
    )
    monkeypatch.delenv("KB_SKIP_SAFETY", raising=False)

    for endpoint, payload in [
        ("/api/kb/ingest/text", {"content": "x", "source_name": "x.md"}),
        ("/api/kb/ingest/url", {"url": "https://example.com"}),
    ]:
        client.post(endpoint, json=payload)

    client.post(
        "/api/kb/ingest/file",
        files={"file": ("f.md", b"x", "text/plain")},
    )

    kb: FakeKnowledgeBase = app.state.kb
    assert not kb.ingest_text_calls
    assert not kb.ingest_url_calls
    assert not kb.ingest_url_content_calls
    assert not kb.ingest_file_calls


def test_safety_bypass_200(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """KB_SKIP_SAFETY=TRUE bypasses the gate and calls the facade."""
    _authed(client)
    _patch_attribution(monkeypatch)
    monkeypatch.setattr(
        ingest_routes_module,
        "_missing_safety_deps",
        lambda: ["detect-secrets", "scrubadub"],
    )
    monkeypatch.setenv("KB_SKIP_SAFETY", "TRUE")

    resp = client.post(
        "/api/kb/ingest/text",
        json={"content": "x", "source_name": "x.md"},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    assert kb.ingest_text_calls, "ingest_text should be called when safety bypassed"


# ─── RuntimeError → 503 ───────────────────────────────────────────────────────


def test_text_runtime_error_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RuntimeError from ingest_text maps to 503."""
    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    kb._ingest_raises = RuntimeError("LLM not configured")

    resp = client.post(
        "/api/kb/ingest/text",
        json={"content": "x", "source_name": "x.md"},
    )
    assert resp.status_code == 503
    assert "LLM not configured" in resp.json()["detail"]


def test_url_runtime_error_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RuntimeError from ingest_url maps to 503."""
    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    kb._ingest_raises = RuntimeError("embedder missing")

    resp = client.post(
        "/api/kb/ingest/url",
        json={"url": "https://example.com"},
    )
    assert resp.status_code == 503
    assert "embedder missing" in resp.json()["detail"]


def test_file_runtime_error_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RuntimeError from ingest_file maps to 503."""
    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    kb._ingest_raises = RuntimeError("no extraction LLM")

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("f.md", b"content", "text/plain")},
    )
    assert resp.status_code == 503
    assert "no extraction LLM" in resp.json()["detail"]


# ─── error action passthrough (200) ──────────────────────────────────────────


def test_file_error_action_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """action='error' from ingest_file passes through as HTTP 200 — no error mapping."""
    from kb_core.ingest.ingester import FileResult

    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    kb.file_result = FileResult(
        path="upload.md",
        action="error",
        reason="Denied by allowlist",
    )

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("upload.md", b"data", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "error"
    assert body["reason"] == "Denied by allowlist"
    assert body["path"] == "upload.md"


# ─── response JSON equality ───────────────────────────────────────────────────


def test_file_response_matches_file_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Response JSON field-for-field equals the configured file_result."""
    _authed(client)
    _patch_attribution(monkeypatch)

    kb: FakeKnowledgeBase = app.state.kb
    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("doc.md", b"data", "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json() == dataclasses.asdict(kb.file_result)


# ─── team monkeypatch assertions ──────────────────────────────────────────────


def test_url_team_none_forwarded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When resolve_attribution returns team=None, team=None is explicitly forwarded."""
    _authed(client)
    _patch_attribution(monkeypatch, team=None)

    resp = client.post(
        "/api/kb/ingest/url",
        json={"url": "https://example.com"},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    _, kwargs = kb.ingest_url_calls[-1]
    assert "team" in kwargs
    assert kwargs["team"] is None


def test_file_team_kwarg_forwarded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """team kwarg is forwarded to ingest_file from resolve_attribution."""
    _authed(client)
    _patch_attribution(monkeypatch, team="team-x")

    resp = client.post(
        "/api/kb/ingest/file",
        files={"file": ("f.md", b"data", "text/plain")},
    )
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    _, kwargs = kb.ingest_file_calls[-1]
    assert kwargs["team"] == "team-x"
