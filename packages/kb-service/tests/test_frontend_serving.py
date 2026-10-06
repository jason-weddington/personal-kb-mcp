"""Hermetic tests for mount_frontend.

Use a fresh FastAPI() instance, not kb_service.main.app.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kb_service.main import (
    PACKAGED_STATIC,
    mount_frontend,
    resolve_static_dir,
)


@pytest.fixture()
def dist_dir(tmp_path: Path) -> Path:
    """Create a minimal dist directory with index.html and assets/app.js."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>SPA</html>")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('app')")
    return dist


def make_app(dist: Path) -> FastAPI:
    app = FastAPI()
    mount_frontend(app, dist)
    return app


def test_mount_frontend_returns_true_when_index_exists(dist_dir: Path) -> None:
    app = FastAPI()
    result = mount_frontend(app, dist_dir)
    assert result is True


def test_mount_frontend_returns_false_when_dist_missing(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no-dist"
    app = FastAPI()
    result = mount_frontend(app, nonexistent)
    assert result is False


def test_missing_dist_no_op_get_returns_404(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no-dist"
    app = FastAPI()
    mount_frontend(app, nonexistent)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 404


def test_root_returns_index_html(dist_dir: Path) -> None:
    client = TestClient(make_app(dist_dir))
    response = client.get("/")
    assert response.status_code == 200
    assert b"SPA" in response.content


def test_deep_route_returns_index_html(dist_dir: Path) -> None:
    client = TestClient(make_app(dist_dir))
    response = client.get("/some/deep/route")
    assert response.status_code == 200
    assert b"SPA" in response.content


def test_asset_served_directly(dist_dir: Path) -> None:
    client = TestClient(make_app(dist_dir))
    response = client.get("/assets/app.js")
    assert response.status_code == 200
    assert b"console.log" in response.content


def test_api_path_returns_404(dist_dir: Path) -> None:
    client = TestClient(make_app(dist_dir), raise_server_exceptions=False)
    response = client.get("/api/nonexistent")
    assert response.status_code == 404


def test_path_traversal_does_not_serve_out_of_tree_file(
    tmp_path: Path, dist_dir: Path
) -> None:
    """A traversal request must never serve a file outside dist/."""
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")

    client = TestClient(make_app(dist_dir), raise_server_exceptions=False)

    # Try percent-encoded traversal (%2e%2e)
    response = client.get("/%2e%2e/secret.txt")
    assert response.status_code in (200, 404)
    assert b"TOP SECRET" not in response.content

    # Try raw dotdot path
    response2 = client.get("/../secret.txt")
    assert response2.status_code in (200, 404)
    assert b"TOP SECRET" not in response2.content


def _make_dist(root: Path, name: str) -> Path:
    d = root / name
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(f"<html>{name}</html>")
    (d / "assets" / "app.js").write_text("console.log('x')")
    return d


def test_resolve_prefers_env_dir(tmp_path: Path) -> None:
    env, dev, pkg = (_make_dist(tmp_path, n) for n in ("env", "dev", "pkg"))
    assert resolve_static_dir(str(env), dev, pkg) == (env, "KB_SERVICE_STATIC_DIR")


def test_resolve_falls_to_dev_build(tmp_path: Path) -> None:
    dev, pkg = (_make_dist(tmp_path, n) for n in ("dev", "pkg"))
    assert resolve_static_dir(str(tmp_path / "missing"), dev, pkg) == (
        dev,
        "dev build",
    )


def test_resolve_falls_to_packaged(tmp_path: Path) -> None:
    pkg = _make_dist(tmp_path, "pkg")
    assert resolve_static_dir("", tmp_path / "nodev", pkg) == (pkg, "packaged")


def test_resolve_none_found(tmp_path: Path) -> None:
    assert resolve_static_dir("", tmp_path / "a", tmp_path / "b") is None


def test_resolve_reads_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _make_dist(tmp_path, "env")
    monkeypatch.setenv("KB_SERVICE_STATIC_DIR", str(env))
    result = resolve_static_dir(None, tmp_path / "a", tmp_path / "b")
    assert result == (env, "KB_SERVICE_STATIC_DIR")


def test_packaged_only_app_serves_ui(tmp_path: Path) -> None:
    pkg = _make_dist(tmp_path, "pkg")
    resolved = resolve_static_dir("", tmp_path / "nodev", pkg)
    assert resolved is not None
    client = TestClient(make_app(resolved[0]))
    root = client.get("/")
    assert root.status_code == 200
    assert b"pkg" in root.content
    assert client.get("/assets/app.js").status_code == 200


def test_real_packaged_static_has_index() -> None:
    """The committed packaged UI must exist so installs serve it."""
    assert (PACKAGED_STATIC / "index.html").is_file()
