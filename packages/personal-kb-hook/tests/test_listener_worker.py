"""Tests for personal_kb_hook.listener_worker.

All network access is mocked via monkeypatch (same shim pattern as
tests/test_http_index.py). Zero real HTTP calls.

Covers:
- Pinned request URL / headers / body bytes.
- 200 + valid map → atomic cache merge preserving whispered_map_ids;
  missing/None long_title coerced to ''.
- null map / non-dict body / missing 'map' key / invalid map dict /
  HTTPError / timeout / bad JSON → cache untouched.
- Request tmp file always deleted.
- Exit 0 on all paths (main() never raises).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import listener_worker

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _MockResponse:
    """Minimal context-manager HTTP response shim (mirrors test_http_index.py)."""

    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _ok_response(body: dict[str, Any]) -> _MockResponse:
    return _MockResponse(json.dumps(body))


@pytest.fixture
def worker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate env vars and HOME for cache path expansion."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-key")
    cache_dir = tmp_path / ".cache" / "personal_kb"
    cache_dir.mkdir(parents=True)
    return {"root": tmp_path, "cache_dir": cache_dir}


def _make_request_file(tmp_path: Path, data: dict[str, Any]) -> Path:
    """Write a request JSON file and return its path."""
    req_file = tmp_path / "req.json"
    req_file.write_text(json.dumps(data), encoding="utf-8")
    return req_file


def _run_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_data: dict[str, Any],
    cache_path: Path,
    urlopen_impl: Any,
) -> None:
    """Run listener_worker.main() with the given request data and mocked urlopen."""
    req_file = _make_request_file(tmp_path, request_data)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen_impl)
    monkeypatch.setattr(
        sys,
        "argv",
        ["listener_worker", str(req_file), str(cache_path)],
    )
    listener_worker.main()


# ---------------------------------------------------------------------------
# Pinned request URL, headers, body bytes
# ---------------------------------------------------------------------------


def test_request_url_and_headers(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker POSTs to the pinned endpoint with correct Authorization header."""
    captured: list[urllib.request.Request] = []

    def capture_req(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        captured.append(req)
        return _ok_response({"pointer": None})

    cache_path = worker_env["cache_dir"] / "listener-s1.json"
    request_data = {"text": "hello", "project_ref": "my-proj", "operated": []}
    _run_worker(monkeypatch, tmp_path, request_data, cache_path, capture_req)

    assert len(captured) == 1
    req = captured[0]
    assert req.full_url == "https://kb.example.com/api/kb/listener"
    assert req.get_header("Authorization") == "Bearer test-key"
    assert req.get_header("Content-type") == "application/json"
    assert req.method == "POST"


def test_request_body_bytes(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request body is json.dumps(request_data).encode('utf-8')."""
    captured: list[urllib.request.Request] = []

    def capture_req(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        captured.append(req)
        return _ok_response({"pointer": None})

    cache_path = worker_env["cache_dir"] / "listener-s2.json"
    request_data = {
        "text": "test text",
        "project_ref": "proj-a",
        "operated": ["mcp:agent-gtd"],
    }
    _run_worker(monkeypatch, tmp_path, request_data, cache_path, capture_req)

    assert len(captured) == 1
    body_bytes = captured[0].data
    assert body_bytes is not None
    body = json.loads(body_bytes.decode("utf-8"))
    assert body == request_data


def test_request_trailing_slash_stripped(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing slash on PERSONAL_KB_URL is stripped before appending endpoint."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com/")
    captured: list[urllib.request.Request] = []

    def capture_req(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        captured.append(req)
        return _ok_response({"pointer": None})

    cache_path = worker_env["cache_dir"] / "listener-s3.json"
    req_data = {"text": "x", "project_ref": None, "operated": []}
    _run_worker(monkeypatch, tmp_path, req_data, cache_path, capture_req)
    assert captured[0].full_url == "https://kb.example.com/api/kb/listener"


# ---------------------------------------------------------------------------
# project_ref=null case
# ---------------------------------------------------------------------------


def test_project_ref_null_sent_in_body(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """project_ref=null is serialised as JSON null and sent correctly."""
    captured: list[urllib.request.Request] = []

    def capture_req(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        captured.append(req)
        return _ok_response({"pointer": None})

    cache_path = worker_env["cache_dir"] / "listener-null.json"
    req_data = {"text": "x" * 300, "project_ref": None, "operated": []}
    _run_worker(monkeypatch, tmp_path, req_data, cache_path, capture_req)
    body = json.loads(captured[0].data.decode("utf-8"))  # type: ignore[union-attr]
    assert body["project_ref"] is None


# ---------------------------------------------------------------------------
# Valid map -> cache merge (preserving whispered_map_ids)
# ---------------------------------------------------------------------------


def test_valid_map_written_to_cache(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 + valid map dict -> pending written, whispered_map_ids preserved."""
    cache_path = worker_env["cache_dir"] / "listener-s4.json"
    # Pre-populate cache with existing whispered ids
    existing_cache = {"pending": None, "whispered_map_ids": ["kb-00001", "kb-00002"]}
    cache_path.write_text(json.dumps(existing_cache), encoding="utf-8")

    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "hello", "project_ref": "proj", "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response(
            {"pointer": {"id": "kb-00099", "short_title": "NewMap", "long_title": "Details"}}
        ),
    )

    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"] == {
        "id": "kb-00099",
        "short_title": "NewMap",
        "long_title": "Details",
    }
    # whispered_map_ids must be preserved
    assert "kb-00001" in result["whispered_map_ids"]
    assert "kb-00002" in result["whispered_map_ids"]


def test_missing_long_title_coerced_to_empty(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing long_title in response map is coerced to ''."""
    cache_path = worker_env["cache_dir"] / "listener-lt1.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response(
            {"pointer": {"id": "kb-00010", "short_title": "NoLongTitle"}}
        ),
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"]["long_title"] == ""


def test_none_long_title_coerced_to_empty(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """long_title: null in response map is coerced to ''."""
    cache_path = worker_env["cache_dir"] / "listener-lt2.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response(
            {"pointer": {"id": "kb-00011", "short_title": "NullLT", "long_title": None}}
        ),
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"]["long_title"] == ""


def test_fresh_cache_gets_empty_whispered_ids(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When cache is missing, whispered_map_ids is [] after write."""
    cache_path = worker_env["cache_dir"] / "listener-fresh.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response(
            {"pointer": {"id": "kb-00020", "short_title": "Fresh", "long_title": ""}}
        ),
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["whispered_map_ids"] == []


# ---------------------------------------------------------------------------
# Null map / failure cases → cache untouched
# ---------------------------------------------------------------------------


def _assert_cache_untouched(cache_path: Path) -> None:
    assert not cache_path.exists(), f"Cache file was written but should be untouched: {cache_path}"


def test_null_map_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """null map -> no-op success; cache not written."""
    cache_path = worker_env["cache_dir"] / "listener-null2.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response({"pointer": None}),
    )
    _assert_cache_untouched(cache_path)


def test_non_dict_body_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-dict JSON response body → cache untouched."""
    cache_path = worker_env["cache_dir"] / "listener-ndb.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _MockResponse(json.dumps([1, 2, 3])),
    )
    _assert_cache_untouched(cache_path)


def test_missing_map_key_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response dict missing 'map' key → cache untouched."""
    cache_path = worker_env["cache_dir"] / "listener-nmk.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response({"map": {"id": "kb-1", "short_title": "x"}}),
    )
    _assert_cache_untouched(cache_path)


def test_invalid_map_empty_id_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map with empty id → cache untouched."""
    cache_path = worker_env["cache_dir"] / "listener-eid.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response({"pointer": {"id": "", "short_title": "Bad"}}),
    )
    _assert_cache_untouched(cache_path)


def test_invalid_map_missing_short_title_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map missing short_title → cache untouched."""
    cache_path = worker_env["cache_dir"] / "listener-mst.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response({"pointer": {"id": "kb-1"}}),
    )
    _assert_cache_untouched(cache_path)


def test_invalid_map_non_str_long_title_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map with non-str long_title (e.g. int 42) → cache untouched (failure)."""
    cache_path = worker_env["cache_dir"] / "listener-nltf.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response(
            {"pointer": {"id": "kb-1", "short_title": "Bad", "long_title": 42}}
        ),
    )
    _assert_cache_untouched(cache_path)


def test_http_error_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTPError → no cache write; worker exits 0."""
    cache_path = worker_env["cache_dir"] / "listener-httperr.json"

    def raise_http(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(
            url="https://kb.example.com/api/kb/listener",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    req_file = _make_request_file(tmp_path, {"text": "x", "project_ref": None, "operated": []})
    monkeypatch.setattr(urllib.request, "urlopen", raise_http)
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()  # must not raise
    _assert_cache_untouched(cache_path)


def test_timeout_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TimeoutError → no cache write; worker exits 0."""
    cache_path = worker_env["cache_dir"] / "listener-timeout.json"

    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise TimeoutError("timed out")

    req_file = _make_request_file(tmp_path, {"text": "x", "project_ref": None, "operated": []})
    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()
    _assert_cache_untouched(cache_path)


def test_bad_json_response_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON response body → no cache write; worker exits 0."""
    cache_path = worker_env["cache_dir"] / "listener-badjson.json"
    req_file = _make_request_file(tmp_path, {"text": "x", "project_ref": None, "operated": []})
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _MockResponse("not json {{{"))
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()
    _assert_cache_untouched(cache_path)


def test_non_dict_map_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """map value is a non-dict (e.g. string) → cache untouched."""
    cache_path = worker_env["cache_dir"] / "listener-strmap.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "project_ref": None, "operated": []},
        cache_path,
        lambda *a, **kw: _ok_response({"pointer": "not-a-dict"}),
    )
    _assert_cache_untouched(cache_path)


# ---------------------------------------------------------------------------
# Tmp file always deleted
# ---------------------------------------------------------------------------


def test_tmp_file_deleted_on_success(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request tmp file is deleted after a successful run."""
    req_file = _make_request_file(tmp_path, {"text": "x", "project_ref": None, "operated": []})
    cache_path = worker_env["cache_dir"] / "listener-td1.json"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _ok_response({"pointer": None}))
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()
    assert not req_file.exists()


def test_tmp_file_deleted_on_http_error(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request tmp file is deleted even when urlopen raises."""
    req_file = _make_request_file(tmp_path, {"text": "x", "project_ref": None, "operated": []})
    cache_path = worker_env["cache_dir"] / "listener-td2.json"

    def raise_err(*a: object, **kw: object) -> None:
        raise TimeoutError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", raise_err)
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()
    assert not req_file.exists()


def test_tmp_file_deleted_on_bad_json_response(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request tmp file is deleted even when response JSON is bad."""
    req_file = _make_request_file(tmp_path, {"text": "x", "project_ref": None, "operated": []})
    cache_path = worker_env["cache_dir"] / "listener-td3.json"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _MockResponse("bad"))
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()
    assert not req_file.exists()


# ---------------------------------------------------------------------------
# Always exit 0 (main() never raises)
# ---------------------------------------------------------------------------


def test_main_never_raises_on_missing_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() with no argv silently exits 0."""
    monkeypatch.setattr(sys, "argv", ["listener_worker"])
    listener_worker.main()  # must not raise


def test_main_never_raises_on_missing_tmp_file(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() with a missing tmp file silently exits 0."""
    cache_path = worker_env["cache_dir"] / "listener-missing.json"
    monkeypatch.setattr(sys, "argv", ["w", str(tmp_path / "no_such_file.json"), str(cache_path)])
    listener_worker.main()  # must not raise
