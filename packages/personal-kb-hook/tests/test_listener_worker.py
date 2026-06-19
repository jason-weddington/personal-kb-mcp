"""Tests for personal_kb_hook.listener_worker (P2: roster fan-out).

All network access is mocked via monkeypatch (same shim pattern as
tests/test_http_index.py). Zero real HTTP calls.

Covers the P2 fan-out behaviour:
- Single-entry roster (the legacy fallback path) produces ONE POST with
  byte-identical headers and body.
- Multi-KB roster fans out sequentially — one POST per entry.url in roster
  order.
- Per-KB failure isolation: a failure for any single KB never aborts the
  loop; surviving KBs' pointers still flow into the cache.
- Suppress-only client-side arbitration: drop nulls, title-dedup (with
  ``str.strip().casefold()`` normalisation), one-per-KB cap; tie-break
  order is source_label-in-roster → 'personal' → first roster entry.
- Cache schema is the (label, id)-provenance shape: ``pending`` is a list
  of per-KB pointer dicts ``{label, id, short_title, long_title}`` (≤1 per
  label); ``whispered_map_ids`` is a list of ``[label, id]`` lists with
  tolerant legacy bare-id back-parse.
- Empty roster: ZERO POSTs, tmp deleted, cache untouched, exit 0.
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

from personal_kb_hook import listener_worker, roster

if TYPE_CHECKING:
    from pathlib import Path


def _stub_roster(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[tuple[str, str, str]],
) -> None:
    """Pin the value returned by listener_worker.roster.load_roster()."""
    kb_entries = [roster.KbEntry(label=label, url=url, key=key) for label, url, key in entries]
    monkeypatch.setattr(listener_worker.roster, "load_roster", lambda: kb_entries)


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
    """Isolate env vars and HOME for cache path expansion.

    Pins ``HOME`` + ``XDG_CONFIG_HOME`` under ``tmp_path`` so any stray
    ``kbs.json`` on the build machine cannot leak in. The legacy env
    fallback (``PERSONAL_KB_URL`` + ``PERSONAL_KB_API_KEY``) drives the
    single-KB byte-identical path; tests that need a multi-KB roster call
    :func:`_stub_roster` explicitly.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
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
    """200 + valid pointer -> per-KB pending list written, whispered_map_ids preserved.

    Pre-populated whispered_map_ids in the legacy bare-id form are
    back-parsed tolerantly to ``[_LEGACY_LABEL, id]`` pairs.
    """
    cache_path = worker_env["cache_dir"] / "listener-s4.json"
    # Pre-populate cache with existing whispered ids in the LEGACY bare-id form.
    existing_cache = {"pending": [], "whispered_map_ids": ["kb-00001", "kb-00002"]}
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
    # P2 schema: pending is a LIST of per-KB pointer dicts (≤1 per label).
    # Single-KB legacy roster always uses label='personal'.
    assert result["pending"] == [
        {
            "label": "personal",
            "id": "kb-00099",
            "short_title": "NewMap",
            "long_title": "Details",
        }
    ]
    # Legacy bare-id whispered_map_ids back-parsed to [label, id] pairs.
    assert ["personal", "kb-00001"] in result["whispered_map_ids"]
    assert ["personal", "kb-00002"] in result["whispered_map_ids"]


def test_missing_long_title_coerced_to_empty(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing long_title in response pointer is coerced to ''."""
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
    assert result["pending"][0]["long_title"] == ""
    assert result["pending"][0]["label"] == "personal"


def test_none_long_title_coerced_to_empty(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """long_title: null in response pointer is coerced to ''."""
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
    assert result["pending"][0]["long_title"] == ""
    assert result["pending"][0]["label"] == "personal"


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
    assert result["pending"] == [
        {
            "label": "personal",
            "id": "kb-00020",
            "short_title": "Fresh",
            "long_title": "",
        }
    ]


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


# ---------------------------------------------------------------------------
# P2: roster fan-out — multi-KB sequential POST per entry.url
# ---------------------------------------------------------------------------


def test_multi_kb_fan_out_sequential_posts(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-KB roster → one POST per entry.url, in roster order, each gets the body."""
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )

    captured: list[urllib.request.Request] = []

    def capture_req(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        captured.append(req)
        return _ok_response({"pointer": None})

    cache_path = worker_env["cache_dir"] / "listener-fan.json"
    request_data = {
        "text": "hello",
        "cwd_project": "personal-kb",
        "operating": [],
        "source_label": "personal-kb",
    }
    _run_worker(monkeypatch, tmp_path, request_data, cache_path, capture_req)

    assert len(captured) == 2
    # Order matches roster order.
    assert captured[0].full_url == "https://personal.kb/api/kb/listener"
    assert captured[1].full_url == "https://team.kb/api/kb/listener"
    # Distinct Bearer tokens.
    assert captured[0].get_header("Authorization") == "Bearer p-secret"
    assert captured[1].get_header("Authorization") == "Bearer t-secret"
    # Both posts carry the same body (json.dumps(request_data).encode('utf-8')).
    assert captured[0].data == captured[1].data
    body = json.loads(captured[0].data.decode("utf-8"))  # type: ignore[union-attr]
    assert body == request_data
    # Both POSTs returned null → no cache write.
    _assert_cache_untouched(cache_path)


def test_per_kb_failure_isolation_second_pointer_still_cached(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First KB raises TimeoutError; the second KB's valid pointer still reaches the cache."""
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        if "personal.kb" in req.full_url:
            raise TimeoutError("personal kb timed out")
        return _ok_response(
            {"pointer": {"id": "kb-team-1", "short_title": "TeamMap", "long_title": "Team"}}
        )

    cache_path = worker_env["cache_dir"] / "listener-isol.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
        cache_path,
        routed_urlopen,
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"] == [
        {
            "label": "team",
            "id": "kb-team-1",
            "short_title": "TeamMap",
            "long_title": "Team",
        }
    ]


def test_title_dedup_tie_break_keeps_personal_winner(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two KBs return the SAME short_title → only the tie-break winner ('personal') survives.

    source_label is a project_ref ('personal-kb'), NOT a roster label, so
    step (1) does not match; step (2) selects the 'personal' KB.
    """
    _stub_roster(
        monkeypatch,
        [
            ("team", "https://team.kb/", "t-secret"),
            ("personal", "https://personal.kb/", "p-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        if "personal.kb" in req.full_url:
            return _ok_response(
                {
                    "pointer": {
                        "id": "kb-personal-7",
                        "short_title": "Auth",
                        "long_title": "Personal flow",
                    }
                }
            )
        return _ok_response(
            {
                "pointer": {
                    "id": "kb-team-7",
                    "short_title": "Auth",
                    "long_title": "Team flow",
                }
            }
        )

    cache_path = worker_env["cache_dir"] / "listener-tie.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {
            "text": "x",
            "cwd_project": "personal-kb",
            "operating": [],
            "source_label": "personal-kb",
        },
        cache_path,
        routed_urlopen,
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"] == [
        {
            "label": "personal",
            "id": "kb-personal-7",
            "short_title": "Auth",
            "long_title": "Personal flow",
        }
    ]


def test_title_dedup_normalizes_with_strip_casefold(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Title normalisation = ``str.strip().casefold()`` — case + whitespace insensitive."""
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        if "personal.kb" in req.full_url:
            return _ok_response(
                {"pointer": {"id": "kb-p-1", "short_title": "  Auth ", "long_title": ""}}
            )
        return _ok_response({"pointer": {"id": "kb-t-1", "short_title": "auth", "long_title": ""}})

    cache_path = worker_env["cache_dir"] / "listener-norm.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
        cache_path,
        routed_urlopen,
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    # Only one survives; 'personal' wins per the tie-break rule.
    assert len(result["pending"]) == 1
    assert result["pending"][0]["label"] == "personal"


def test_distinct_titles_per_kb_both_survive_one_per_label(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two KBs return DIFFERENT short_titles → both survive (one per label)."""
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        if "personal.kb" in req.full_url:
            return _ok_response(
                {
                    "pointer": {
                        "id": "kb-p-1",
                        "short_title": "PersonalThing",
                        "long_title": "",
                    }
                }
            )
        return _ok_response(
            {"pointer": {"id": "kb-t-1", "short_title": "TeamThing", "long_title": ""}}
        )

    cache_path = worker_env["cache_dir"] / "listener-distinct.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
        cache_path,
        routed_urlopen,
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    labels = sorted(p["label"] for p in result["pending"])
    assert labels == ["personal", "team"]
    # One per label (defensive cap is no-op here, but cardinality must still be 2).
    assert len(result["pending"]) == 2


def test_label_id_provenance_in_whispered_map_ids(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing [label, id] whispered_map_ids are preserved verbatim across a write."""
    _stub_roster(
        monkeypatch,
        [("personal", "https://personal.kb/", "p-secret")],
    )

    cache_path = worker_env["cache_dir"] / "listener-prov.json"
    cache_path.write_text(
        json.dumps(
            {
                "pending": [],
                "whispered_map_ids": [
                    ["personal", "kb-prev-1"],
                    ["team", "kb-prev-2"],
                ],
            }
        ),
        encoding="utf-8",
    )
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
        cache_path,
        lambda *a, **kw: _ok_response(
            {"pointer": {"id": "kb-new", "short_title": "New", "long_title": ""}}
        ),
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert ["personal", "kb-prev-1"] in result["whispered_map_ids"]
    assert ["team", "kb-prev-2"] in result["whispered_map_ids"]


def test_empty_roster_no_op(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty roster → ZERO POSTs, tmp file deleted, cache untouched, exit 0."""
    _stub_roster(monkeypatch, [])

    call_count = {"n": 0}

    def counting_urlopen(req: object, timeout: float | None = None) -> _MockResponse:
        call_count["n"] += 1
        return _ok_response({"pointer": None})

    cache_path = worker_env["cache_dir"] / "listener-empty.json"
    req_file = _make_request_file(
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
    )
    monkeypatch.setattr(urllib.request, "urlopen", counting_urlopen)
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()

    assert call_count["n"] == 0
    assert not req_file.exists(), "tmp file must still be deleted on empty-roster path"
    _assert_cache_untouched(cache_path)


def test_first_roster_entry_tie_break_when_no_personal(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither source_label nor 'personal' is in the roster, the first entry wins."""
    _stub_roster(
        monkeypatch,
        [
            ("alpha", "https://alpha.kb/", "a-secret"),
            ("beta", "https://beta.kb/", "b-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        return _ok_response(
            {"pointer": {"id": "kb-same", "short_title": "SameTitle", "long_title": ""}}
        )

    cache_path = worker_env["cache_dir"] / "listener-first.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {
            "text": "x",
            "cwd_project": "some-project",
            "operating": [],
            "source_label": "some-project",
        },
        cache_path,
        routed_urlopen,
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"] == [
        {
            "label": "alpha",
            "id": "kb-same",
            "short_title": "SameTitle",
            "long_title": "",
        }
    ]


def test_all_pointers_null_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All KBs return null pointer → arbitration produces nothing → cache untouched."""
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )
    cache_path = worker_env["cache_dir"] / "listener-allnull.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
        cache_path,
        lambda *a, **kw: _ok_response({"pointer": None}),
    )
    _assert_cache_untouched(cache_path)


def test_all_kbs_fail_no_cache_write(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every KB's POST raises → arbitration empty → cache untouched; main exits 0."""
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )

    def raise_err(*a: object, **kw: object) -> None:
        raise urllib.error.URLError("everything's down")

    cache_path = worker_env["cache_dir"] / "listener-allfail.json"
    req_file = _make_request_file(
        tmp_path,
        {"text": "x", "cwd_project": None, "operating": [], "source_label": None},
    )
    monkeypatch.setattr(urllib.request, "urlopen", raise_err)
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])
    listener_worker.main()  # must not raise
    _assert_cache_untouched(cache_path)
    assert not req_file.exists()


def test_source_label_matching_roster_wins_tie_break(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If source_label EQUALS a roster label, that KB wins the title-collision tie-break.

    (Today this requires a contrived setup since cli.py sets source_label
    to project_ref_stop, not to a KB label — but the AC pins the rule.)
    """
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        if "personal.kb" in req.full_url:
            return _ok_response(
                {"pointer": {"id": "kb-p", "short_title": "Same", "long_title": "p"}}
            )
        return _ok_response({"pointer": {"id": "kb-t", "short_title": "Same", "long_title": "t"}})

    cache_path = worker_env["cache_dir"] / "listener-srcwin.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        # source_label = 'team' matches a KB label.
        {"text": "x", "cwd_project": None, "operating": [], "source_label": "team"},
        cache_path,
        routed_urlopen,
    )
    result = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["pending"] == [
        {"label": "team", "id": "kb-t", "short_title": "Same", "long_title": "t"}
    ]


# ---------------------------------------------------------------------------
# Whisper-debug RUN block (local plaintext log)
# ---------------------------------------------------------------------------


def test_whisper_debug_run_block_written(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-KB roster -> RUN block: header, transcript, per-KB lines, outcome.

    Covers the three per-KB cases in one go:
    - the 'team' KB returns a non-null pointer WITH a server reason
      -> ``team -> kb-team-1 (matched ...)``
    - the 'personal' KB returns a null pointer WITH a server reason
      -> ``personal -> none (no-injection: ...)``
    - the 'alpha' KB's POST raises (transport-error)
      -> ``alpha -> none (transport-error)``
    Outcome line: ``=> WHISPER kb-team-1 next turn`` (post-arbitration).
    """
    _stub_roster(
        monkeypatch,
        [
            ("personal", "https://personal.kb/", "p-secret"),
            ("team", "https://team.kb/", "t-secret"),
            ("alpha", "https://alpha.kb/", "a-secret"),
        ],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        if "personal.kb" in req.full_url:
            return _ok_response(
                {
                    "pointer": None,
                    "reason": "no-injection: LLM unavailable",
                }
            )
        if "team.kb" in req.full_url:
            return _ok_response(
                {
                    "pointer": {
                        "id": "kb-team-1",
                        "short_title": "TeamMap",
                        "long_title": "Team",
                    },
                    "reason": "matched kb-team-1 (unanimous 3/3)",
                }
            )
        # alpha.kb -> transport-error
        raise TimeoutError("alpha down")

    cache_path = worker_env["cache_dir"] / "listener-debug.json"
    session_id = "sess-debug-1"
    _run_worker(
        monkeypatch,
        tmp_path,
        {
            "text": "which machine runs traefik in the home lab",
            "cwd_project": "home-lab",
            "operating": ["mcp:agent-gtd", "mcp:personal-kb"],
            "source_label": "home-lab",
            "session_id": session_id,
        },
        cache_path,
        routed_urlopen,
    )

    debug_log = worker_env["cache_dir"] / f"whisper-debug-{session_id}.log"
    assert debug_log.exists(), "whisper-debug log must be written when session_id is set"
    content = debug_log.read_text(encoding="utf-8")

    # Header carries cwd_project + operating manifest.
    assert "listener run" in content
    assert "cwd_project=home-lab" in content
    # Defensive: operating manifest rendered as [<comma-joined>].
    assert "operating=[mcp:agent-gtd,mcp:personal-kb]" in content

    # Transcript excerpt is the request text, whitespace-collapsed.
    assert "transcript: which machine runs traefik in the home lab" in content

    # Per-KB lines in roster order: personal -> none with server reason;
    # team -> kb-team-1 with server reason; alpha -> none with transport-error.
    assert "personal -> none (no-injection: LLM unavailable)" in content
    assert "team -> kb-team-1 (matched kb-team-1 (unanimous 3/3))" in content
    assert "alpha -> none (transport-error)" in content

    # Outcome reflects post-arbitration winners: kb-team-1 survives.
    assert "=> WHISPER kb-team-1 next turn" in content


def test_whisper_debug_run_block_no_whisper_outcome(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All KBs null -> outcome is a single ``=> no whisper`` line."""
    _stub_roster(
        monkeypatch,
        [("personal", "https://personal.kb/", "p-secret")],
    )

    def routed_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        return _ok_response(
            {
                "pointer": None,
                "reason": "no-injection: no candidates from retrieval",
            }
        )

    cache_path = worker_env["cache_dir"] / "listener-nowhisper.json"
    session_id = "sess-nowhisper"
    _run_worker(
        monkeypatch,
        tmp_path,
        {
            "text": "x" * 250,
            "cwd_project": None,
            "operating": [],
            "source_label": None,
            "session_id": session_id,
        },
        cache_path,
        routed_urlopen,
    )

    debug_log = worker_env["cache_dir"] / f"whisper-debug-{session_id}.log"
    content = debug_log.read_text(encoding="utf-8")
    assert "personal -> none (no-injection: no candidates from retrieval)" in content
    assert "=> no whisper" in content
    # No WHISPER line on the no-winner path.
    assert "=> WHISPER" not in content


def test_whisper_debug_run_block_silent_when_unwritable(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure (debug-log dir replaced by a regular file) does NOT raise.

    The worker still completes normally — cache is written, exit is clean.
    """
    _stub_roster(
        monkeypatch,
        [("personal", "https://personal.kb/", "p-secret")],
    )

    # Replace the cache_dir with a file so the debug-log path's parent
    # mkdir + open both fail. The cache_path argument points at a separate
    # file under the same dir — also broken, but the worker swallows that
    # silently too (cache write inside _merge_into_cache uses tempfile +
    # os.replace, which also OSErrors quietly).
    cache_dir = worker_env["cache_dir"]
    # Remove the dir, replace with a file at the same path.
    import shutil

    shutil.rmtree(cache_dir)
    cache_dir.write_text("not a dir", encoding="utf-8")

    cache_path = cache_dir / "listener-unwritable.json"

    session_id = "sess-broken"
    req_file = tmp_path / "req-unwritable.json"
    req_file.write_text(
        json.dumps(
            {
                "text": "x" * 250,
                "cwd_project": None,
                "operating": [],
                "source_label": None,
                "session_id": session_id,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(
            {
                "pointer": {"id": "kb-1", "short_title": "X", "long_title": ""},
                "reason": "matched kb-1 (unanimous 3/3)",
            }
        ),
    )
    monkeypatch.setattr(sys, "argv", ["w", str(req_file), str(cache_path)])

    # MUST NOT raise.
    listener_worker.main()


def test_whisper_debug_run_block_skipped_when_no_session_id(
    worker_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No session_id in request -> no whisper-debug log file is created."""
    _stub_roster(
        monkeypatch,
        [("personal", "https://personal.kb/", "p-secret")],
    )

    cache_path = worker_env["cache_dir"] / "listener-nosid.json"
    _run_worker(
        monkeypatch,
        tmp_path,
        # session_id intentionally omitted
        {"text": "x" * 250, "cwd_project": None, "operating": [], "source_label": None},
        cache_path,
        lambda *a, **kw: _ok_response({"pointer": None, "reason": "no-injection: LLM unavailable"}),
    )

    # No file should be present matching whisper-debug-*.log.
    debug_logs = list(worker_env["cache_dir"].glob("whisper-debug-*.log"))
    assert debug_logs == []
