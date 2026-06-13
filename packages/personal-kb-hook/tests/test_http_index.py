"""Tests for personal_kb_hook.http_index.load_index().

All network access is mocked via monkeypatch — no real HTTP calls.
urllib.request.urlopen is replaced inline with a callable; no third-party
mocking libraries (httpx/respx) are used.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from personal_kb_hook import http_index

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def http_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate KB_DB_PATH and clear both HTTP env vars."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    return {
        "db_path": db_path,
        "maps_index": db_path.parent / "maps_index.default.jsonl",
    }


REMOTE_PROJECT = "remote-proj"
REMOTE_MAPS = [{"id": "kb-remote-1", "short_title": "remote", "long_title": "Remote entry"}]


class _MockResponse:
    """Minimal context-manager response shim for urllib.request.urlopen."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_MockResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _ok_response(projects: list[dict[str, Any]]) -> _MockResponse:
    return _MockResponse(json.dumps({"projects": projects}))


# ---------------------------------------------------------------------------
# (a) Env gating — urlopen must never be called when either var is missing/empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,key",
    [
        # URL set + key unset (not in env at all)
        ("https://example.com", None),
        # URL unset (not in env at all) + key set
        (None, "mykey"),
        # URL='' (empty string) + key set
        ("", "mykey"),
        # URL set + key='' (empty string)
        ("https://example.com", ""),
        # both unset (not in env at all)
        (None, None),
    ],
    ids=[
        "url-set-key-unset",
        "url-unset-key-set",
        "url-empty-key-set",
        "url-set-key-empty",
        "both-unset",
    ],
)
def test_env_gating_no_urlopen(
    url: str | None,
    key: str | None,
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urlopen is never called when either env var is absent or empty; returns {}."""
    if url is not None:
        monkeypatch.setenv("PERSONAL_KB_URL", url)
    if key is not None:
        monkeypatch.setenv("PERSONAL_KB_API_KEY", key)

    def must_not_be_called(*args: object, **kwargs: object) -> _MockResponse:
        raise AssertionError("urlopen must not be called when env vars are missing/empty")

    monkeypatch.setattr(urllib.request, "urlopen", must_not_be_called)

    result = http_index.load_index()
    assert result == {}


# ---------------------------------------------------------------------------
# (b) Happy path: env set + 200 with valid body -> remote content returned
# ---------------------------------------------------------------------------


def test_happy_path_returns_remote_content(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid HTTP response with content is returned."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response([{"project_ref": REMOTE_PROJECT, "maps": REMOTE_MAPS}]),
    )

    result = http_index.load_index()
    assert REMOTE_PROJECT in result
    assert result[REMOTE_PROJECT][0]["id"] == "kb-remote-1"


# ---------------------------------------------------------------------------
# (c) Timeout -> returns empty index
# ---------------------------------------------------------------------------


def test_timeout_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A socket timeout returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    def raise_timeout(*args: object, **kwargs: object) -> _MockResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (d) Malformed JSON body -> returns empty index
# ---------------------------------------------------------------------------


def test_malformed_json_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-JSON response body returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _MockResponse("not json at all {{{"),
    )

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (e) Wrong top-level shape -> returns empty index
# ---------------------------------------------------------------------------


def test_wrong_shape_projects_not_list_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Body {"projects": "nope"} (non-list projects) returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _MockResponse(json.dumps({"projects": "nope"})),
    )

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (f) HTTPError 401 -> returns empty index
# ---------------------------------------------------------------------------


def test_http_error_401_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 401 HTTPError returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "bad-key")

    def raise_401(*args: object, **kwargs: object) -> _MockResponse:
        raise urllib.error.HTTPError(
            url="https://kb.example.com/api/kb/maps-index",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_401)

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (g) URL and Authorization header contract (incl. trailing-slash normalisation)
# ---------------------------------------------------------------------------


def test_http_request_contract_url_and_auth_header(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request URL has trailing slash stripped; Authorization header is exact."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com/")  # trailing slash
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "my-secret-key")

    captured: list[urllib.request.Request] = []

    def capture_req(
        req: urllib.request.Request,
        timeout: float | None = None,
    ) -> _MockResponse:
        captured.append(req)
        return _ok_response([])

    monkeypatch.setattr(urllib.request, "urlopen", capture_req)

    http_index.load_index()

    assert len(captured) == 1
    req = captured[0]
    assert req.full_url == "https://kb.example.com/api/kb/maps-index"
    assert req.get_header("Authorization") == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# (h) Partial-record tolerance: bad records skipped, good ones kept
# ---------------------------------------------------------------------------


def test_tolerance_skips_malformed_map_records(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed map records are skipped; valid ones are returned."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    projects = [
        {
            "project_ref": "my-proj",
            "maps": [
                # Valid record — must appear in output.
                {"id": "kb-1", "short_title": "good", "long_title": ""},
                # Empty id — must be skipped.
                {"id": "", "short_title": "bad-id", "long_title": "should be skipped"},
                # Non-dict map element — must be skipped.
                "oops-not-a-dict",
                # long_title is an int (42) — must be skipped, not coerced.
                {"id": "kb-2", "short_title": "bad-long-title", "long_title": 42},
                # missing short_title — must be skipped.
                {"id": "kb-3", "long_title": "no short_title"},
                # Another valid record.
                {"id": "kb-4", "short_title": "also-good", "long_title": "kept"},
            ],
        }
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index()
    assert "my-proj" in result
    ids = [m["id"] for m in result["my-proj"]]
    # Valid records kept:
    assert "kb-1" in ids
    assert "kb-4" in ids
    # Malformed records skipped:
    assert "kb-2" not in ids  # long_title=42 skipped
    assert "kb-3" not in ids  # missing short_title skipped
    # Non-dict element and empty-id entries produce no entry at all.
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# (i) Successful-but-empty response is authoritative — returns {}
# ---------------------------------------------------------------------------


def test_empty_response_is_authoritative_no_local_fallback(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 with {"projects": []} returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response([]),
    )

    result = http_index.load_index()
    assert result == {}


# ---------------------------------------------------------------------------
# Extra: socket.timeout also returns empty (variant of timeout test)
# ---------------------------------------------------------------------------


def test_socket_timeout_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TimeoutError (socket.timeout alias) returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    def raise_socket_timeout(*args: object, **kwargs: object) -> _MockResponse:
        raise TimeoutError("socket timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_socket_timeout)

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Extra: URL with non-http scheme returns empty (scheme validation)
# ---------------------------------------------------------------------------


def test_invalid_scheme_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-http(s) PERSONAL_KB_URL scheme silently returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "ftp://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    def must_not_be_called(*args: object, **kwargs: object) -> _MockResponse:
        raise AssertionError("urlopen must not be called for ftp:// scheme")

    monkeypatch.setattr(urllib.request, "urlopen", must_not_be_called)

    result = http_index.load_index()
    assert result == {}


# ---------------------------------------------------------------------------
# Extra: duplicate project_ref in response follows last-wins
# ---------------------------------------------------------------------------


def test_duplicate_project_ref_last_wins(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate project_ref entries in the response follow last-wins semantics."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    projects = [
        {
            "project_ref": "dup-proj",
            "maps": [{"id": "kb-first", "short_title": "first", "long_title": ""}],
        },
        {
            "project_ref": "dup-proj",
            "maps": [{"id": "kb-second", "short_title": "second", "long_title": ""}],
        },
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index()
    assert "dup-proj" in result
    ids = [m["id"] for m in result["dup-proj"]]
    # Last entry wins.
    assert ids == ["kb-second"]
    assert "kb-first" not in ids


# ---------------------------------------------------------------------------
# Extra: long_title None in remote response coerced to ""
# ---------------------------------------------------------------------------


def test_long_title_none_coerced_to_empty_string(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """long_title: null in the response is coerced to '' (not skipped)."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    projects = [
        {
            "project_ref": "null-lt-proj",
            "maps": [{"id": "kb-1", "short_title": "alpha", "long_title": None}],
        }
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index()
    assert "null-lt-proj" in result
    entry = result["null-lt-proj"][0]
    assert entry["id"] == "kb-1"
    assert entry["long_title"] == ""


# ---------------------------------------------------------------------------
# Extra: URLError (DNS / connection failure) -> returns empty index
# ---------------------------------------------------------------------------


def test_urlerror_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A URLError (DNS / connection failure) returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    def raise_urlerror(*args: object, **kwargs: object) -> _MockResponse:
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", raise_urlerror)

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Extra: non-JSON-object top-level (e.g. a JSON array) -> returns empty index
# ---------------------------------------------------------------------------


def test_json_array_top_level_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A top-level JSON array (not an object) returns {}."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _MockResponse(json.dumps([{"project_ref": "x", "maps": []}])),
    )

    result = http_index.load_index()
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Extra: MapEntry fields match the TypedDict exactly
# ---------------------------------------------------------------------------


def test_returned_entries_are_map_entry_typed(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned entries contain exactly the id/short_title/long_title keys."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")

    projects = [
        {
            "project_ref": "typed-proj",
            "maps": [{"id": "kb-1", "short_title": "s", "long_title": "l"}],
        }
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index()
    entry = result["typed-proj"][0]
    assert entry["id"] == "kb-1"
    assert entry["short_title"] == "s"
    assert entry["long_title"] == "l"
