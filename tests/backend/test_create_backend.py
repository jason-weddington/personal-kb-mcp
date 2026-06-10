"""Tests for the create_backend() factory — call-time env resolution."""

from unittest.mock import MagicMock

import pytest


def test_create_backend_returns_local_when_no_url(monkeypatch):
    """create_backend() returns LocalBackend when PERSONAL_KB_URL is unset."""
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    from personal_kb.backend import create_backend
    from personal_kb.backend.local import LocalBackend as _LocalBackend

    kb = MagicMock()
    backend = create_backend(kb)
    assert isinstance(backend, _LocalBackend)
    assert backend.is_remote is False


def test_create_backend_returns_http_when_url_set(monkeypatch):
    """create_backend() returns HttpBackend when PERSONAL_KB_URL is set."""
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-key")

    from personal_kb.backend import create_backend
    from personal_kb.backend.http import HttpBackend as _HttpBackend

    backend = create_backend()
    assert isinstance(backend, _HttpBackend)
    assert backend.is_remote is True


def test_create_backend_empty_url_treated_as_unset(monkeypatch):
    """An empty PERSONAL_KB_URL string is treated the same as unset."""
    monkeypatch.setenv("PERSONAL_KB_URL", "")

    from personal_kb.backend import create_backend
    from personal_kb.backend.local import LocalBackend

    kb = MagicMock()
    backend = create_backend(kb)
    assert isinstance(backend, LocalBackend)


def test_create_backend_raises_when_no_url_and_no_kb(monkeypatch):
    """create_backend() raises ValueError when URL unset and no kb provided."""
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)

    from personal_kb.backend import create_backend

    with pytest.raises(ValueError, match="requires a KnowledgeBase"):
        create_backend()


def test_create_backend_env_read_at_call_time(monkeypatch):
    """Env is read at call time — monkeypatching after import still works."""
    # Import first, then patch env
    import personal_kb.backend as backend_pkg  # noqa: F401
    from personal_kb.backend import create_backend

    # Start: no URL → local
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    kb = MagicMock()
    from personal_kb.backend.local import LocalBackend

    b1 = create_backend(kb)
    assert isinstance(b1, LocalBackend)

    # Now set URL → HTTP (env read at call time)
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.test")
    from personal_kb.backend.http import HttpBackend

    b2 = create_backend()
    assert isinstance(b2, HttpBackend)

    # Remove again → local
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    b3 = create_backend(kb)
    assert isinstance(b3, LocalBackend)


def test_create_backend_api_key_empty_string_yields_empty(monkeypatch):
    """Empty PERSONAL_KB_API_KEY is normalised to '' (not None) for the bearer token."""
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "")

    from personal_kb.backend import create_backend
    from personal_kb.backend.http import HttpBackend

    backend = create_backend()
    assert isinstance(backend, HttpBackend)
    # The bearer header is set with the api_key
    assert backend._api_key == ""


def test_create_backend_kb_ignored_in_http_mode(monkeypatch):
    """The kb argument is silently ignored when PERSONAL_KB_URL is set."""
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.example.com")

    from personal_kb.backend import create_backend
    from personal_kb.backend.http import HttpBackend

    kb = MagicMock()
    backend = create_backend(kb)  # kb passed but should be ignored
    assert isinstance(backend, HttpBackend)
