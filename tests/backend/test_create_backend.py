"""Tests for the create_backend() factory — call-time env resolution."""

from unittest.mock import MagicMock

import pytest


def test_create_backend_raises_when_no_url(monkeypatch):
    """create_backend() raises ValueError when PERSONAL_KB_URL is unset.

    The in-process local backend has been removed, so a URL is always
    required — with or without a ``kb`` argument.
    """
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    from personal_kb.backend import create_backend

    # No kb argument
    with pytest.raises(ValueError, match="PERSONAL_KB_URL"):
        create_backend()

    # Even with a kb argument (now vestigial)
    with pytest.raises(ValueError, match="PERSONAL_KB_URL"):
        create_backend(MagicMock())


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
    """An empty PERSONAL_KB_URL string is treated the same as unset (raises)."""
    monkeypatch.setenv("PERSONAL_KB_URL", "")

    from personal_kb.backend import create_backend

    with pytest.raises(ValueError, match="PERSONAL_KB_URL"):
        create_backend(MagicMock())


def test_create_backend_env_read_at_call_time(monkeypatch):
    """Env is read at call time — monkeypatching after import still works."""
    # Import first, then patch env
    import personal_kb.backend as backend_pkg  # noqa: F401
    from personal_kb.backend import create_backend

    # Start: no URL → raises
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    with pytest.raises(ValueError, match="PERSONAL_KB_URL"):
        create_backend(MagicMock())

    # Now set URL → HTTP (env read at call time)
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.test")
    from personal_kb.backend.http import HttpBackend

    b2 = create_backend()
    assert isinstance(b2, HttpBackend)

    # Remove again → raises
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    with pytest.raises(ValueError, match="PERSONAL_KB_URL"):
        create_backend(MagicMock())


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
