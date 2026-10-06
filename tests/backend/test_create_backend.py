"""Tests for the create_backend() factory — call-time env resolution."""

from unittest.mock import MagicMock

import pytest

from personal_kb.config import LOCAL_KB_API_KEY, LOCAL_KB_URL


def _clear(monkeypatch):
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)


def test_create_backend_unset_url_uses_local_default(monkeypatch):
    """Unset URL + key means the local daemon URL with the no-auth sentinel."""
    _clear(monkeypatch)
    from personal_kb.backend import create_backend

    backend = create_backend()
    assert backend._base_url.rstrip("/") == LOCAL_KB_URL
    assert backend._api_key == LOCAL_KB_API_KEY


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
    """An empty PERSONAL_KB_URL behaves exactly like unset."""
    _clear(monkeypatch)
    monkeypatch.setenv("PERSONAL_KB_URL", "")

    from personal_kb.backend import create_backend

    backend = create_backend(MagicMock())
    assert backend._base_url.rstrip("/") == LOCAL_KB_URL


def test_create_backend_env_read_at_call_time(monkeypatch):
    """Env is read at call time — monkeypatching after import still works."""
    from personal_kb.backend import create_backend

    _clear(monkeypatch)
    assert create_backend()._base_url.rstrip("/") == LOCAL_KB_URL

    monkeypatch.setenv("PERSONAL_KB_URL", "http://127.0.0.1:9999")
    assert create_backend()._base_url.rstrip("/") == "http://127.0.0.1:9999"


def test_create_backend_remote_without_key_fails_loudly(monkeypatch):
    """A remote URL with no key must NOT get the local sentinel."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "")

    from personal_kb.backend import create_backend

    with pytest.raises(ValueError, match="PERSONAL_KB_API_KEY"):
        create_backend()


def test_create_backend_kb_ignored_in_http_mode(monkeypatch):
    """The kb argument is silently ignored."""
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "k")

    from personal_kb.backend import create_backend
    from personal_kb.backend.http import HttpBackend

    backend = create_backend(MagicMock())
    assert isinstance(backend, HttpBackend)
