"""KB backend package — HTTP backend over the personal-kb service.

Usage::

    from personal_kb.backend import create_backend, Backend, HttpBackend

    backend = create_backend()  # HTTP backend pointing at PERSONAL_KB_URL

:func:`create_backend` reads ``PERSONAL_KB_URL`` and ``PERSONAL_KB_API_KEY``
from :data:`os.environ` **at call time** — not at import time — so tests can
monkeypatch the env after importing this module and still get the correct
backend. ``PERSONAL_KB_URL`` is required; the in-process local backend has
been removed, so an unset URL raises :class:`ValueError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_core.knowledge_base import KnowledgeBase

from personal_kb.backend.http import BackendHttpError, HttpBackend
from personal_kb.backend.protocol import Backend


def _assert_http_backend_conforms(backend: HttpBackend) -> Backend:
    """mypy-only guard: fails `mypy src` the moment HttpBackend drifts from Backend."""
    return backend


def create_backend(kb: KnowledgeBase | None = None) -> HttpBackend:
    """Return an :class:`HttpBackend` based on environment variables.

    Resolution:

    1. If ``PERSONAL_KB_URL`` is set (and non-empty): return
       :class:`HttpBackend` pointing at that URL with the API key from
       ``PERSONAL_KB_API_KEY``.
    2. Otherwise: raise :class:`ValueError` — the in-process local backend
       has been removed, so a URL is always required.

    Reads the env **at call time** — not at module import time — so
    monkeypatching env vars in tests works correctly. The *kb* argument is
    vestigial (retained so existing call-sites that still pass a
    ``KnowledgeBase`` remain valid) and is ignored.
    """
    from personal_kb.config import get_personal_kb_api_key, get_personal_kb_url

    url = get_personal_kb_url()
    if url:
        api_key = get_personal_kb_api_key() or ""
        return HttpBackend(base_url=url, api_key=api_key)

    raise ValueError(
        "create_backend() requires PERSONAL_KB_URL; the local in-process backend has been removed."
    )


__all__ = [
    "Backend",
    "BackendHttpError",
    "HttpBackend",
    "create_backend",
]
