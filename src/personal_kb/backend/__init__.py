"""KB backend package — HTTP backend over the personal-kb service.

Usage::

    from personal_kb.backend import create_backend, Backend, HttpBackend

    backend = create_backend()  # HTTP backend pointing at PERSONAL_KB_URL

:func:`create_backend` reads ``PERSONAL_KB_URL`` and ``PERSONAL_KB_API_KEY``
from :data:`os.environ` **at call time** — not at import time — so tests can
monkeypatch the env after importing this module and still get the correct
backend. An unset ``PERSONAL_KB_URL`` means the local daemon default.
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

    The URL comes from ``PERSONAL_KB_URL`` (unset means the local default).
    A remote URL with no ``PERSONAL_KB_API_KEY`` raises :class:`ValueError`.

    Reads the env **at call time** — not at module import time — so
    monkeypatching env vars in tests works correctly. The *kb* argument is
    vestigial (retained so existing call-sites that still pass a
    ``KnowledgeBase`` remain valid) and is ignored.
    """
    from personal_kb.config import get_personal_kb_api_key, get_personal_kb_url

    url = get_personal_kb_url()
    api_key = get_personal_kb_api_key()
    if api_key is None:
        raise ValueError(
            f"PERSONAL_KB_API_KEY is required for the non-local PERSONAL_KB_URL {url!r}."
        )
    return HttpBackend(base_url=url, api_key=api_key)


__all__ = [
    "Backend",
    "BackendHttpError",
    "HttpBackend",
    "create_backend",
]
