"""KB backend package — selects Local or HTTP backend at call time.

Usage::

    from personal_kb.backend import create_backend, Backend, LocalBackend, HttpBackend

    backend = create_backend(kb=kb_instance)  # local if PERSONAL_KB_URL unset
    backend = create_backend()                 # HTTP if PERSONAL_KB_URL is set

:func:`create_backend` reads ``PERSONAL_KB_URL`` and ``PERSONAL_KB_API_KEY``
from :data:`os.environ` **at call time** — not at import time — so tests can
monkeypatch the env after importing this module and still get the correct
backend type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_core.knowledge_base import KnowledgeBase

from personal_kb.backend.http import BackendHttpError, HttpBackend
from personal_kb.backend.local import LocalBackend
from personal_kb.backend.protocol import Backend


def create_backend(kb: KnowledgeBase | None = None) -> LocalBackend | HttpBackend:
    """Return a backend based on environment variables.

    Resolution order:

    1. If ``PERSONAL_KB_URL`` is set (and non-empty): return
       :class:`HttpBackend` pointing at that URL with the API key from
       ``PERSONAL_KB_API_KEY``.  The ``kb`` argument is ignored.
    2. Otherwise: return :class:`LocalBackend` wrapping *kb*.

    Reads the env **at call time** — not at module import time — so
    monkeypatching env vars in tests works correctly.
    """
    from personal_kb.config import get_personal_kb_api_key, get_personal_kb_url

    url = get_personal_kb_url()
    if url:
        api_key = get_personal_kb_api_key() or ""
        return HttpBackend(base_url=url, api_key=api_key)

    if kb is None:
        raise ValueError(
            "create_backend() requires a KnowledgeBase instance when PERSONAL_KB_URL is not set."
        )
    return LocalBackend(kb)


__all__ = [
    "Backend",
    "BackendHttpError",
    "HttpBackend",
    "LocalBackend",
    "create_backend",
]
