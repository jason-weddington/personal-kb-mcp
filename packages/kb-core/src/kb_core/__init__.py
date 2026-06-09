"""``kb_core`` — channel-agnostic engine for personal-kb.

This package will host the reusable engine (hybrid search, graph RAG,
store, ingest, agentic query) that the MCP server, the web explorer, and
any future consumer share. The current MCP server (``personal_kb``)
becomes one consumer; later extraction waves move logic modules here one
at a time.

Wave 1 (this commit) only scaffolds the package and ships ``KbConfig``
plus the import-purity guard. No logic has been moved yet.

Two invariants are enforced by ``tests/test_import_purity.py`` and MUST
hold for every later wave:

* ``kb_core`` reads NO ``os.environ`` / ``os.getenv`` at import or runtime.
  Config is passed explicitly via :class:`kb_core.config.KbConfig`.
* A bare ``import kb_core`` does NOT pull in MCP/HTTP-server stacks
  (``fastmcp``, ``fastapi``, ``uvicorn``) or provider SDKs (``anthropic``,
  ``boto3``). Those are optional extras activated by the consumer.

The dist name is ``kb-core`` and the import name is ``kb_core`` (locked).
"""

from kb_core.config import (
    AgenticConfig,
    AnthropicProviderConfig,
    Attribution,
    BedrockProviderConfig,
    DatabaseConfig,
    EmbeddingConfig,
    IngestConfig,
    KbConfig,
    OllamaProviderConfig,
    PostgresConfig,
    ProviderConfig,
    ProviderRoleConfig,
    SqliteConfig,
)

__all__ = [
    "AgenticConfig",
    "AnthropicProviderConfig",
    "Attribution",
    "BedrockProviderConfig",
    "DatabaseConfig",
    "EmbeddingConfig",
    "IngestConfig",
    "KbConfig",
    "OllamaProviderConfig",
    "PostgresConfig",
    "ProviderConfig",
    "ProviderRoleConfig",
    "SqliteConfig",
]
