"""Explicit, env-free configuration dataclasses for ``kb_core``.

``personal_kb.config`` is environment-driven (every tunable is a
``KB_*`` env var read at startup). ``kb_core`` reverses that: every
tunable becomes a typed field on a dataclass, and the consumer assembles
a :class:`KbConfig` and hands it to the engine. There are **zero**
``os.environ`` / ``os.getenv`` reads anywhere in this package — that is
enforced by ``tests/test_import_purity.py`` via an AST scan over the
``kb_core`` source tree.

The shape mirrors the env surface of ``personal_kb.config`` so the
mapping in the MCP server's startup wiring (later wave) is mechanical:

* :class:`DatabaseConfig` — discriminated ``SqliteConfig | PostgresConfig``
  (covers ``KB_DB_PATH``, ``KB_DATABASE_URL``, ``KB_PG_POOL_*``,
  ``KB_PG_IAM_AUTH``, ``KB_PG_REGION``, ``KB_EMBEDDING_DIM``).
* :class:`EmbeddingConfig` — Ollama embeddings (``KB_OLLAMA_URL``,
  ``KB_EMBEDDING_MODEL``, ``KB_OLLAMA_TIMEOUT``, ``KB_EMBEDDING_DIM``).
* :class:`ProviderConfig` — per-role (extraction / query / synthesis)
  LLM provider selection plus per-provider credentials/timeouts
  (``KB_EXTRACTION_PROVIDER``, ``KB_QUERY_PROVIDER``, ``KB_ANTHROPIC_*``,
  ``KB_BEDROCK_*``, ``KB_AWS_PROFILE``, ``KB_OLLAMA_MODEL``,
  ``KB_OLLAMA_LLM_TIMEOUT``).
* :class:`IngestConfig` — file ingestion + dedup + safety
  (``KB_INGEST_*``, ``KB_AGENTIC_INGEST``, ``KB_SKIP_SAFETY``).
* :class:`AgenticConfig` — agent loops
  (``KB_AGENTIC_QUERY``, ``KB_AGENTIC_SYNTHESIS``, ``KB_AGENTIC_MAX_CALLS``).
* :class:`Attribution` — entry attribution
  (``KB_CONTRIBUTOR``, ``KB_TEAM``).

Server-only env vars deliberately have no home here because they govern
the MCP server / web explorer / CLI hook, not the engine:
``KB_MANAGER``, ``KB_INSTANCE_ROLE``, ``KB_AUTO_EXPLORE``,
``KB_EXPLORE_PORT``, ``KB_LOG_LEVEL``. They stay in
``personal_kb.config``.

Defaults match the current ``personal_kb.config`` defaults so callers
that pass ``KbConfig()`` get today's behavior. Default paths are
intentionally **not** ``Path.expanduser()``-ed at construction time —
``Path.expanduser`` reads ``$HOME``, and we want construction to be a
pure value operation. The DB layer (later wave) calls ``expanduser`` at
use time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Default values are sourced from ``personal_kb.config`` so a bare
# ``KbConfig()`` is byte-identical to today's defaults. If you change
# either side, change both — there is no test enforcing this drift
# today (the env-driven module reads its defaults from string literals
# in argument positions, which is awkward to import).

_DEFAULT_DB_PATH = "~/.local/share/personal_kb/knowledge.db"
_DEFAULT_EMBEDDING_DIM = 1024
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
_DEFAULT_EMBEDDING_TIMEOUT = 10.0
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
_DEFAULT_ANTHROPIC_TIMEOUT = 30.0
_DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_BEDROCK_REGION = "us-east-1"
_DEFAULT_BEDROCK_TIMEOUT = 60.0
_DEFAULT_OLLAMA_LLM_MODEL = "qwen3:4b"
_DEFAULT_OLLAMA_LLM_TIMEOUT = 120.0
_DEFAULT_PG_POOL_MIN = 1
_DEFAULT_PG_POOL_MAX = 5
_DEFAULT_PG_REGION = "us-east-1"
_DEFAULT_INGEST_MAX_FILE_SIZE = 10 * 1024 * 1024
_DEFAULT_INGEST_CHUNK_SIZE = 16000
_DEFAULT_INGEST_CHUNK_OVERLAP = 600
_DEFAULT_INGEST_DEDUP_THRESHOLD = 0.06
_DEFAULT_AGENTIC_MAX_CALLS = 4


ProviderType = Literal["anthropic", "bedrock", "ollama"]
"""LLM provider identifier. Mirrors ``personal_kb.config._VALID_PROVIDERS``."""

DatabaseBackend = Literal["sqlite", "postgres"]
"""Discriminator value for :data:`DatabaseConfig`."""


# --- Database -----------------------------------------------------------------


@dataclass(frozen=True)
class SqliteConfig:
    """Configuration for the embedded SQLite + sqlite-vec backend."""

    path: Path = Path(_DEFAULT_DB_PATH)
    """Filesystem path to the SQLite DB file.

    May contain a leading ``~`` — the DB layer expands it at open time.
    """

    embedding_dim: int = _DEFAULT_EMBEDDING_DIM
    """Vector dimensionality. Must match the embedding model."""

    backend: Literal["sqlite"] = "sqlite"
    """Discriminator (always ``"sqlite"``) for the :data:`DatabaseConfig` union."""


@dataclass(frozen=True)
class PostgresConfig:
    """Configuration for the Postgres + pgvector backend (requires the ``postgres`` extra)."""

    dsn: str
    """``postgresql://...`` connection string."""

    embedding_dim: int = _DEFAULT_EMBEDDING_DIM
    """Vector dimensionality. Must match the embedding model."""

    pool_min: int = _DEFAULT_PG_POOL_MIN
    """Minimum size of the asyncpg connection pool."""

    pool_max: int = _DEFAULT_PG_POOL_MAX
    """Maximum size of the asyncpg connection pool."""

    iam_auth: bool = False
    """If ``True``, sign an RDS IAM auth token at connect time (requires the ``iam`` extra)."""

    region: str = _DEFAULT_PG_REGION
    """AWS region used for RDS IAM token signing when ``iam_auth`` is set."""

    backend: Literal["postgres"] = "postgres"
    """Discriminator (always ``"postgres"``) for the :data:`DatabaseConfig` union."""


DatabaseConfig = SqliteConfig | PostgresConfig
"""Discriminated union of supported backends. Switch on ``.backend`` or use ``isinstance``."""


# --- Embeddings ---------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the Ollama-hosted embedding model."""

    ollama_url: str = _DEFAULT_OLLAMA_URL
    """Base URL of the Ollama server (e.g. ``http://localhost:11434``)."""

    timeout: float = _DEFAULT_EMBEDDING_TIMEOUT
    """Per-request timeout in seconds."""

    model: str = _DEFAULT_EMBEDDING_MODEL
    """Name of the embedding model loaded in Ollama."""

    dim: int = _DEFAULT_EMBEDDING_DIM
    """Vector dimensionality. Must match :class:`SqliteConfig` / :class:`PostgresConfig`."""


# --- LLM providers ------------------------------------------------------------


@dataclass(frozen=True)
class AnthropicProviderConfig:
    """Credentials and tuning for the Anthropic API provider (requires the ``anthropic`` extra)."""

    model: str = _DEFAULT_ANTHROPIC_MODEL
    """Anthropic model identifier (e.g. ``claude-haiku-4-5``)."""

    timeout: float = _DEFAULT_ANTHROPIC_TIMEOUT
    """Per-request timeout in seconds."""

    api_key: str | None = None
    """API key. ``None`` lets the underlying SDK find credentials elsewhere."""


@dataclass(frozen=True)
class BedrockProviderConfig:
    """Credentials and tuning for the AWS Bedrock provider (requires the ``aws`` extra)."""

    model: str = _DEFAULT_BEDROCK_MODEL
    """Bedrock model identifier (cross-region inference profile recommended)."""

    timeout: float = _DEFAULT_BEDROCK_TIMEOUT
    """Per-request timeout in seconds."""

    region: str = _DEFAULT_BEDROCK_REGION
    """AWS region for Bedrock calls."""

    profile: str | None = None
    """AWS profile name (boto3 credential chain). ``None`` uses the default chain."""


@dataclass(frozen=True)
class OllamaProviderConfig:
    """Configuration for using Ollama as an LLM provider."""

    model: str = _DEFAULT_OLLAMA_LLM_MODEL
    """Ollama LLM model name (distinct from the embedding model)."""

    timeout: float = _DEFAULT_OLLAMA_LLM_TIMEOUT
    """Per-request timeout in seconds (LLM completions are slower than embeddings)."""

    url: str = _DEFAULT_OLLAMA_URL
    """Base URL of the Ollama server."""


@dataclass(frozen=True)
class ProviderRoleConfig:
    """Provider selection plus per-provider credentials for a single role.

    The active provider is :attr:`provider`; the other two configs are
    held alongside so that switching providers at runtime requires no
    extra plumbing. This is intentionally redundant — providers are
    cheap value objects with sensible defaults.
    """

    provider: ProviderType = "anthropic"
    """Active provider for this role."""

    anthropic: AnthropicProviderConfig = field(default_factory=AnthropicProviderConfig)
    """Anthropic creds/tuning (used when ``provider == "anthropic"``)."""

    bedrock: BedrockProviderConfig = field(default_factory=BedrockProviderConfig)
    """Bedrock creds/tuning (used when ``provider == "bedrock"``)."""

    ollama: OllamaProviderConfig = field(default_factory=OllamaProviderConfig)
    """Ollama config (used when ``provider == "ollama"``)."""


@dataclass(frozen=True)
class ProviderConfig:
    """Per-role LLM provider selection.

    Roles map to the three distinct LLM call sites in the engine:

    * ``extraction`` — graph enrichment (``KB_EXTRACTION_PROVIDER``).
    * ``query`` — query planning / agentic loop (``KB_QUERY_PROVIDER``).
    * ``synthesis`` — final answer synthesis. Today's env config reuses
      ``KB_QUERY_PROVIDER`` for synthesis; here it gets its own field so
      callers can split them if desired (default matches today).
    """

    extraction: ProviderRoleConfig = field(default_factory=ProviderRoleConfig)
    """Provider used for graph extraction during enrichment."""

    query: ProviderRoleConfig = field(default_factory=ProviderRoleConfig)
    """Provider used for query planning and the agentic ReAct loop."""

    synthesis: ProviderRoleConfig = field(default_factory=ProviderRoleConfig)
    """Provider used for final-answer synthesis."""


# --- Ingest -------------------------------------------------------------------


@dataclass(frozen=True)
class IngestConfig:
    """Tuning for file/URL ingestion and pre-store safety checks."""

    max_file_size: int = _DEFAULT_INGEST_MAX_FILE_SIZE
    """Maximum bytes per ingested file. Default 10 MiB."""

    chunk_size: int = _DEFAULT_INGEST_CHUNK_SIZE
    """Chunk size in characters for large-file chunking."""

    chunk_overlap: int = _DEFAULT_INGEST_CHUNK_OVERLAP
    """Overlap in characters between adjacent chunks."""

    dedup_threshold: float = _DEFAULT_INGEST_DEDUP_THRESHOLD
    """Hybrid-search score above which an incoming chunk is treated as a duplicate."""

    agentic_ingest: bool = True
    """If ``True``, run KB-aware dedup during ingestion."""

    skip_safety: bool = False
    """If ``True``, bypass secret scanning on store (``KB_SKIP_SAFETY`` semantics)."""


# --- Agentic loops ------------------------------------------------------------


@dataclass(frozen=True)
class AgenticConfig:
    """Tuning for the ReAct query agent and the agentic-synthesis path."""

    agentic_query: bool = True
    """Enable the agentic ReAct loop for ``kb_ask`` auto strategy."""

    agentic_synthesis: bool = True
    """Enable agentic retrieval + coverage check for ``kb_summarize``."""

    max_tool_calls: int = _DEFAULT_AGENTIC_MAX_CALLS
    """Maximum tool calls allowed in a single agentic query loop."""


# --- Attribution --------------------------------------------------------------


@dataclass(frozen=True)
class Attribution:
    """Per-entry attribution defaults (``KB_CONTRIBUTOR`` / ``KB_TEAM``)."""

    contributor: str | None = None
    """Contributor name attached to newly stored entries."""

    team: str | None = None
    """Team name attached to newly stored entries."""


# --- Top-level config ---------------------------------------------------------


@dataclass(frozen=True)
class KbConfig:
    """Top-level kb_core configuration. Compose, freeze, hand to the engine.

    A bare ``KbConfig()`` produces today's default behavior (SQLite at
    ``~/.local/share/personal_kb/knowledge.db``, Ollama embeddings,
    Anthropic LLM, agentic loops on, all safety on). The MCP server's
    startup wiring (later wave) builds this from env vars; library
    consumers build it from their own source of truth.
    """

    database: DatabaseConfig = field(default_factory=SqliteConfig)
    """Storage backend (SQLite or Postgres)."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    """Embedding model configuration."""

    providers: ProviderConfig = field(default_factory=ProviderConfig)
    """Per-role LLM provider selection."""

    ingest: IngestConfig = field(default_factory=IngestConfig)
    """File/URL ingestion tuning."""

    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    """Agentic ReAct / synthesis tuning."""

    attribution: Attribution = field(default_factory=Attribution)
    """Default contributor/team metadata for stored entries."""
