"""``KnowledgeBase`` — the public facade over the ``kb_core`` engine.

This is the API that library and microservice consumers use. It is
**additive**: the existing channel paths (MCP server, web explorer) keep
constructing the underlying pieces directly. The facade is what an
embedding consumer reaches for instead of wiring those pieces by hand.

Construction always flows through :class:`kb_core.config.KbConfig`:

* :meth:`KnowledgeBase.create` takes a fully-built ``KbConfig`` and opens
  every dependency the engine needs — the database (via
  :class:`kb_core.config.DatabaseConfig`), the embedder (when
  ``config.embedding`` is set), and the per-role LLM clients (from
  ``config.providers``). The DB schema is applied on open so a library
  consumer can hand the facade a brand-new database file and have it work.
* :func:`create_sqlite` / :func:`create_postgres` are sugar over
  :meth:`KnowledgeBase.create` — they assemble a ``KbConfig`` from the
  most common knobs (path/dsn, embedding dim, pool sizing, IAM auth) so
  the caller does not need to construct a config dataclass first.

The facade itself owns no business logic. It composes the lifted modules
(``kb_core.search.hybrid``, ``kb_core.store.knowledge_store``,
``kb_core.query``, ``kb_core.preflight``, ``kb_core.graph``,
``kb_core.ingest.ingester``) and forwards calls. Identical semantics to
the existing channel paths — see the W5 facade end-to-end tests for the
parity assertions.

Two rules the facade enforces:

* ``embedder=None`` is a first-class, explicit FTS-only choice. There is
  no silent "Ollama is down, fall back" path — if the caller wants
  embeddings, they configure them; if they want FTS-only, they pass
  ``embedding=None`` to the factory.
* ``embedding_dim`` routes to the backend schema, **not** the embedder.
  The embedder's dimension comes from ``EmbeddingConfig.dim``.

Async context manager — call sites are expected to use
``async with KnowledgeBase.create(...) as kb:`` (or the factory sugar).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kb_core.config import (
    AgenticConfig,
    Attribution,
    IngestConfig,
    KbConfig,
    PostgresConfig,
    ProviderConfig,
    SqliteConfig,
)
from kb_core.db.queries import get_entry
from kb_core.graph.builder import GraphBuilder
from kb_core.graph.queries import (
    bfs_entries,
    entries_for_scope,
    find_path,
    get_graph_vocabulary,
    get_neighbors,
    supersedes_chain,
)
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.preflight import build_project_context
from kb_core.search.embedder_protocol import BatchEmbedder
from kb_core.search.embeddings import EmbeddingClient
from kb_core.search.hybrid import hybrid_search
from kb_core.store.knowledge_store import KnowledgeStore

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime, timedelta
    from types import TracebackType

    from kb_core.config import (
        DatabaseConfig,
        EmbeddingConfig,
        ProviderRoleConfig,
    )
    from kb_core.db.backend import Database
    from kb_core.graph.enricher import GraphEnricher
    from kb_core.ingest.ingester import FileIngester, FileResult, IngestResult
    from kb_core.llm.provider import LLMProvider
    from kb_core.models.search import SearchQuery, SearchResult
    from kb_core.search.embedder_protocol import Embedder

logger = logging.getLogger(__name__)


# Sentinel for "use the value derived from the KbConfig". Distinguishes
# "caller did not pass an override" from "caller explicitly passed None".
class _UseConfig:
    """Sentinel singleton used as a default for override kwargs."""


_USE_CONFIG: Any = _UseConfig()


# ---------------------------------------------------------------------------
# Graph accessor
# ---------------------------------------------------------------------------


class _GraphAccessor:
    """Lightweight wrapper exposing :mod:`kb_core.graph.queries` as bound methods.

    Holds a reference to the parent facade's :class:`Database`. Created
    once per :class:`KnowledgeBase` and reused. Read-only — no graph
    mutation lives here (deterministic edges are built inside the store
    methods via :class:`GraphBuilder`; LLM enrichment via
    :class:`GraphEnricher`).
    """

    def __init__(self, db: Database) -> None:
        """Bind the accessor to the parent facade's database handle."""
        self._db = db

    async def neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 50,
    ) -> list[tuple[str, str, str]]:
        """List ``(neighbor_id, edge_type, direction)`` tuples for ``node_id``.

        Direction is ``"outgoing"`` or ``"incoming"``.
        """
        return await get_neighbors(
            self._db, node_id, edge_types=edge_types, direction=direction, limit=limit
        )

    async def bfs_entries(
        self,
        start_node: str,
        max_depth: int = 2,
        edge_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, int, list[str]]]:
        """BFS from ``start_node``; collect entry nodes with depth + path."""
        return await bfs_entries(
            self._db, start_node, max_depth=max_depth, edge_types=edge_types, limit=limit
        )

    async def find_path(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
    ) -> list[tuple[str, str, str]] | None:
        """Shortest path between two nodes via BFS, or ``None`` if not found."""
        return await find_path(self._db, source, target, max_depth=max_depth)

    async def supersedes_chain(self, entry_id: str) -> list[str]:
        """Full supersedes chain containing ``entry_id``, oldest first."""
        return await supersedes_chain(self._db, entry_id)

    async def entries_for_scope(
        self,
        scope: str,
        entry_type: str | None = None,
        order_by: str = "created_at",
    ) -> list[str]:
        """Entry IDs for a scope string (``project:X``, ``tag:Y``, ``kb-XXXXX``, etc.)."""
        return await entries_for_scope(self._db, scope, entry_type=entry_type, order_by=order_by)

    async def vocabulary(self, max_nodes: int = 200) -> dict[str, list[str]]:
        """Non-entry node IDs grouped by type, ordered by connection count."""
        return await get_graph_vocabulary(self._db, max_nodes=max_nodes)


# ---------------------------------------------------------------------------
# Internal helpers — DB open and LLM construction
# ---------------------------------------------------------------------------


async def _open_sqlite(cfg: SqliteConfig) -> Database:
    """Open a SQLite backend, load ``sqlite-vec`` if available, apply schema.

    Mirrors ``personal_kb.db.connection._create_sqlite`` minus the env-driven
    parts (which the SqliteConfig dataclass has snapshotted). The schema is
    applied here so a library consumer who hands in a brand-new file gets a
    ready-to-use db.
    """
    import aiosqlite

    path_str = str(cfg.path)
    # Expand a leading ``~`` only when we actually use the path — the
    # config is meant to be a pure value (no $HOME read on construction).
    if path_str.startswith("~"):
        path_str = str(Path(path_str).expanduser())

    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(path_str)
    conn.row_factory = aiosqlite.Row

    # WAL + foreign keys mirror the channel path's defaults.
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")

    # Best-effort sqlite-vec load — search degrades to FTS-only if missing.
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        raw: Any = conn._conn

        def _load_vec() -> None:
            raw.enable_load_extension(True)
            sqlite_vec.load(raw)
            raw.enable_load_extension(False)

        execute_threadsafe: Any = conn._execute
        await execute_threadsafe(_load_vec)
        logger.debug("sqlite-vec extension loaded")
    except Exception:
        logger.warning("sqlite-vec extension not available — vector search disabled")

    from kb_core.db.sqlite_backend import SQLiteBackend

    db = SQLiteBackend(conn)
    await db.apply_schema(embedding_dim=cfg.embedding_dim)
    return db


async def _open_postgres(cfg: PostgresConfig) -> Database:
    """Open a Postgres backend (pool + optional IAM auth) and apply schema."""
    from kb_core.db.postgres_backend import PostgresBackend

    create_kwargs: dict[str, Any] = {"pool_min": cfg.pool_min, "pool_max": cfg.pool_max}
    if cfg.iam_auth:
        from kb_core.db.iam_auth import make_ssl_context, make_token_factory, parse_dsn

        dsn = parse_dsn(cfg.dsn)
        create_kwargs["password"] = make_token_factory(dsn.host, dsn.port, dsn.username, cfg.region)
        create_kwargs["ssl"] = make_ssl_context()
        logger.info("Postgres IAM auth enabled: host=%s region=%s", dsn.host, cfg.region)

    db = await PostgresBackend.create(cfg.dsn, **create_kwargs)
    await db.apply_schema(embedding_dim=cfg.embedding_dim)
    return db


async def _open_database(cfg: DatabaseConfig) -> Database:
    """Discriminated-union dispatch over :data:`kb_core.config.DatabaseConfig`."""
    if isinstance(cfg, SqliteConfig):
        return await _open_sqlite(cfg)
    if isinstance(cfg, PostgresConfig):
        return await _open_postgres(cfg)
    msg = f"Unknown database backend: {cfg!r}"
    raise TypeError(msg)


def _build_llm(role: ProviderRoleConfig) -> LLMProvider | None:
    """Build the LLM client for a single role, or ``None`` if unsupported.

    Provider modules are imported lazily so that a bare ``import kb_core``
    never pulls anthropic/boto3/smithy (enforced by the purity guard).
    """
    if role.provider == "anthropic":
        from kb_core.llm.anthropic import AnthropicLLMClient

        return AnthropicLLMClient(role.anthropic)
    if role.provider == "bedrock":
        from kb_core.llm.bedrock import BedrockLLMClient

        return BedrockLLMClient(role.bedrock)
    if role.provider == "ollama":
        from kb_core.llm.ollama import OllamaLLMClient

        return OllamaLLMClient(role.ollama)
    return None


# ---------------------------------------------------------------------------
# KnowledgeBase facade
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """High-level facade over the ``kb_core`` engine.

    Composes the lifted modules and exposes them as a single async
    object. Open it via :meth:`create` (or the :func:`create_sqlite` /
    :func:`create_postgres` factory sugar) and use it as an async context
    manager — :meth:`close` releases every owned resource (DB pool,
    HTTP clients, LLM SDK clients) it built itself. Externally supplied
    embedders are NOT closed (caller owns them).
    """

    def __init__(
        self,
        *,
        config: KbConfig,
        db: Database,
        store: KnowledgeStore,
        embedder: Embedder | None,
        graph_builder: GraphBuilder,
        graph_enricher: GraphEnricher | None,
        extraction_llm: LLMProvider | None,
        query_llm: LLMProvider | None,
        synthesis_llm: LLMProvider | None,
        _owned_embedder: bool = False,
        _owned_extraction_llm: bool = False,
        _owned_query_llm: bool = False,
        _owned_synthesis_llm: bool = False,
    ) -> None:
        """Wire in pre-built dependencies.

        Most callers should use :meth:`create` (or the
        :func:`create_sqlite` / :func:`create_postgres` factories) rather
        than calling this constructor directly. The ``_owned_*`` flags
        track which resources the facade itself built (and therefore
        must close on shutdown) vs ones the caller supplied (which the
        caller is responsible for).
        """
        self._config = config
        self._db = db
        self._store = store
        self._embedder = embedder
        self._graph_builder = graph_builder
        self._graph_enricher = graph_enricher
        self._extraction_llm = extraction_llm
        self._query_llm = query_llm
        self._synthesis_llm = synthesis_llm
        self._graph = _GraphAccessor(db)
        # Ownership flags — see _close_owned() below.
        self._owned_embedder = _owned_embedder
        self._owned_extraction_llm = _owned_extraction_llm
        self._owned_query_llm = _owned_query_llm
        self._owned_synthesis_llm = _owned_synthesis_llm
        # Database is ALWAYS owned by the facade (we always open it in create()).
        self._owned_db = True

    # -- Lifecycle ----------------------------------------------------------

    @classmethod
    async def create(
        cls,
        config: KbConfig,
        *,
        embedder: Embedder | None | _UseConfig = _USE_CONFIG,
        extraction_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
        query_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
        synthesis_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
    ) -> KnowledgeBase:
        """Open every dependency the engine needs from a :class:`KbConfig`.

        The DB is opened (via the connection logic for SQLite or Postgres)
        and the schema is applied so a brand-new file is ready to use.
        The embedder is built from ``config.embedding`` (``None`` ->
        FTS-only). The per-role LLM clients are built from
        ``config.providers``.

        Test/advanced overrides (``embedder``, ``extraction_llm``,
        ``query_llm``, ``synthesis_llm``) accept either a pre-built
        instance or ``None`` to bypass that piece entirely; the sentinel
        default means "derive from ``config``". Overridden instances are
        NOT closed by :meth:`close` — the caller owns them.
        """
        db = await _open_database(config.database)

        # Embedder: respect the override sentinel.
        owned_embedder = False
        if isinstance(embedder, _UseConfig):
            if config.embedding is not None:
                resolved_embedder: Embedder | None = EmbeddingClient(db, config=config.embedding)
                owned_embedder = True
            else:
                resolved_embedder = None
        else:
            resolved_embedder = embedder

        # LLMs.
        owned_extraction = False
        if isinstance(extraction_llm, _UseConfig):
            resolved_extraction: LLMProvider | None = _build_llm(config.providers.extraction)
            owned_extraction = resolved_extraction is not None
        else:
            resolved_extraction = extraction_llm

        owned_query = False
        if isinstance(query_llm, _UseConfig):
            resolved_query: LLMProvider | None = _build_llm(config.providers.query)
            owned_query = resolved_query is not None
        else:
            resolved_query = query_llm

        owned_synthesis = False
        if isinstance(synthesis_llm, _UseConfig):
            resolved_synthesis: LLMProvider | None = _build_llm(config.providers.synthesis)
            owned_synthesis = resolved_synthesis is not None
        else:
            resolved_synthesis = synthesis_llm

        store = KnowledgeStore(db)
        graph_builder = GraphBuilder(db)

        graph_enricher: GraphEnricher | None = None
        if resolved_extraction is not None:
            from kb_core.graph.enricher import GraphEnricher as _GraphEnricher

            graph_enricher = _GraphEnricher(db, resolved_extraction)

        return cls(
            config=config,
            db=db,
            store=store,
            embedder=resolved_embedder,
            graph_builder=graph_builder,
            graph_enricher=graph_enricher,
            extraction_llm=resolved_extraction,
            query_llm=resolved_query,
            synthesis_llm=resolved_synthesis,
            _owned_embedder=owned_embedder,
            _owned_extraction_llm=owned_extraction,
            _owned_query_llm=owned_query,
            _owned_synthesis_llm=owned_synthesis,
        )

    async def __aenter__(self) -> KnowledgeBase:
        """Return self (the facade is already open by the time ``create`` returns)."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release every resource the facade owns."""
        await self.close()

    async def close(self) -> None:
        """Release every resource the facade itself opened.

        Externally supplied embedders/LLMs are NOT closed — caller owns them.
        Failures during teardown are logged at WARNING and never raised.
        """
        # Close in reverse construction order. Each close is best-effort.
        if self._owned_synthesis_llm and self._synthesis_llm is not None:
            try:
                await self._synthesis_llm.close()
            except Exception:
                logger.warning("Synthesis LLM close failed", exc_info=True)
        if self._owned_query_llm and self._query_llm is not None:
            try:
                await self._query_llm.close()
            except Exception:
                logger.warning("Query LLM close failed", exc_info=True)
        if self._owned_extraction_llm and self._extraction_llm is not None:
            try:
                await self._extraction_llm.close()
            except Exception:
                logger.warning("Extraction LLM close failed", exc_info=True)
        if self._owned_embedder and self._embedder is not None:
            close_fn = getattr(self._embedder, "close", None)
            if callable(close_fn):
                try:
                    await close_fn()
                except Exception:
                    logger.warning("Embedder close failed", exc_info=True)
        if self._owned_db:
            try:
                await self._db.close()
            except Exception:
                logger.warning("Database close failed", exc_info=True)

    # -- Accessors ----------------------------------------------------------

    @property
    def config(self) -> KbConfig:
        """The :class:`KbConfig` this facade was created with."""
        return self._config

    @property
    def db(self) -> Database:
        """The underlying :class:`Database` handle (advanced; prefer the facade methods)."""
        return self._db

    @property
    def graph(self) -> _GraphAccessor:
        """Read-only graph traversal accessor."""
        return self._graph

    @property
    def embedder(self) -> Embedder | None:
        """The configured embedder, or ``None`` for explicit FTS-only mode."""
        return self._embedder

    @property
    def knowledge_store(self) -> KnowledgeStore:
        """The underlying :class:`KnowledgeStore` (advanced; prefer facade methods).

        Named ``knowledge_store`` (not ``store``) so it doesn't collide with the
        :meth:`store` method that creates entries.
        """
        return self._store

    @property
    def graph_builder(self) -> GraphBuilder:
        """The deterministic :class:`GraphBuilder` (advanced; prefer facade methods)."""
        return self._graph_builder

    @property
    def graph_enricher(self) -> GraphEnricher | None:
        """The LLM-driven :class:`GraphEnricher`, or ``None`` when no extraction LLM is set."""
        return self._graph_enricher

    @property
    def extraction_llm(self) -> LLMProvider | None:
        """The LLM used for graph extraction, or ``None`` if unavailable."""
        return self._extraction_llm

    @property
    def query_llm(self) -> LLMProvider | None:
        """The LLM used for query planning + the agentic ReAct loop, or ``None`` if unavailable."""
        return self._query_llm

    @property
    def synthesis_llm(self) -> LLMProvider | None:
        """The (stronger) LLM used for human-facing synthesis, or ``None`` if unavailable."""
        return self._synthesis_llm

    # -- Search -------------------------------------------------------------

    async def search(
        self,
        query: SearchQuery,
        *,
        contributor: str | None = None,
    ) -> tuple[list[SearchResult], int]:
        """Hybrid (FTS + vector) search; FTS-only when no embedder is configured.

        Identical contract to :func:`kb_core.search.hybrid.hybrid_search` —
        the facade just forwards. Returns ``(results, filtered_count)``.
        """
        return await hybrid_search(self._db, self._embedder, query, contributor=contributor)

    # -- Store CRUD ---------------------------------------------------------

    async def store(
        self,
        *,
        short_title: str,
        long_title: str,
        knowledge_details: str,
        entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
        project_ref: str | None = None,
        source_context: str | None = None,
        confidence_level: float = 0.9,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        contributor: str | None = None,
        team: str | None = None,
        sensitivity: str | None = None,
        expires_at: datetime | None = None,
        enrich: bool = True,
    ) -> KnowledgeEntry:
        """Create one entry through the full store pipeline.

        Creates the entry, embeds it (when an embedder is configured),
        builds deterministic graph edges, and optionally runs the LLM
        enricher. Falls back gracefully when any optional dependency is
        missing — none of the optional pieces fail the call.

        ``contributor``/``team`` default to the values on
        ``config.attribution`` when not given.
        """
        contributor = (
            contributor if contributor is not None else self._config.attribution.contributor
        )
        team = team if team is not None else self._config.attribution.team

        entry = await self._store.create_entry(
            short_title=short_title,
            long_title=long_title,
            knowledge_details=knowledge_details,
            entry_type=entry_type,
            project_ref=project_ref,
            source_context=source_context,
            confidence_level=confidence_level,
            tags=tags,
            hints=hints,
            contributor=contributor,
            team=team,
            sensitivity=sensitivity,  # type: ignore[arg-type]
            expires_at=expires_at,
        )

        await self._embed_one(entry)
        await self._build_graph(entry)
        if enrich:
            await self._enrich_one(entry)
        return entry

    async def store_batch(
        self,
        entries: list[dict[str, Any]],
        *,
        enrich: bool = True,
    ) -> list[KnowledgeEntry]:
        """Create multiple entries; batch-embed + batch-enrich at the end.

        Each ``entries`` dict requires ``short_title``, ``long_title``,
        ``knowledge_details`` and accepts the same optional fields as
        :meth:`store`. Returns the list of created entries in input
        order. A failure on any single entry is logged and skipped — the
        rest still proceed.
        """
        created: list[KnowledgeEntry] = []
        for entry_dict in entries:
            try:
                entry_type_raw = entry_dict.get("entry_type", "factual_reference")
                entry_type = (
                    entry_type_raw
                    if isinstance(entry_type_raw, EntryType)
                    else EntryType(str(entry_type_raw))
                )
                entry = await self._store.create_entry(
                    short_title=str(entry_dict["short_title"]),
                    long_title=str(entry_dict["long_title"]),
                    knowledge_details=str(entry_dict["knowledge_details"]),
                    entry_type=entry_type,
                    project_ref=entry_dict.get("project_ref"),
                    source_context=entry_dict.get("source_context"),
                    confidence_level=float(entry_dict.get("confidence_level", 0.9)),
                    tags=list(entry_dict["tags"]) if entry_dict.get("tags") else None,
                    hints=dict(entry_dict["hints"]) if entry_dict.get("hints") else None,
                    contributor=entry_dict.get("contributor", self._config.attribution.contributor),
                    team=entry_dict.get("team", self._config.attribution.team),
                    sensitivity=entry_dict.get("sensitivity"),
                    expires_at=entry_dict.get("expires_at"),
                )
            except Exception:
                logger.warning(
                    "store_batch: skipping entry %r",
                    entry_dict.get("short_title"),
                    exc_info=True,
                )
                continue

            await self._build_graph(entry)
            created.append(entry)

        # Batch embed (single backend call across the whole list).
        if created and self._embedder is not None:
            try:
                texts = [e.embedding_text for e in created]
                embed_batch = getattr(self._embedder, "embed_batch", None)
                if callable(embed_batch):
                    embeddings = await embed_batch(texts)
                    store_embeddings = getattr(self._embedder, "store_embeddings", None)
                    if embeddings is not None and callable(store_embeddings):
                        pairs = list(zip([e.id for e in created], embeddings, strict=True))
                        await store_embeddings(pairs)
                        for e in created:
                            await self._store.mark_embedding(e.id, True)
                else:
                    # Embedder lacks batch API — embed one at a time.
                    for entry in created:
                        await self._embed_one(entry)
            except Exception:
                logger.warning("store_batch: batch embedding failed", exc_info=True)

        # Batch enrich.
        if enrich and self._graph_enricher is not None and created:
            try:
                await self._graph_enricher.enrich_batch(created)
            except Exception:
                logger.warning("store_batch: batch enrichment failed", exc_info=True)
            finally:
                self._graph_enricher.clear_vocab_cache()

        return created

    async def update(
        self,
        entry_id: str,
        *,
        knowledge_details: str | None = None,
        change_reason: str | None = None,
        confidence_level: float | None = None,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        updated_by: str | None = None,
        sensitivity: str | None = None,
        expires_at: datetime | None = None,
        short_title: str | None = None,
        long_title: str | None = None,
        entry_type: EntryType | None = None,
        project_ref: str | None = None,
        source_context: str | None = None,
        enrich: bool = True,
    ) -> KnowledgeEntry:
        """Update an existing entry and refresh embedding/graph.

        Mirrors :meth:`KnowledgeStore.update_entry`; on a content change
        the entry is re-embedded and its deterministic graph edges are
        rebuilt. Optional LLM enrichment runs when ``enrich`` is True
        and an enricher is configured.
        """
        content_changed = knowledge_details is not None
        entry = await self._store.update_entry(
            entry_id,
            knowledge_details=knowledge_details,
            change_reason=change_reason,
            confidence_level=confidence_level,
            tags=tags,
            hints=hints,
            updated_by=updated_by,
            sensitivity=sensitivity,  # type: ignore[arg-type]
            expires_at=expires_at,
            short_title=short_title,
            long_title=long_title,
            entry_type=entry_type,
            project_ref=project_ref,
            source_context=source_context,
        )

        # Re-embed only when the embedding text actually changed.
        if content_changed:
            await self._embed_one(entry)
        await self._build_graph(entry)
        if enrich and content_changed:
            await self._enrich_one(entry)
        return entry

    async def deactivate(self, entry_id: str, *, contributor: str | None = None) -> KnowledgeEntry:
        """Soft-delete an entry. ``contributor`` defaults to the configured one."""
        contributor = (
            contributor if contributor is not None else self._config.attribution.contributor
        )
        return await self._store.deactivate_entry(entry_id, contributor=contributor)

    async def reactivate(self, entry_id: str, *, contributor: str | None = None) -> KnowledgeEntry:
        """Restore a previously deactivated entry."""
        contributor = (
            contributor if contributor is not None else self._config.attribution.contributor
        )
        return await self._store.reactivate_entry(entry_id, contributor=contributor)

    async def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Return a single entry by ID, or ``None`` if not found."""
        return await get_entry(self._db, entry_id)

    async def bulk_update(
        self,
        filters: dict[str, object],
        updates: dict[str, object],
        *,
        contributor: str | None = None,
        dry_run: bool = False,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """Bulk metadata update — see :meth:`KnowledgeStore.bulk_update`."""
        contributor = (
            contributor if contributor is not None else self._config.attribution.contributor
        )
        return await self._store.bulk_update(
            filters, updates, contributor=contributor, dry_run=dry_run
        )

    # -- Query / synthesis --------------------------------------------------

    async def ask(
        self,
        question: str,
        *,
        scope: str | None = None,
        agentic: bool | None = None,
        max_tool_calls: int | None = None,
        limit: int = 20,
        include_graph_context: bool = True,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
        """Retrieve entries for ``question`` via the agentic-or-planner path.

        Returns ``(entries_with_context, agent_turns_used)`` — the same
        contract as :func:`kb_core.query.retrieve_entries`. ``agentic``
        and ``max_tool_calls`` default to the values on
        ``config.agentic`` when not specified.
        """
        from kb_core.query import retrieve_entries

        agentic_resolved = agentic if agentic is not None else self._config.agentic.agentic_query
        max_tool_calls_resolved = (
            max_tool_calls if max_tool_calls is not None else self._config.agentic.max_tool_calls
        )
        return await retrieve_entries(
            self._db,
            self._embedder,
            self._query_llm,
            question,
            scope=scope,
            include_graph_context=include_graph_context,
            limit=limit,
            event_callback=event_callback,
            agentic=agentic_resolved,
            max_tool_calls=max_tool_calls_resolved,
        )

    async def summarize(
        self,
        question: str,
        *,
        scope: str | None = None,
        agentic: bool | None = None,
        agentic_synthesis: bool | None = None,
        max_tool_calls: int | None = None,
        limit: int = 20,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """Synthesize a natural-language answer for ``question``.

        Forwards to :func:`kb_core.query.synthesize_answer`. ``agentic``,
        ``agentic_synthesis`` and ``max_tool_calls`` default to
        ``config.agentic``.
        """
        from kb_core.query import synthesize_answer

        agentic_resolved = agentic if agentic is not None else self._config.agentic.agentic_query
        agentic_synthesis_resolved = (
            agentic_synthesis
            if agentic_synthesis is not None
            else self._config.agentic.agentic_synthesis
        )
        max_tool_calls_resolved = (
            max_tool_calls if max_tool_calls is not None else self._config.agentic.max_tool_calls
        )
        return await synthesize_answer(
            self._db,
            self._embedder,
            self._query_llm,
            question,
            scope=scope,
            limit=limit,
            event_callback=event_callback,
            synthesis_llm=self._synthesis_llm,
            agentic=agentic_resolved,
            agentic_synthesis=agentic_synthesis_resolved,
            max_tool_calls=max_tool_calls_resolved,
        )

    # -- Ingest -------------------------------------------------------------

    def _build_ingester(
        self,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileIngester | None:
        """Build a :class:`FileIngester` from current dependencies, or None.

        Returns ``None`` when extraction is not possible (no extraction
        LLM and no embedder — both are required by the ingest pipeline).

        ``contributor`` and ``team`` override ``config.attribution`` for this
        ingester instance only. When ``None`` (the default), the
        construction-time attribution values from ``config.attribution`` are
        used — preserving today's behavior for callers that don't pass them.
        """
        if self._extraction_llm is None:
            return None
        if self._embedder is None:
            return None
        from kb_core.ingest.dedup_agent import DedupAgent
        from kb_core.ingest.ingester import FileIngester

        # ingest_*() needs a BatchEmbedder; assert at runtime for narrower typing.
        if not isinstance(self._embedder, BatchEmbedder):
            return None
        dedup_agent: DedupAgent | None = None
        if self._config.ingest.agentic_ingest:
            dedup_agent = DedupAgent(
                self._db,
                self._embedder,
                self._extraction_llm,
                threshold=self._config.ingest.dedup_threshold,
            )
        return FileIngester(
            db=self._db,
            store=self._store,
            embedder=self._embedder,
            graph_builder=self._graph_builder,
            graph_enricher=self._graph_enricher,
            llm=self._extraction_llm,
            dedup_agent=dedup_agent,
            contributor=(
                contributor if contributor is not None else self._config.attribution.contributor
            ),
            team=team if team is not None else self._config.attribution.team,
            config=self._config.ingest,
        )

    async def ingest_file(
        self,
        path: Path | str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Ingest a file from disk through the full pipeline.

        Requires an extraction LLM and an embedder; raises
        :class:`RuntimeError` if either is missing.

        ``contributor`` and ``team`` override the construction-time
        ``config.attribution`` for this call only. When ``None`` (the
        default), the ctor attribution is used — existing callers are
        100% unaffected.
        """
        ingester = self._build_ingester(contributor=contributor, team=team)
        if ingester is None:
            msg = (
                "ingest_file requires both an extraction LLM and an embedder. "
                "Configure providers.extraction and embedding on KbConfig."
            )
            raise RuntimeError(msg)
        return await ingester.ingest_file(Path(path), project_ref=project_ref, dry_run=dry_run)

    async def ingest_text(
        self,
        content: str,
        source_name: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Ingest raw text (e.g. from an upload) through the full pipeline.

        ``contributor`` and ``team`` override the construction-time
        ``config.attribution`` for this call only. When ``None`` (the
        default), the ctor attribution is used — existing callers are
        100% unaffected.

        ``dry_run=True`` runs the extraction pipeline but does not write any
        entries to the database. Returns a :class:`FileResult` with
        ``action="dry_run"`` and the entry count that *would* be created.
        """
        ingester = self._build_ingester(contributor=contributor, team=team)
        if ingester is None:
            msg = (
                "ingest_text requires both an extraction LLM and an embedder. "
                "Configure providers.extraction and embedding on KbConfig."
            )
            raise RuntimeError(msg)
        return await ingester.ingest_text(
            content, source_name, project_ref=project_ref, dry_run=dry_run
        )

    async def ingest_url(
        self,
        url: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Fetch a URL, extract its article content, and ingest it.

        Thin wrapper over :meth:`FileIngester.ingest_url`. Requires an
        extraction LLM and an embedder; raises :class:`RuntimeError` if
        either is missing.

        ``contributor`` and ``team`` override the construction-time
        ``config.attribution`` for this call only. When ``None`` (the
        default), the ctor attribution is used — existing callers are
        100% unaffected.
        """
        ingester = self._build_ingester(contributor=contributor, team=team)
        if ingester is None:
            msg = (
                "ingest_url requires both an extraction LLM and an embedder. "
                "Configure providers.extraction and embedding on KbConfig."
            )
            raise RuntimeError(msg)
        return await ingester.ingest_url(url, project_ref=project_ref, dry_run=dry_run)

    async def ingest_url_content(
        self,
        content: str,
        source_url: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Ingest pre-fetched URL content (e.g. from authenticated sites).

        Skips the HTTP fetch + HTML extraction stages — feeds ``content``
        straight into the pipeline with ``source_url`` as the attribution.
        Useful when the caller already has clean text (WebFetch output,
        internal wiki dumps, JavaScript-rendered pages). Requires an
        extraction LLM and an embedder.

        ``contributor`` and ``team`` override the construction-time
        ``config.attribution`` for this call only. When ``None`` (the
        default), the ctor attribution is used — existing callers are
        100% unaffected.
        """
        ingester = self._build_ingester(contributor=contributor, team=team)
        if ingester is None:
            msg = (
                "ingest_url_content requires both an extraction LLM and an embedder. "
                "Configure providers.extraction and embedding on KbConfig."
            )
            raise RuntimeError(msg)
        return await ingester._ingest_content(
            content, source_url, project_ref=project_ref, dry_run=dry_run
        )

    async def ingest_directory(
        self,
        dir_path: Path | str,
        *,
        project_ref: str | None = None,
        recursive: bool = True,
        dry_run: bool = False,
    ) -> IngestResult:
        """Ingest all eligible files from a directory through the pipeline."""
        ingester = self._build_ingester()
        if ingester is None:
            msg = (
                "ingest_directory requires both an extraction LLM and an embedder. "
                "Configure providers.extraction and embedding on KbConfig."
            )
            raise RuntimeError(msg)
        return await ingester.ingest_directory(
            Path(dir_path), project_ref=project_ref, recursive=recursive, dry_run=dry_run
        )

    # -- Project context / maps --------------------------------------------

    async def preflight(
        self,
        project_ref: str,
        *,
        team: str | None = None,
        since: timedelta | None = None,
    ) -> str:
        """Compact project context primer — see :func:`build_project_context`.

        ``team`` defaults to ``config.attribution.team``.
        """
        team_resolved = team if team is not None else self._config.attribution.team
        return await build_project_context(self._db, project_ref, team=team_resolved, since=since)

    async def maps_for_project(
        self, project_ref: str, *, team: str | None = None
    ) -> list[dict[str, str]]:
        """Return active ``mental_map`` entries for a project as dicts.

        Returns plain data ``{"id", "short_title", "long_title"}`` computed
        live, in-memory, from the database — callers (e.g. the hosted
        service's maps route) render this directly without any on-disk index.

        Uses the exact same predicate as :func:`build_project_context`'s
        maps section: active + ``entry_type='mental_map'`` + matching
        ``project_ref`` (and optional team), ordered by ``created_at``
        descending, no limit.
        """
        team_resolved = team if team is not None else self._config.attribution.team
        # Import the SQL builder used by preflight to guarantee parity.
        from kb_core.preflight import _maps_sql

        sql, has_team = _maps_sql(team_resolved)
        params: list[str] = [project_ref]
        if has_team and team_resolved is not None:
            params.append(team_resolved)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        out: list[dict[str, str]] = []
        for row in rows:
            out.append(
                {
                    "id": str(row[0]),
                    "short_title": str(row[1] or ""),
                    "long_title": str(row[2] or ""),
                }
            )
        return out

    # -- Embeddings (optional) ---------------------------------------------

    async def embed(self, text: str) -> list[float] | None:
        """Embed a single string, or ``None`` when no embedder is configured."""
        if self._embedder is None:
            return None
        return await self._embedder.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch of strings, or ``None`` when no embedder is configured."""
        if self._embedder is None:
            return None
        embed_batch = getattr(self._embedder, "embed_batch", None)
        if not callable(embed_batch):
            # Fall back to per-text embedding when the embedder isn't a BatchEmbedder.
            results: list[list[float]] = []
            for text in texts:
                vec = await self._embedder.embed(text)
                if vec is None:
                    return None
                results.append(vec)
            return results
        out: list[list[float]] | None = await embed_batch(texts)
        return out

    # -- Internal helpers ---------------------------------------------------

    async def _embed_one(self, entry: KnowledgeEntry) -> None:
        """Embed a single entry. Failures are logged and never raise."""
        if self._embedder is None:
            return
        try:
            embedding = await self._embedder.embed(entry.embedding_text)
            if embedding is None:
                return
            # The embedder may not expose store_embedding; use the DB path.
            store_embedding = getattr(self._embedder, "store_embedding", None)
            if callable(store_embedding):
                await store_embedding(entry.id, embedding)
            else:
                await self._db.vector_store(entry.id, embedding)
                await self._db.commit()
            await self._store.mark_embedding(entry.id, True)
        except Exception:
            logger.warning("Failed to embed entry %s", entry.id, exc_info=True)

    async def _build_graph(self, entry: KnowledgeEntry) -> None:
        """Build deterministic graph edges for an entry. Best-effort."""
        try:
            await self._graph_builder.build_for_entry(entry)
        except Exception:
            logger.warning("Failed to build graph for entry %s", entry.id, exc_info=True)

    async def _enrich_one(self, entry: KnowledgeEntry) -> None:
        """LLM-enrich a single entry's graph edges. Best-effort."""
        if self._graph_enricher is None:
            return
        try:
            await self._graph_enricher.enrich_entry(entry)
        except Exception:
            logger.warning("Failed to enrich entry %s", entry.id, exc_info=True)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


async def create_sqlite(
    path: Path | str,
    *,
    embedding_dim: int = 1024,
    embedding: EmbeddingConfig | None = None,
    providers: ProviderConfig | None = None,
    ingest: IngestConfig | None = None,
    agentic: AgenticConfig | None = None,
    attribution: Attribution | None = None,
    embedder: Embedder | None | _UseConfig = _USE_CONFIG,
    extraction_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
    query_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
    synthesis_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
) -> KnowledgeBase:
    """Open a SQLite-backed :class:`KnowledgeBase`.

    Convenience over :meth:`KnowledgeBase.create` that builds the
    :class:`KbConfig` for the caller. The default behaviour is **FTS-only**
    — pass ``embedding=EmbeddingConfig(...)`` to opt into Ollama
    embeddings. Schema is applied on open, so the path can point at a
    brand-new file.

    ``embedding_dim`` controls the vector table dimensionality at the
    storage layer; it does NOT change the embedder's output dimension
    (that comes from ``embedding.dim``). The two must match if both are
    set — callers are responsible for keeping them in sync.
    """
    config = KbConfig(
        database=SqliteConfig(path=Path(str(path)), embedding_dim=embedding_dim),
        embedding=embedding,
        providers=providers if providers is not None else ProviderConfig(),
        ingest=ingest if ingest is not None else IngestConfig(),
        agentic=agentic if agentic is not None else AgenticConfig(),
        attribution=attribution if attribution is not None else Attribution(),
    )
    return await KnowledgeBase.create(
        config,
        embedder=embedder,
        extraction_llm=extraction_llm,
        query_llm=query_llm,
        synthesis_llm=synthesis_llm,
    )


async def create_postgres(
    dsn: str,
    *,
    embedding_dim: int = 1024,
    pool_min: int = 1,
    pool_max: int = 5,
    iam_auth: bool = False,
    region: str = "us-east-1",
    embedding: EmbeddingConfig | None = None,
    providers: ProviderConfig | None = None,
    ingest: IngestConfig | None = None,
    agentic: AgenticConfig | None = None,
    attribution: Attribution | None = None,
    embedder: Embedder | None | _UseConfig = _USE_CONFIG,
    extraction_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
    query_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
    synthesis_llm: LLMProvider | None | _UseConfig = _USE_CONFIG,
) -> KnowledgeBase:
    """Open a Postgres-backed :class:`KnowledgeBase` (requires the ``postgres`` extra).

    Schema is applied on open inside an advisory lock so concurrent
    starts converge cleanly. Set ``iam_auth=True`` for RDS/Aurora IAM
    authentication (requires the ``iam`` extra).
    """
    config = KbConfig(
        database=PostgresConfig(
            dsn=dsn,
            embedding_dim=embedding_dim,
            pool_min=pool_min,
            pool_max=pool_max,
            iam_auth=iam_auth,
            region=region,
        ),
        embedding=embedding,
        providers=providers if providers is not None else ProviderConfig(),
        ingest=ingest if ingest is not None else IngestConfig(),
        agentic=agentic if agentic is not None else AgenticConfig(),
        attribution=attribution if attribution is not None else Attribution(),
    )
    return await KnowledgeBase.create(
        config,
        embedder=embedder,
        extraction_llm=extraction_llm,
        query_llm=query_llm,
        synthesis_llm=synthesis_llm,
    )


__all__ = [
    "KnowledgeBase",
    "create_postgres",
    "create_sqlite",
]
