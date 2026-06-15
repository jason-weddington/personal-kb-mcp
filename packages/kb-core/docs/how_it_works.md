# kb-core: How It Works (Engine Internals)

This document explains how the `kb-core` engine is built and how its
pieces fit together. It is the maintainer / contributor view: every
section names the files, classes, and constants that implement the
behavior so you can jump straight from a question into the code.

The doc covers the **engine layer only** — everything that lives under
`packages/kb-core/src/kb_core/`. Consumer concerns (MCP tools, the
`personal-kb-hook` CLI, the listener, the web graph explorer) live in
the `personal_kb` package and are intentionally out of scope here.
References use the `kb_core/<module>/<file>.py` form so each path
resolves directly under `src/kb_core/`.

> **Configuration is explicit.** `kb_core` reads **no environment
> variables** — every tunable is a typed field on
> `kb_core.config.KbConfig` (or one of its nested dataclasses). The MCP
> server, the web explorer, and any other consumer is responsible for
> turning env vars / config files / CLI flags into a `KbConfig` and
> handing it to the engine. The entry points the rest of this doc
> refers to are `create_sqlite` and `create_postgres` in
> `kb_core/knowledge_base.py` (both re-exported from
> `kb_core/__init__.py`).

---

## 1. Storage Engine

The engine's source of truth is a SQL database with five logical
groupings of tables, all defined in `kb_core/db/schema.py`:

- **`knowledge_entries`** — one row per entry. Columns include
  `id` (5-digit `kb-XXXXX` string, primary key), `project_ref`,
  `short_title`, `long_title`, `knowledge_details`, `entry_type`,
  `confidence_level`, `tags` (a single `TEXT` column — see below),
  `hints` (JSON), `created_at` / `updated_at` / `last_accessed` /
  `expires_at`, `superseded_by`, `is_active`, `has_embedding`,
  `version`, plus multi-user columns (`contributor`, `team`,
  `updated_by`) and `sensitivity` added by the v2 migration.
- **`entry_versions`** — append-only history for `update_entry`; one
  row per version with `version_number`, `knowledge_details`,
  `change_reason`, `confidence_level`, and `created_at`.
- **`knowledge_fts`** — a virtual `fts5` table mirrored from
  `knowledge_entries` via three triggers (`knowledge_fts_ai`,
  `knowledge_fts_ad`, `knowledge_fts_au`). Tokenizer: `porter unicode61`.
- **`knowledge_vec`** — a virtual `vec0` table created at the
  configured embedding dim (default 1024). Backed by `sqlite-vec` on
  SQLite; the Postgres backend uses a `vector` column on a regular
  table.
- **`graph_nodes` + `graph_edges`** — the knowledge graph (DDL in
  `apply_graph_schema`); see §10 for how these are written.
- **`ingested_files`** — bookkeeping for the file-ingestion pipeline
  (DDL in `apply_ingest_schema`).
- **`entry_id_seq`** — a single-row table holding the next entry
  number. Created by the main schema and seeded by `INIT_SEQ_SQL`.
- Two telemetry / governance tables — `search_events` and
  `agent_feedback` (DDL in `apply_search_events_schema` /
  `apply_feedback_schema`) — plus an `audit_events` table and a
  `deployment_config` table.

### `kb-XXXXX` 5-digit IDs

Every entry ID is a `kb-` prefix followed by a zero-padded 5-digit
number. The format string lives in
`kb_core/db/queries.py::next_entry_id`:

```python
async def next_entry_id(db: Database) -> str:
    val = await db.next_sequence_value()
    return f"kb-{val:05d}"
```

`next_sequence_value` is the atomic increment exposed on the
`Database` Protocol (see §2); the underlying row lives in
`entry_id_seq`.

### Tags are space-separated text

Despite the column name suggesting a JSON list, `knowledge_entries.tags`
is a single `TEXT` column written by `insert_entry` / `update_entry`
(`kb_core/db/queries.py`) as `tags_text = " ".join(entry.tags)`. The
filter-only and tag-match SQL throughout the engine therefore uses the
pattern `(' ' || tags || ' ') LIKE '% <tag> %'` (see `hybrid.py` and
both backends' `fts_search` / `vector_search`). This is intentional:
FTS5 indexes the tags column verbatim, so a `MATCH 'python sqlite'`
query hits an entry tagged either or both.

---

## 2. Database Protocol + backends

`kb_core` defines a single Protocol that every backend implements:
`Database` in `kb_core/db/backend.py`. All engine SQL is written against
this Protocol, using `?` placeholders and SQLite-flavored syntax; the
Postgres backend translates at execute time (`?` → `$N`,
`INSERT OR IGNORE` → `ON CONFLICT DO NOTHING`).

The Protocol's method surface:

| Method | Purpose |
| --- | --- |
| `execute(sql, params)` | Run one statement, return a `Cursor`. |
| `executemany(sql, params_seq)` | Run a statement once per param tuple. |
| `executescript(sql)` | Run a DDL / migration script. |
| `commit()` | Commit the open transaction (no-op on Postgres). |
| `close()` | Close the connection / pool. |
| `fts_search(query, *, limit, project_ref, entry_type, tags, contributor, team)` | BM25 / tsvector full-text leg. |
| `vector_store(entry_id, embedding)` | Upsert a row in the vector table. |
| `vector_search(embedding, limit, *, project_ref, entry_type, tags, contributor, team)` | KNN by cosine distance; filters pushed down to SQL. |
| `vector_delete(entry_id)` | Delete an entry's embedding. |
| `delete_llm_edges(entry_id)` | Remove LLM-enriched edges for an entry. |
| `vacuum()` | Backend-specific optimization. |
| `next_sequence_value()` | Atomic get-and-increment for `kb-XXXXX` IDs. |
| `transaction()` | `async with`-style atomic transaction (savepoints on nesting). |
| `apply_schema(*, embedding_dim=1024)` | Apply every DDL for this backend. |

Two concrete backends:

### `SQLiteBackend` — `kb_core/db/sqlite_backend.py`

Wraps an `aiosqlite.Connection`. Notable details:

- WAL mode and foreign keys are enabled at open time (see
  `_open_sqlite` in `kb_core/knowledge_base.py`).
- `sqlite-vec` is loaded best-effort; if loading fails, vector search
  is disabled but FTS continues to work.
- `apply_schema(embedding_dim=…)` applies the main schema, the graph
  schema, the ingest schema, the vec schema (sized to
  `embedding_dim`), and the v2 multi-user migration.
- `next_sequence_value` reads-and-updates the `entry_id_seq` row in a
  single connection — safe under SQLite's single-writer model.
- `fts_search` runs a `MATCH` query against `knowledge_fts`, optionally
  joined with `knowledge_entries` for metadata filters.

### `PostgresBackend` — `kb_core/db/postgres_backend.py`

Wraps an `asyncpg.Pool`. Notable details:

- Created via `PostgresBackend.create(dsn, *, pool_min, pool_max,
  password=…, ssl=…)`. The `password` parameter is a `Callable[[], str]`
  so RDS IAM tokens (which expire) can be re-signed per connect.
- `transaction()` reserves a single pool connection in a
  `ContextVar` so nested calls reuse it (asyncpg creates savepoints).
- `commit()` is a no-op — asyncpg auto-commits each statement outside
  a transaction.
- FTS uses a `tsvector` column with `plainto_tsquery('english', …)` and
  `ts_rank_cd`. Scores are negated so that, like the SQLite FTS5 BM25
  result, lower means better.
- Vector search uses `pgvector`'s cosine distance operator `<=>`.
- `apply_schema` runs inside a Postgres advisory lock so concurrent
  process starts converge cleanly on a fresh database.

Both backends honor the same metadata-filter set (`project_ref`,
`entry_type`, `tags`, `contributor`, `team`) on `fts_search` /
`vector_search` so the hybrid RRF caller can trust both legs apply
identical scoping.

---

## 3. Aurora / RDS IAM authentication

`kb_core/db/iam_auth.py` keeps all `boto3` knowledge out of
`PostgresBackend`. It exports three helpers:

- `parse_dsn(url) -> DSNComponents` — extracts `host`, `port` (defaults
  to 5432), and `username` from a `postgresql://` URL. Raises
  `ValueError` if either is missing.
- `make_token_factory(host, port, username, region) -> Callable[[], str]`
  — captures a `boto3.client("rds", region_name=region)` in a closure
  and returns a zero-arg callable that signs a fresh
  `generate_db_auth_token(...)` per call (tokens are valid ~15 minutes).
- `make_ssl_context() -> ssl.SSLContext` — returns
  `ssl.create_default_context()` (`CERT_REQUIRED` + `check_hostname`).

The wiring lives in `_open_postgres` in `kb_core/knowledge_base.py`:
when `PostgresConfig.iam_auth` is `True`, the open path parses the DSN,
builds the token factory and SSL context, and passes both to
`PostgresBackend.create` as `password=…` and `ssl=…`. `boto3` only
needs to be installed when IAM auth is actually enabled (the `iam`
extra).

---

## 4. Full-Text Search (FTS5 / tsvector)

`kb_core/search/fts.py` is a thin wrapper that forwards to
`db.fts_search(...)` and swallows backend exceptions:

```python
async def fts_search(db, query, limit=20, *, project_ref, entry_type, tags, contributor, team):
    if not query.strip():
        return []
    try:
        return await db.fts_search(query, limit=limit, …)
    except Exception:
        logger.warning("FTS search failed for query: %s", query, exc_info=True)
        return []
```

Returns `list[tuple[entry_id, score]]` — for SQLite, the `fts5` `bm25()`
ranking (lower / more-negative = better); for Postgres, the negated
`ts_rank_cd` so the convention matches across backends. The hybrid
fuser does not look at the magnitudes — it only uses the per-leg rank
position (§6).

---

## 5. Vector Embeddings

Three modules cooperate to produce and consume vectors:

### `kb_core/search/embedder_protocol.py`

Defines the structural interfaces the rest of the engine programs
against:

- `Embedder` — `embed(text) -> list[float] | None` plus
  `search_similar(query_embedding, limit, *, project_ref, entry_type,
  tags, contributor, team) -> list[tuple[entry_id, distance]]`.
- `BatchEmbedder` — superset adding `embed_batch(texts) -> list[list[float]]
  | None` and `store_embeddings(pairs) -> None` for the ingest pipeline.

Read paths (hybrid + vector search, dedup) only require `Embedder`;
write paths (ingestion) require the richer `BatchEmbedder` so they can
embed a batch in one HTTP call and persist all vectors together.

### `kb_core/search/embeddings.py` — `EmbeddingClient`

The default concrete embedder. Takes an `EmbeddingConfig`
(`kb_core/config.py`) at construction and a reference to the
`Database` (so it can persist vectors). HTTP transport is
`httpx.AsyncClient`.

**Engine defaults**, all sourced from `EmbeddingConfig` (no
environment reads):

| Field | Default | Source |
| --- | --- | --- |
| `model` | `qwen3-embedding:0.6b` | `_DEFAULT_EMBEDDING_MODEL` |
| `dim` | `1024` | `_DEFAULT_EMBEDDING_DIM` |
| `ollama_url` | `http://localhost:11434` | `_DEFAULT_OLLAMA_URL` |
| `timeout` | `10.0` | `_DEFAULT_EMBEDDING_TIMEOUT` |

The vector-table dimensionality is a separate-but-paired knob:
`SqliteConfig.embedding_dim` / `PostgresConfig.embedding_dim` (both
default `1024`) and the `embedding_dim` keyword on `create_sqlite` /
`create_postgres` are routed straight into `db.apply_schema(embedding_dim=…)`,
which sizes the `vec0` virtual table on SQLite (`FLOAT[<dim>]`) and
the `vector` column on Postgres. The embedder's output dim
(`EmbeddingConfig.dim`) and the storage dim must match — the facade
does not auto-reconcile them.

Notable behavior:

- `is_available()` does a `GET {ollama_url}/api/tags`; success is
  cached, failure is **not** (retries on the next call). The cache is
  invalidated whenever an embed call fails.
- `embed(text)` POSTs to `{ollama_url}/api/embed` with body
  `{"model": …, "input": text}` and expects `{"embeddings": [[...]]}` —
  returning `data["embeddings"][0]`. On any error it returns `None`
  rather than raising.
- `embed_batch(texts)` posts the same endpoint with `"input": texts`,
  expects `len(embeddings) == len(texts)`, and returns `None` on
  mismatch.
- `store_embedding` / `store_embeddings` forward to
  `db.vector_store(...)` and `commit()`.
- `search_similar` forwards directly to `db.vector_search(...)` so the
  same metadata filters (`project_ref`, `entry_type`, `tags`,
  `contributor`, `team`) reach the SQL layer.

### `kb_core/search/vector.py`

A thin function `vector_search(embedder, query, limit, …)` that calls
`embedder.embed(query)` and (on success) `embedder.search_similar(...)`
with the same filter kwargs. Returns `[]` if the embedder returns
`None` from `embed`, so callers can branch on "vector unavailable"
without exceptions.

The whole vector path is optional. Hybrid search runs FTS-only when
either the configured embedder is missing (`embedding=None` on
`KbConfig`) or `is_available()` is currently false.

---

## 6. Hybrid Search + Reciprocal Rank Fusion

`kb_core/search/hybrid.py::hybrid_search` is the single function the
facade and the agentic loop go through to search the KB. It fuses the
FTS and vector legs with **Reciprocal Rank Fusion** (RRF).

Top-of-file constant:

```python
RRF_K = 60  # standard value from the literature
```

The fusion algorithm:

1. **Filter-only short-circuit.** If `query.query` is empty or just
   `"*"` and any metadata filter is set, run `_filter_only_search`
   (which orders by `created_at DESC` and stamps each result with
   `match_source="filter"`) and return. If there is no query *and* no
   filters, return `([], 0)`.
2. **Over-fetch.** Set `fetch_limit = query.limit * 3` and run FTS with
   that limit. The 3× over-fetch gives the RRF re-rank head room.
3. **Optional vector leg.** If an embedder is configured, run
   `vector_search` with the *same* filter kwargs as the FTS leg, so the
   vector leg can't smuggle in wrong-type / wrong-project results past
   the FTS filters. If the vector leg returns at least one row,
   `match_source` upgrades from `"fts"` to `"hybrid"`.
4. **RRF accumulation.** Walk both result lists and accumulate the
   score per `entry_id`:
   ```python
   for rank, (entry_id, _score) in enumerate(fts_results):
       rrf_scores[entry_id] += 1.0 / (RRF_K + rank + 1)
   for rank, (entry_id, _dist) in enumerate(vec_results):
       rrf_scores[entry_id] += 1.0 / (RRF_K + rank + 1)
   ```
   `rank` is the 0-based position in the loop, so the first-place
   entry's contribution is `1 / (60 + 0 + 1) = 1/61`.
5. **Relative score threshold.** Sort descending by `rrf_scores`. If
   `query.min_score_ratio > 0`, compute
   `min_score = top_score * min_score_ratio` and drop everything below.
   `filtered_count` is the number of candidates removed by this filter
   (it is returned to the caller alongside the result list).
6. **Defense-in-depth re-filter.** For each surviving `entry_id`,
   re-fetch the entry, drop inactive entries, and re-apply every
   metadata filter from the query — even though both legs already did
   so at the SQL layer. This is intentional: a future backend bug must
   not be able to leak wrong-scope results.
7. **Expiry + decay.** Drop entries past `expires_at` (unless
   `include_expired`), compute the effective confidence with the decay
   formula (see §8) using `entry.updated_at or entry.created_at` as
   the anchor (so updates reset the clock), and drop entries below
   confidence `0.3` (unless `include_stale`).
8. **Build `SearchResult`s.** Each surviving entry is wrapped in a
   `SearchResult` carrying the fused score, the effective confidence,
   any staleness warning, the `match_source` (`"fts"`, `"hybrid"`, or
   `"filter"`), the per-leg signals (`vector_similarity`,
   `fts_matched`, `fts_rank`), and the entry itself.
9. **Telemetry.** A `search_events` row is recorded fire-and-forget
   with the query text, result count, top score, and `match_source` —
   this is what feeds the search-quality dashboards (the consumer-side
   `kb_maintain` tool reads this table; the engine just writes it).

The function returns `(results, filtered_count)`.

---

## 7. Graph traversal primitives

The graph layer in `kb_core` is deliberately **primitive-only**:
deterministic traversals plus a one-line formatter. Higher-level
orchestration (e.g. "if a search returns fewer than N results, augment
with neighbor entries") belongs to the consumer.

`kb_core/graph/queries.py` exports:

- `get_neighbors(db, node_id, *, edge_types=None, direction="both",
  limit=50)` — returns `(neighbor_id, edge_type, direction)` tuples.
  `direction` is `"outgoing"` or `"incoming"` and is reported per row.
- `bfs_entries(db, start_node, *, max_depth=2, edge_types=None,
  limit=20)` — BFS from `start_node`; collects entry nodes reached at
  each depth as `(entry_id, depth, path)`. The default `max_depth=2`
  is the engine default (consumers may tighten it).
- `find_path(db, source, target, *, max_depth=4)` — BFS shortest path;
  returns `[(node, edge_type, next_node), ...]` or `None`. Default
  `max_depth=4`.
- `entries_for_scope(db, scope, entry_type=None, order_by="created_at")`
  — turn a scope string (`project:X`, `tag:Y`, `person:Z`, `tool:Z`,
  `kb-XXXXX`, or a bare entry-type literal) into a list of entry IDs.
- `supersedes_chain(db, entry_id)` — walk `supersedes` edges in both
  directions; returns the full chain oldest-first.
- `get_graph_vocabulary(db, *, max_nodes=200)` — `{node_type: [names]}`
  for non-entry nodes, ordered by connection count.

These are exposed on the facade as a `_GraphAccessor` (the `graph`
property on `KnowledgeBase`) with `neighbors`, `bfs_entries`,
`find_path`, `supersedes_chain`, `entries_for_scope`, and `vocabulary`
methods — see `kb_core/knowledge_base.py`.

The one engine-side formatter is `kb_core/formatting.py::format_graph_hint`:

```python
def format_graph_hint(entry: KnowledgeEntry, via_node: str) -> str:
    """One-liner hint: See also: [kb-00042] Title (via concept:async-io)."""
    return f"See also: [{entry.id}] {entry.short_title} (via {via_node})"
```

**Important boundary:** `kb_core` does **not** decide when to augment a
sparse result set with graph hints. There is no `collect_graph_hints`
function anywhere under `kb_core/` (verifiable: `grep -r collect_graph_hints
kb_core/` returns nothing). The "fewer than 3 results, so look up
hints" orchestration lives in the `personal_kb` consumer's tools
layer, on top of these primitives.

---

## 8. Confidence Decay + `mental_map` exemption

`kb_core/confidence/decay.py` implements time-based confidence decay.
The formula is exponential, with a per-entry-type half-life:

```python
effective = base * 2 ** (-age_days / half_life)
```

Half-lives, keyed by `EntryType` members:

| `EntryType` | Half-life (days) | Intuition |
| --- | --- | --- |
| `FACTUAL_REFERENCE` | **90** | Facts go stale fast (~3 months). |
| `DECISION` | **365** | Decisions persist but context shifts. |
| `PATTERN_CONVENTION` | **730** | Conventions are durable (~2 years). |
| `LESSON_LEARNED` | **1825** | Hard-won lessons stick (~5 years). |
| _(other / future types)_ | **365.0 default** via `HALF_LIVES.get(entry_type, 365.0)` |

`STALENESS_THRESHOLD = 0.5` — `staleness_warning` returns a warning
string when `effective_confidence < 0.5`, else `None`. Hybrid search
itself uses a lower hard threshold (`< 0.3`) to actually drop entries
from results unless `include_stale=True` is set.

The decay anchor is the **most recent of `created_at` and
`last_accessed`** — entries that keep getting retrieved maintain their
confidence. Both timestamps are normalized to UTC before comparison.
(Inside `hybrid_search`, the anchor passed in is
`entry.updated_at or entry.created_at`, so an explicit update via
`KnowledgeStore.update_entry` also resets the clock.)

### `MENTAL_MAP` is exempt

```python
if entry_type == EntryType.MENTAL_MAP:
    return base_confidence
```

`mental_map` entries are structural orientation nodes, not
value-bearing assertions. They never decay on a clock — there is no
value to go stale. They also do not benefit from read-frequency
self-heal for the same reason. This is the only exempt type.

---

## 9. Entry TTL / Expiry

`kb_core/ttl.py` parses TTL strings and computes absolute expiry
datetimes:

```python
_TTL_PATTERN = re.compile(r"^(\d+)([hdw])$")
```

Supported units (exactly):

| Suffix | Unit |
| --- | --- |
| `h` | hours |
| `d` | days |
| `w` | weeks |

Examples: `"24h"`, `"7d"`, `"2w"`. A zero amount (e.g. `"0d"`) raises
`ValueError("TTL must be greater than zero.")`. Anything else that
doesn't match the regex raises a `ValueError` whose message lists the
allowed units.

```python
def parse_ttl(ttl: str) -> timedelta: ...
def compute_expires_at(ttl: str, now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now(UTC)
    return now + parse_ttl(ttl)
```

`compute_expires_at` returns an absolute UTC `datetime`. The
`expires_at` column on `knowledge_entries` is the result of this
computation; hybrid search drops expired entries unless the query
opts in with `include_expired=True`.

---

## 10. The Knowledge Graph

The graph is stored in two tables:

- `graph_nodes (node_id, node_type, properties, created_at)`
- `graph_edges (id, source, target, edge_type, properties, created_at,
  UNIQUE(source, target, edge_type))`

DDL lives in `apply_graph_schema` (`kb_core/db/schema.py`). Two
distinct code paths write to it.

### Deterministic edges — `kb_core/graph/builder.py`

`GraphBuilder.build_for_entry(entry)` rebuilds an entry's outgoing
edges from purely structural signals. The operation is **delete-and-
rebuild per entry**:

```python
async with self._db.transaction():
    await self._clear_edges_for_source(entry.id)
    # _clear_edges_for_source runs:
    #   DELETE FROM graph_edges WHERE source = ?
    # then re-inserts the current edges
```

What gets re-inserted (each step is best-effort and idempotent):

1. The entry node itself (`node_type="entry"`).
2. **Tags** → `tag:<name>` nodes connected by `has_tag` edges.
3. **Project** → `project:<ref>` node connected by `in_project`.
4. **Supersedes hints** → outgoing `supersedes` edge for every kb-ID
   in `hints["supersedes"]`. Invalid IDs are logged and skipped.
5. **`superseded_by`** → reversed `supersedes` edge from the
   superseder to this entry.
6. **Text references** — every distinct `kb-XXXXX` pattern in
   `knowledge_details` becomes a `references` edge.
7. **`related_entities` hints** — dict-form lets the author pick the
   `edge_type`; string-form defaults to `related_to`.
8. **`person` hints** → `person:<name>` nodes connected by
   `mentions_person`.
9. **`tool` hints** → `tool:<name>` nodes connected by `uses_tool`.

The whole rebuild runs in a single DB transaction so a partially built
graph is never visible.

### LLM enrichment — `kb_core/graph/enricher.py`

`GraphEnricher.enrich_entry(entry)` (and `enrich_batch(entries)`) prompts
the configured **extraction** LLM (see §11) to extract a list of
entity relationships and inserts each as an edge.

The entity-type set is **closed and short**:

```python
_VALID_ENTITY_TYPES = {"person", "tool", "concept", "technology"}
```

Any entity coming back from the LLM whose `entity_type` is outside that
set is dropped during parsing. Relationship labels are deliberately
**open-ended** — the LLM picks the verb (e.g. `uses`, `replaces`,
`depends_on`, `implements`) and the enricher just stamps it onto the
edge's `edge_type`. The LLM prompt explicitly asks for verbs that
describe *how* the entry relates to the entity, not just that a link
exists.

LLM-enriched edges are deletable in a single call via
`db.delete_llm_edges(entry_id)` (used during re-enrichment).

---

## 11. The Entry Pipeline (create / versioning)

Entries are created through `KnowledgeStore.create_entry` and updated
through `KnowledgeStore.update_entry`, both in
`kb_core/store/knowledge_store.py`. Each wraps the multi-table write in
`async with self.db.transaction():` so a partial write rolls back.

`create_entry` imports `next_entry_id` and `insert_version` from
`kb_core/db/queries.py` and builds an `EntryVersion` from
`kb_core/models/version.py`:

```python
from kb_core.db.queries import (
    deactivate_entry_db, get_entry, insert_entry, insert_version,
    next_entry_id, reactivate_entry_db, row_to_entry, update_entry,
)
from kb_core.models.version import EntryVersion
```

The shape of `create_entry`:

1. Open a transaction.
2. `entry_id = await next_entry_id(db)` — mints a fresh `kb-XXXXX`.
3. Build a `KnowledgeEntry` (Pydantic model) with the supplied fields,
   `version=1`, and timestamps set to `now`.
4. `await insert_entry(db, entry)` — writes the row; the FTS triggers
   mirror it into `knowledge_fts`.
5. Build an `EntryVersion(version_number=1, change_reason="Initial
   creation", ...)` and `await insert_version(db, version)`.
6. Fire-and-forget `audit_events` insert.

`update_entry` is structurally similar — fetches the existing row,
checks `is_active`, computes the new version number
(`existing.version + 1`), merges `hints`, optionally clears
`has_embedding` if the content changed, persists the updated entry,
and appends a new `EntryVersion`.

> **Where things live, precisely:** `next_entry_id` and `insert_version`
> live in `kb_core/db/queries.py` and are **imported** by
> `KnowledgeStore`. They do NOT live in `kb_core/store/knowledge_store.py`.
> Similarly, `EntryVersion` is defined in `kb_core/models/version.py`.

The facade `KnowledgeBase.store(...)` is what the rest of the system
calls — it forwards to `KnowledgeStore.create_entry`, then runs the
optional follow-ups (embed, deterministic graph rebuild, optional LLM
enrichment) each of which is best-effort and logs on failure.

---

## 12. Batch storage

`KnowledgeBase.store_batch(entries, *, enrich=True)` in
`kb_core/knowledge_base.py` creates many entries with a single
embedding call and a single enrichment call. The shape:

1. Loop the input dicts; for each, call `KnowledgeStore.create_entry`
   inside the existing per-entry transaction. A failure on any single
   entry is logged and skipped; the rest still proceed.
2. After each successful create, call the deterministic graph builder
   for that entry (`_build_graph`).
3. Once the whole list is created, batch-embed in one
   `embedder.embed_batch(texts)` call. If the embedder is a
   `BatchEmbedder`, persist all `(entry_id, vector)` pairs via
   `embedder.store_embeddings(pairs)` and stamp `has_embedding=True`.
   If it isn't (or `embed_batch` returns `None`), fall back to per-entry
   `_embed_one`.
4. If `enrich=True` and an enricher is configured, call
   `graph_enricher.enrich_batch(created)` and then
   `clear_vocab_cache()`.

Returns the list of created `KnowledgeEntry`s in input order.

---

## 13. File ingestion

The ingest pipeline lives under `kb_core/ingest/`. Five files split
the work:

- **`kb_core/ingest/ingester.py`** — `FileIngester`, the orchestrator.
  Exposes `ingest_file(path)`, `ingest_text(content, source_name)`,
  `ingest_url(url)`, `ingest_url_content(content, source_url)`, and
  `ingest_directory(dir_path)`. Each goes through deny-list +
  extension + symlink + size checks, computes a content hash to skip
  unchanged files, runs PII redaction, and then delegates to a shared
  pipeline that calls the LLM extractor, the optional dedup agent, and
  finally `KnowledgeStore` for the writes.
- **`kb_core/ingest/safety.py`** — `check_deny_list(path)`,
  `detect_secrets_in_content(content)` (uses `detect-secrets` when
  available), `redact_pii(content)` (uses `scrubadub` when available),
  and `run_safety_pipeline(path, content)` which chains them. All
  three optional libraries are off the critical path: if they're
  missing, the pipeline degrades to "no extra safety checks" rather
  than failing.
- **`kb_core/ingest/dedup_agent.py`** — `DedupAgent.check(...)` runs a
  hybrid search against the existing KB with the candidate chunk's
  text and, if the top hit's RRF score exceeds the dedup threshold
  (`IngestConfig.dedup_threshold`, default `0.06`), prompts the LLM to
  decide whether the candidate is a duplicate, an update, or new.
- **`kb_core/ingest/extractor.py`** — `summarize_file(...)` and
  `extract_entries(...)` are the LLM-side prompts that turn raw file
  content into one summary and a list of `ExtractedEntry` records
  ready for storage.
- **`kb_core/ingest/html_extract.py`** — `extract_content(html,
  url=None)` uses `trafilatura` (optional) to pull clean article
  content out of an HTML page; used by `FileIngester.ingest_url`.
- **`kb_core/ingest/chunker.py`** — `chunk_content(text, chunk_size,
  chunk_overlap)` splits oversized files at heading / paragraph
  boundaries with overlap (defaults from `IngestConfig`:
  `chunk_size=16000`, `chunk_overlap=600`).

The orchestrator requires both an extraction LLM and a `BatchEmbedder`.
The facade's `_build_ingester` returns `None` if either is missing;
the user-facing `ingest_*` methods raise `RuntimeError` in that case
with a message pointing at `providers.extraction` and `embedding` on
`KbConfig`.

---

## 14. Dual/multi LLM architecture

`kb_core/config.py::ProviderConfig` carves the engine's LLM use into
three **roles**, each independently configurable:

| Role | Where it's used |
| --- | --- |
| `extraction` | Graph enrichment (`GraphEnricher`) and file-ingest LLM calls (`extractor.py`, `dedup_agent.py`). |
| `query` | Query planning + the agentic ReAct loop (`kb_core/query.py`, `kb_core/graph/planner.py`, `kb_core/graph/agent.py`). |
| `synthesis` | Final-answer synthesis for `summarize` / `ask` answer-mode. |

Each role is a `ProviderRoleConfig` (`provider: "anthropic" | "bedrock"
| "ollama"`, plus an `AnthropicProviderConfig`, `BedrockProviderConfig`,
and `OllamaProviderConfig` held alongside so a runtime switch needs no
extra plumbing).

`synthesis` is a **first-class** `ProviderConfig` field distinct from
`query`. The docstring on `ProviderConfig.synthesis` records the
historical fact that env-driven `personal_kb.config` reuses
`KB_QUERY_PROVIDER` for synthesis — so today's defaults pick the same
provider for both — but in `kb_core` itself the two are wired through
separate fields and the facade builds two distinct LLM clients. Splitting
them is just changing the field; no other plumbing is required.

### `LLMProvider` Protocol — `kb_core/llm/provider.py`

Every provider satisfies a four-method async Protocol:

```python
class LLMProvider(Protocol):
    async def is_available(self) -> bool: ...
    async def generate(self, prompt: str, *, system: str | None = None) -> str | None: ...
    async def generate_chat(self, messages: list[Message], *, system: str | None = None) -> str | None: ...
    async def close(self) -> None: ...
```

`Message` is `dict[str, str]` with `"role"` ∈ `{"user", "assistant"}`
and `"content"`. `None` is the documented "unavailable" return value —
every call site is responsible for handling it (and the engine does:
no LLM means FTS-only search, no enrichment, no dedup).

### Concrete providers (all optional)

- `kb_core/llm/anthropic.py::AnthropicLLMClient` — wraps the
  `anthropic` SDK's Messages API. Requires the `anthropic` extra.
- `kb_core/llm/bedrock.py::BedrockLLMClient` — wraps
  `aws-sdk-bedrock-runtime` (Smithy). Requires the `aws` extra.
  Handles AWS profile resolution, env credential detection, and an
  optional bearer-token auth scheme.
- `kb_core/llm/ollama.py::OllamaLLMClient` — HTTP-only client (uses
  `httpx` from the nucleus). No extra needed beyond a reachable
  Ollama server.

Every provider is optional. Base `kb-core` is **FTS-only and
zero-LLM** — a `KbConfig` with `embedding=None` and providers left at
defaults will still run `store`, `update`, `search`, and the
deterministic graph build; only the LLM-driven features go dark.

---

## 15. Graceful Degradation

Optional dependencies degrade silently across the engine:

- **No embedder configured** (`embedding=None` on `KbConfig`, or
  `is_available()` is currently False) → hybrid search runs FTS-only
  and reports `match_source="fts"`. `KnowledgeBase.embed(...)` and
  `embed_batch(...)` return `None`.
- **`sqlite-vec` extension not loadable** → the SQLite open path logs
  a warning, the vec0 schema is not applied, and vector search calls
  return `[]`. FTS continues unaffected.
- **No extraction LLM** → `KnowledgeBase.graph_enricher` is `None`;
  `_enrich_one` returns immediately; `store_batch`'s enrich branch is
  skipped; the ingest path's `_build_ingester` returns `None`, which
  causes `ingest_*` methods to raise `RuntimeError`.
- **No query LLM** → the agentic ReAct loop and the planner fall back
  to the heuristic auto-search path inside `kb_core/query.py`.
- **No synthesis LLM** → `KnowledgeBase.summarize` returns a
  formatted-entries fallback string built directly from the top
  results.
- **Missing safety libraries** (`detect-secrets`, `scrubadub`) →
  `kb_core/ingest/safety.py` skips the corresponding check rather than
  failing the ingest.
- **Missing HTML extractor** (`trafilatura`) → `ingest_url` returns an
  error result rather than raising; `ingest_url_content` (which takes
  pre-fetched text) still works.
- **Missing PDF reader** (`pymupdf`) → `_read_pdf` raises
  `ImportError`, which `ingest_file` turns into a `FileResult` with
  `action="error"` and a "install pymupdf" reason.
- **`store_batch` per-entry failures** are logged and skipped; the
  rest of the batch still proceeds.

The unifying rule: **engine availability is an explicit property of
the configuration**, not a runtime guess. The facade never silently
swaps providers under the caller's feet — if you want LLM-driven
behavior, you configure it; if you want FTS-only, you pass
`embedding=None` (and leave providers at their unused defaults).

---

## Engine-side query / synthesis primitives

Five modules implement the agentic query path at the primitive level
and are exposed on the facade as `ask`, `summarize`, `preflight`, and
`coverage`:

- `kb_core/query.py` — `retrieve_entries(question, ...)` and
  `synthesize_answer(question, ...)`; the orchestration the facade
  calls.
- `kb_core/graph/planner.py` — `QueryPlanner` produces a `QueryPlan`
  from the question.
- `kb_core/graph/agent.py` — `agentic_query(...)` is the ReAct loop
  (search + graph + refinement, capped by
  `AgenticConfig.max_tool_calls`).
- `kb_core/preflight.py` — `build_project_context(project_ref, *, team,
  since)` returns the compact project-context primer.
- `kb_core/coverage.py` — `assess_coverage(...)` is the post-synthesis
  coverage check.

These are documented here only as engine capabilities. The
**agent-facing UX** of `kb_ask` (strategy choice, MCP tool wiring,
prompt budgeting, output formatting for an MCP channel) belongs to the
`personal_kb` consumer and is documented there. Within `kb_core`,
they are just async functions over the `Database` / `Embedder` /
`LLMProvider` Protocols.

---

## File map (one-line per file referenced)

| Module | Role |
| --- | --- |
| `kb_core/__init__.py` | Re-exports `KbConfig`, `KnowledgeBase`, `create_sqlite`, `create_postgres`, all `*Config` dataclasses. |
| `kb_core/config.py` | `KbConfig` + nested `DatabaseConfig` / `EmbeddingConfig` / `ProviderConfig` / `IngestConfig` / `AgenticConfig` / `Attribution`. |
| `kb_core/knowledge_base.py` | `KnowledgeBase` facade, `_GraphAccessor`, `create_sqlite`, `create_postgres`. |
| `kb_core/formatting.py` | `format_graph_hint(entry, via_node)` and result-list helpers. |
| `kb_core/ttl.py` | `parse_ttl`, `compute_expires_at`, the `^(\d+)([hdw])$` regex. |
| `kb_core/preflight.py` | `build_project_context`, `_maps_sql`, etc. |
| `kb_core/coverage.py` | `assess_coverage`, `CoverageResult`. |
| `kb_core/query.py` | `retrieve_entries`, `synthesize_answer`. |
| `kb_core/db/backend.py` | `Database`, `Cursor`, `Row` Protocols. |
| `kb_core/db/schema.py` | DDL + `apply_*_schema` helpers. |
| `kb_core/db/queries.py` | `next_entry_id`, `insert_entry`, `update_entry`, `get_entry`, `insert_version`, deactivate/reactivate. |
| `kb_core/db/sqlite_backend.py` | `SQLiteBackend` over `aiosqlite` + `sqlite-vec` + FTS5. |
| `kb_core/db/postgres_backend.py` | `PostgresBackend` over `asyncpg` + `pgvector` + tsvector FTS. |
| `kb_core/db/iam_auth.py` | `parse_dsn`, `make_token_factory`, `make_ssl_context`. |
| `kb_core/search/fts.py` | `fts_search` (forwards to backend). |
| `kb_core/search/embedder_protocol.py` | `Embedder` / `BatchEmbedder` Protocols. |
| `kb_core/search/embeddings.py` | `EmbeddingClient` (Ollama `/api/embed`). |
| `kb_core/search/vector.py` | `vector_search` (embed-then-KNN). |
| `kb_core/search/hybrid.py` | `hybrid_search` (RRF fusion, `RRF_K=60`). |
| `kb_core/confidence/decay.py` | `compute_effective_confidence`, `staleness_warning`, `HALF_LIVES`, `STALENESS_THRESHOLD`. |
| `kb_core/graph/builder.py` | `GraphBuilder.build_for_entry` (deterministic edges). |
| `kb_core/graph/enricher.py` | `GraphEnricher` (LLM enrichment), `_VALID_ENTITY_TYPES`. |
| `kb_core/graph/queries.py` | `get_neighbors`, `bfs_entries`, `find_path`, `entries_for_scope`, `supersedes_chain`, `get_graph_vocabulary`. |
| `kb_core/graph/planner.py` | `QueryPlanner`, `QueryPlan`. |
| `kb_core/graph/agent.py` | `agentic_query` ReAct loop. |
| `kb_core/store/knowledge_store.py` | `KnowledgeStore.create_entry` / `update_entry` / `deactivate_entry` / `reactivate_entry` / `bulk_update`. |
| `kb_core/models/entry.py` | `EntryType` (`StrEnum`), `KnowledgeEntry` (`embedding_text` property). |
| `kb_core/models/version.py` | `EntryVersion`. |
| `kb_core/models/search.py` | `SearchQuery`, `SearchResult`. |
| `kb_core/llm/provider.py` | `LLMProvider` Protocol (4 methods). |
| `kb_core/llm/anthropic.py` | `AnthropicLLMClient` (Messages API). |
| `kb_core/llm/bedrock.py` | `BedrockLLMClient` (Smithy / `aws-sdk-bedrock-runtime`). |
| `kb_core/llm/ollama.py` | `OllamaLLMClient` (HTTP). |
| `kb_core/ingest/ingester.py` | `FileIngester` orchestrator + `FileResult` / `IngestResult`. |
| `kb_core/ingest/safety.py` | `check_deny_list`, `detect_secrets_in_content`, `redact_pii`, `run_safety_pipeline`. |
| `kb_core/ingest/dedup_agent.py` | `DedupAgent`, `DedupResult`. |
| `kb_core/ingest/extractor.py` | `summarize_file`, `extract_entries`, `ExtractedEntry`. |
| `kb_core/ingest/html_extract.py` | `extract_content` (HTML → plaintext via `trafilatura`). |
| `kb_core/ingest/chunker.py` | `chunk_content`, `Chunk`. |
