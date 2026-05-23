# personal-kb — Domain Model

## Core Terminology

**Knowledge Entry**: The atomic unit of information in the KB. Each entry is a structured,
typed piece of knowledge with full provenance: who stored it, when, from which project, and
how confident the system is that it remains accurate. Identified by a sequential `kb-XXXXX` ID.

**Entry Type**: A required classification that determines the entry's semantic role and its
confidence decay rate. Four types are defined:

| Type | Meaning | Decay Half-Life |
|---|---|---|
| `factual_reference` | Facts, API behaviors, config values, version constraints | 90 days |
| `decision` | "Chose X because Y" — the rationale matters as much as the choice | 365 days |
| `pattern_convention` | Coding standards, workflow preferences, architectural patterns | 730 days |
| `lesson_learned` | Debugging insights, mistakes made, hard-won workarounds | 1825 days |

**project_ref**: A free-text string scoping an entry to a project context. Used as a filter in
`kb_search`, `kb_ask`, and `kb_summarize`. Typically matches the repository name or a short
project identifier (e.g., `personal-kb`, `agent-gtd`). Not enforced — agents can store entries
without a project_ref for cross-project knowledge.

**Confidence Level**: A float 0.0–1.0 set at creation time (default 0.9). Represents the
initial trustworthiness of the entry. Decays over time via exponential decay applied at read
time — older entries appear less confident even if the stored value hasn't changed.

**Effective Confidence**: The time-decayed confidence at query time:
```
effective = base_confidence × 2^(-age_days / half_life)
```
where `age_days` is measured from `max(created_at, last_accessed)`. Reading an entry via
`kb_get` updates `last_accessed`, resetting the decay clock.

**TTL (Time-to-Live)**: An optional expiry duration set at creation (`7d`, `24h`, `2w` formats).
Entries past their `expires_at` timestamp are excluded from search results by default. Useful
for time-sensitive reference data like sprint goals or temporary configurations.

**Hints**: A JSON dict that directs the knowledge graph builder. The `{"supersedes": "kb-XXXXX"}`
hint creates a directed edge from the new entry to the one it replaces, enabling decision traces.
Other hint keys (e.g., `{"person": "jason"}`, `{"tool": "sqlite"}`) are used by the graph
enricher as entity linking signals.

**Tags**: A list of freeform strings for categorization and search filtering. Unlike project_ref
(single scope), an entry can have multiple tags. Tags become graph nodes — entries sharing a tag
are automatically connected.

**Graph Node**: Any entity in the knowledge graph. Entry nodes have `kb-XXXXX` IDs. Intermediate
nodes (tags, concepts, entity names) have plain string IDs. Both entry and non-entry nodes
participate in graph traversal.

**Staleness Warning**: A text badge surfaced in search results when effective confidence drops
below 0.5. Signals that the information may be outdated and should be verified before acting on.

**Sensitivity**: An optional classification field (`internal`, `restricted`, `public`).
Currently informational only — no access controls are enforced based on this field.

## Business Rules

1. **Entry IDs are permanent**: Once assigned, a `kb-XXXXX` ID is never recycled or changed.
   Entries are deactivated (`is_active=False`) rather than deleted. This preserves graph edge
   integrity and audit trails.

2. **Version increments on every update**: The `version` field starts at 1 and is atomically
   incremented. Consumers checking for changes can compare versions.

3. **Supersession is directional**: When entry B supersedes entry A, store `{"supersedes": "kb-XXXXX"}`
   in B's hints. The graph records the edge B → A. A is typically deactivated via
   `deactivate_entry_id` in the same `kb_store` call. `decision_chain` traversal follows these
   edges to reconstruct history.

4. **Secret scanning before LLM sees content**: During ingestion, each chunk is scanned for
   secrets (API keys, tokens, passwords) before being passed to the extraction LLM. Chunks
   with detected secrets are skipped entirely. This rule applies even when `KB_AGENTIC_INGEST`
   is disabled.

5. **Deduplication before store during ingestion**: When `KB_AGENTIC_INGEST=TRUE`, `DedupAgent`
   hybrid-searches the KB and asks the LLM to confirm whether the new content duplicates an
   existing entry. Confirmed duplicates are skipped. The threshold is configurable via
   `KB_INGEST_DEDUP_THRESHOLD` (default 0.06 RRF score).

6. **Embedding text is deterministic**: The text sent to the embedder is always
   `f"{short_title} {long_title} {knowledge_details}"` (the `embedding_text` property on
   `KnowledgeEntry`). This ensures consistent retrieval regardless of when an entry was embedded.

7. **Contributor and team are server-side injected**: The `contributor` and `team` fields are
   populated from `KB_CONTRIBUTOR` and `KB_TEAM` env vars at the server level. Callers cannot
   forge attribution.

8. **Manager-only tools are gated**: `kb_maintain` and `kb_bulk_update` are only registered
   when `KB_MANAGER=TRUE`. Regular (read/write/query) tools are always registered.

9. **Tool name prefix varies by instance role**: When `KB_INSTANCE_ROLE=personal`, all tool
   names are prefixed `personal_kb_` (e.g., `personal_kb_search`). When `team`, they are
   prefixed `team_kb_`. Default is unprefixed `kb_`. This allows two KB instances to coexist
   in the same MCP session.

## User Roles

**Agent (primary consumer)**: AI coding agents (Claude Code, etc.) store, search, and retrieve
knowledge. They call tools directly via MCP. Agents are expected to search before acting and
store after learning. The KB is optimized for agent use patterns: compact output by default,
full content on demand.

**Human operator (secondary consumer)**: Via the Graph Explorer web UI (`kb_explore` →
`http://localhost:8765`). The UI provides force-directed graph visualization, natural-language
chat with the KB, and write-back tools (update entries, ingest URLs, upload files).

**Manager**: An agent or human operator running with `KB_MANAGER=TRUE`. Has access to
maintenance actions (`kb_maintain`): rebuild embeddings, rebuild graph, vacuum database,
deactivate/purge entries, view feedback and search telemetry. Not intended for regular usage.

**Contributor / Team**: Identity fields attributed to stored entries. Set via `KB_CONTRIBUTOR`
and `KB_TEAM`. Used for filtering in search and for team-based attribution badges in results.
Not a security boundary — attribution is informational.

## Process Flows

### Storing a decision

1. Work session concludes with an architectural choice.
2. Agent calls `kb_store(short_title="...", long_title="...", knowledge_details="rationale...",
   entry_type="decision", project_ref="my-project", hints={"supersedes": "kb-00042"})`.
3. KB assigns `kb-XXXXX`, persists entry.
4. Graph: deterministic edge `kb-XXXXX → kb-00042` (supersedes). Tags → entry edges added.
5. LLM enricher extracts entities (`tool`, `concept`), adds relationship edges.
6. Old entry (`kb-00042`) remains active unless `deactivate_entry_id="kb-00042"` was passed.

### Retrieving related decisions

1. New session starts on the same project.
2. Agent calls `kb_ask("decisions about authentication", strategy="decision_trace")`.
3. `QueryPlanner` produces a plan: strategy=decision_trace, target=latest auth decision.
4. Agent loop: `hybrid_search("authentication decision")` → finds `kb-XXXXX`.
5. Agent loop: `decision_chain("kb-XXXXX")` → follows supersedes chain → returns history.
6. Agent loop: `done(entry_ids=["kb-XXXXX", "kb-00031"])` → returns all entries in the chain.
7. Agent reviews the full decision history with confidence scores and staleness warnings.

### Ingesting a documentation file

1. Agent calls `kb_ingest("docs/architecture.md", project_ref="my-project")`.
2. File validated (extension allowed, no symlink, under 10MB).
3. Chunked at H2 boundaries (16,000 chars, 600 char overlap).
4. Per chunk: secret scan → if clean, extraction LLM produces `ExtractedEntry` list.
5. Per extracted entry: dedup check → if not duplicate, store via standard store flow.
6. Summary returned: N files ingested, M entries created, K skipped (duplicate/flagged).

### Discovering unknown-unknown knowledge

1. Agent starts a new session on a project.
2. Agent calls `kb_preflight(question="starting work on authentication refactor")`.
3. Server injects CWD-matched project context: expiring entries, recent decisions, active conventions.
4. Agent reviews the briefing before beginning work — discovers the auth decision from last week
   and the convention about JWT expiry that might otherwise have been missed.

## Domain Models

### KnowledgeEntry

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | generated | `kb-XXXXX` format |
| `short_title` | `str` | required | Brief identifier for compact display |
| `long_title` | `str` | required | Descriptive title (1-2 sentences) |
| `knowledge_details` | `str` | required | Full entry content |
| `entry_type` | `EntryType` | required | factual_reference / decision / pattern_convention / lesson_learned |
| `project_ref` | `str \| None` | `None` | Project scope filter |
| `source_context` | `str \| None` | `None` | Provenance: URL, file path, session description |
| `contributor` | `str \| None` | from env | Creator (KB_CONTRIBUTOR) |
| `team` | `str \| None` | from env | Team attribution (KB_TEAM) |
| `updated_by` | `str \| None` | from env | Last updater identity |
| `sensitivity` | `str \| None` | `None` | internal / restricted / public (informational) |
| `confidence_level` | `float` | `0.9` | Base confidence 0.0–1.0 |
| `tags` | `list[str]` | `[]` | Freeform categorization tags |
| `hints` | `dict` | `{}` | Graph hints (supersedes, person, tool, related_entities) |
| `created_at` | `datetime \| None` | auto | Creation timestamp |
| `updated_at` | `datetime \| None` | auto | Last update timestamp |
| `last_accessed` | `datetime \| None` | auto | Last kb_get access (decay anchor) |
| `expires_at` | `datetime \| None` | `None` | TTL expiry; excluded from search after this |
| `superseded_by` | `str \| None` | `None` | Entry ID that replaced this one |
| `is_active` | `bool` | `True` | Soft-delete flag |
| `has_embedding` | `bool` | `False` | Whether a vector embedding is stored |
| `version` | `int` | `1` | Incremented on every update |

### EntryType (enum)

```python
class EntryType(StrEnum):
    FACTUAL_REFERENCE = "factual_reference"
    DECISION = "decision"
    PATTERN_CONVENTION = "pattern_convention"
    LESSON_LEARNED = "lesson_learned"
```

### Hints dict schema

The `hints` field accepts any JSON-serializable dict. Known keys with defined semantics:

| Key | Type | Example | Effect |
|---|---|---|---|
| `supersedes` | `str` | `"kb-00042"` | Creates supersedes edge; enables decision_chain traversal |
| `person` | `str` | `"jason"` | Graph node link to a person entity |
| `tool` | `str` | `"sqlite"` | Graph node link to a tool entity |
| `related_entities` | `list[dict]` | `[{"id": "kb-00003", "edge_type": "depends_on"}]` | Explicit cross-entry edges |

Unknown keys are stored and passed to the LLM enricher as additional context.

### SearchResult

| Field | Type | Description |
|---|---|---|
| `entry` | `KnowledgeEntry` | The matched entry |
| `score` | `float` | RRF combined score (higher = better) |
| `effective_confidence` | `float` | Time-decayed confidence at query time |
| `staleness_warning` | `str \| None` | Warning text if effective_confidence < 0.5 |
| `match_source` | `str` | `hybrid` / `fts` / `vector` |

### SearchQuery

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search text |
| `project_ref` | `str \| None` | `None` | Scope to project |
| `entry_type` | `EntryType \| None` | `None` | Filter by type |
| `tags` | `list[str] \| None` | `None` | All must match (AND) |
| `contributor` | `str \| None` | `None` | Filter by attribution |
| `team` | `str \| None` | `None` | Filter by team |
| `limit` | `int` | `10` | Max results (1–50) |
| `include_stale` | `bool` | `False` | Include low-confidence entries |
| `include_expired` | `bool` | `False` | Include past-TTL entries |
| `min_score_ratio` | `float` | `0.5` | Relative threshold (fraction of top score) |

## Pointers

- `src/personal_kb/models/entry.py` — `KnowledgeEntry` model, `EntryType` enum
- `src/personal_kb/models/search.py` — `SearchQuery`, `SearchResult` models
- `src/personal_kb/confidence/decay.py` — `compute_effective_confidence()`, half-lives
- `src/personal_kb/tools/ttl.py` — `parse_ttl()`, `compute_expires_at()`
- `src/personal_kb/graph/builder.py` — deterministic hint-based graph edges
- `src/personal_kb/graph/enricher.py` — LLM entity extraction and relationship types
- `src/personal_kb/graph/queries.py` — `supersedes_chain()`, `bfs_entries()`, `get_neighbors()`
- `src/personal_kb/store/knowledge_store.py` — `KnowledgeStore` CRUD and ID generation
- `src/personal_kb/db/schema.py` — full column definitions for `knowledge_entries`
- `src/personal_kb/server.py` — `_INSTRUCTIONS` string: the tool usage guide injected into MCP
