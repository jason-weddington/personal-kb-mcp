# Scaling to Corpus-Level Ingestion

Personal KB was designed for a specific use case: AI agents capturing *new* knowledge as they work. An agent fixes a thorny bug? It stores the root cause and fix in the KB so the next agent (or the next session) doesn't repeat the investigation. A team makes an architectural decision? It goes in the KB with the rationale, so future work respects the context.

This is a small-write, many-read workload. A typical deployment stores hundreds to low thousands of entries, written one at a time as agents encounter things worth remembering. The ingestion pipeline reflects this: it processes one file at a time, calls an LLM for summarization, extraction, dedup, and graph enrichment on every entry, and generates embeddings sequentially. For a single user or small team, the pipeline completes in seconds per file and the total knowledge base fits comfortably in SQLite.

The question this document addresses: **what would it take to ingest a large existing corpus** — hundreds of thousands of documents, potentially in multiple languages — into Personal KB? The answer is that the current architecture needs significant changes at that scale. This document maps out exactly what breaks, what can be fixed incrementally, and what requires architectural work.

## What breaks at scale

Consider a corpus of 150,000 documents in 12 languages — roughly 1.8 million pages. Running these through the current pipeline:

| Metric | Single file (current) | 1.8M pages |
|--------|----------------------|------------|
| Chunks (~3/page) | 3 | ~5.4M |
| Extracted entries (~3/chunk) | 8 | ~16M |
| LLM calls | 12-15 | **21-27M** |
| Embedding calls | 8 | **~16M** |
| DB writes | ~315 | **~500M** |
| Graph edges | ~40 | **~100M** |

### LLM calls are the dominant cost

The current pipeline makes 12-15 LLM calls per file:

| Stage | Calls per file | Purpose |
|-------|---------------|---------|
| Summarize | 1 | 2-3 sentence summary of the file |
| Extract (per chunk) | 3 | Structured entries from raw text |
| Dedup (per chunk, conditional) | 0-3 | Check if chunk duplicates existing entries |
| Graph enrich (per entry) | 8 | LLM-extracted entity relationships |

At 1.8M files, this totals 21-27 million LLM calls. Even with a fast model (Claude Haiku at ~2s per call), sequential execution takes **1.7 years**. At ~$0.001 per call, the API cost alone is ~$25K.

### Embeddings are the second bottleneck

The embedding client (`search/embeddings.py`) generates one embedding per entry via a single HTTP call to Ollama. There is no batch endpoint. For 16M entries at ~100ms each, that is **18 days** of sequential embedding generation.

### The database write pattern doesn't batch

Each entry creation involves ~6 separate commits: entry insert, version record, audit event, embedding store (delete + insert), embedding flag update, and graph writes. For 16M entries, that is ~96M individual commits. Row-by-row INSERT with per-row commits is orders of magnitude slower than bulk loading.

### The graph vocabulary query has a correlated subquery

`get_graph_vocabulary()` in `graph/queries.py` counts connections per node using a correlated subquery:

```sql
SELECT n.node_id, n.node_type,
  (SELECT COUNT(*) FROM graph_edges WHERE source = n.node_id OR target = n.node_id) AS conn_count
FROM graph_nodes n
WHERE n.node_type != 'entry'
ORDER BY conn_count DESC
LIMIT ?
```

At 100M edges, each subquery does a scan of the edges table. Even with the current `max_nodes=200` cap, this query would take minutes to hours. The graph visualization endpoint (`explorer/graph_data.py`) uses the same pattern and would become completely unusable.

### Entity dedup in the enricher is O(N * M)

`_resolve_node_id()` in `graph/enricher.py` compares each new entity name against the full graph vocabulary using `difflib.SequenceMatcher.ratio()`. With 16M entries producing ~96M entities and a vocabulary that grows into the tens of thousands, this becomes hundreds of billions of string comparisons — days of pure CPU time.

### No parallelism exists in the pipeline

The codebase has zero concurrent file processing — no `asyncio.gather`, no worker pools, no semaphores. The ingester processes one file at a time, and within each file, chunks are processed sequentially (each chunk's extraction context depends on the prior chunk's output). The Postgres connection pool defaults to 5 connections.

## Optimizations for the current use case

Before addressing corpus-scale ingestion, there are five changes that improve performance for the existing single-user / small-team use case. These are worth doing regardless of scale.

### 1. Use batch enrichment during ingestion

The ingester calls `enrich_entry()` individually for each extracted entry (`ingester.py:829`), but `enrich_batch()` already exists (`enricher.py:126`) and is used by `kb_store_batch`. Batch enrichment combines all entries into a single LLM call and loads the vocabulary cache once instead of once per entry.

For a file producing 8 entries, this reduces graph enrichment from 8 LLM calls to 1, and eliminates 7 redundant vocabulary cache loads.

**Estimated impact:** 8 LLM calls per file reduced to 1. Wall clock savings of ~10-15 seconds per file.

### 2. Add batch embedding support

The Ollama `/api/embed` endpoint accepts array input, but `EmbeddingClient.embed()` only sends one string at a time. Adding an `embed_batch()` method and calling it after all entries are extracted would turn N sequential HTTP round-trips into 1.

For a file producing 8 entries, this eliminates 7 HTTP round-trips to Ollama.

**Estimated impact:** 8 embedding calls reduced to 1. Saves ~700ms per file (at ~100ms per call).

### 3. Fix the vocabulary cache lifecycle

`enrich_entry()` loads the vocabulary cache at line 114 and clears it at line 123 — on every single call. When called in a loop during ingestion, this means one `get_graph_vocabulary()` database query per entry, returning the same data each time.

Using `enrich_batch()` (optimization 1) fixes this automatically, since the batch method loads the cache once. For any remaining per-entry enrichment paths, the cache should persist across calls on the same `GraphEnricher` instance rather than being cleared after each use.

**Estimated impact:** Eliminates N-1 redundant DB queries per file ingestion (typically 7 for an 8-entry file).

### 4. Wrap entry creation in a transaction

`KnowledgeStore.create_entry()` performs 3 separate commits per entry (insert, version, audit). Graph building uses a transaction, but the store operations before it do not. Wrapping the full create-embed-graph cycle in a single transaction per entry — or better, per file — would reduce commit overhead significantly.

**Estimated impact:** ~48 commits per 8-entry file reduced to ~3. Marginal for SQLite (single-writer), meaningful for Postgres.

### 5. Add a prefix index for entity dedup

`_resolve_node_id()` iterates all vocabulary names with `SequenceMatcher.ratio()` per entity. A simple prefix index — grouping vocabulary by 3-character prefix and only comparing within the matching bucket — reduces comparisons from O(N) to O(1-10) per lookup while preserving the exact 0.85 similarity threshold. This is 30 lines of stdlib Python with no new dependencies.

At the current vocabulary cap of 200 nodes, this is barely noticeable. But it future-proofs the enricher against vocabulary growth and eliminates a known O(N * M) hotspot identified in the codebase audit.

**Estimated impact:** 80-100x speedup for entity dedup at vocabulary sizes above ~500. Negligible at current scale, but prevents a wall as the KB grows.

## Architecture for corpus-scale ingestion

The optimizations above help the personal/team use case but are insufficient for millions of documents. Corpus-scale ingestion requires architectural changes across four areas.

### Pipeline stage elimination

Not every pipeline stage is needed for a bulk initial load. The stages fall into three tiers:

**Essential (cannot skip):**
- **Chunking** — Drives extraction granularity. Pure string manipulation, essentially free.
- **Extraction** — The only LLM call that is irreducible. Produces the structured entries that everything downstream depends on.
- **Storage** — Entries must exist in the database for search to work.
- **Deterministic graph building** — Tags, projects, supersedes edges, text references. Pure DB writes, no LLM. Covers ~90% of graph query needs.

**Deferrable (run as a background job after initial load):**
- **Embeddings** — Without them, search falls back to FTS-only (still functional). Batch-generate embeddings offline after the entries are loaded. The codebase already gracefully degrades when embeddings are missing.
- **Graph enrichment** — LLM-extracted entity edges are useful but not essential. Run enrichment selectively on a subset of entries (e.g., one language only) as a separate pass.

**Skippable for initial load:**
- **Summarization** — File summaries are stored in `ingested_files` metadata but never used in search or retrieval. Skip entirely for bulk load.
- **Dedup agent** — On a fresh database, there is nothing to deduplicate against. The hybrid search calls are wasted I/O. Disable via `KB_AGENTIC_INGEST=FALSE` for the initial load; re-enable for incremental updates.

Eliminating the skippable stages and deferring embeddings/enrichment removes ~70% of LLM calls (19M of 27M), leaving only the ~5.4M extraction calls as the critical path.

### Parallel file processing

Files are independent — there are no cross-file dependencies in the ingestion pipeline. (Within a file, chunks are sequential because each chunk's extraction context references prior chunks.) This makes file-level parallelism straightforward:

```
                    +-- Worker 1: file_001 -> chunk -> extract -> store -> graph
Orchestrator -------+-- Worker 2: file_002 -> chunk -> extract -> store -> graph
(semaphore=N)       +-- Worker 3: file_003 -> chunk -> extract -> store -> graph
                    +-- ...N workers
```

**Requirements:**
- Postgres backend (SQLite is single-writer)
- Connection pool sized to match worker count (`KB_PG_POOL_MAX=50-100`)
- `asyncio.Semaphore` to cap concurrent LLM calls and prevent rate-limit exhaustion
- Per-worker `GraphEnricher` instances (the vocabulary cache on the shared instance has race conditions)
- Postgres entry ID generation is already atomic (`UPDATE ... RETURNING`), so concurrent workers won't produce duplicate IDs

**Expected throughput:** At ~10 seconds per file with 50 concurrent workers: 1.8M files in ~4 days. With 100 workers: ~2 days.

### Database bulk loading

Row-by-row INSERT with per-row commits is the wrong pattern for millions of entries. Postgres-specific optimizations that would apply:

| Optimization | Current | Bulk alternative | Speedup |
|-------------|---------|-----------------|---------|
| Insert method | Individual INSERT | COPY FROM (binary) | ~67x |
| Indexes | Always on | Disable during load, rebuild after | 5-10x |
| Commits | ~6 per entry | Batch per file or per N entries | 10x |
| Vector inserts | DELETE + INSERT per row | Batch COPY into pgvector | ~50x |
| Entry IDs | One-at-a-time | Pre-allocate ranges per worker | 10x |
| Index rebuild | N/A | `max_parallel_maintenance_workers=7` | 3-4x |

The combination of COPY FROM + disabled indexes + batch commits is a well-documented pattern for pgvector bulk loading.

### Graph at scale

The graph layer needs three changes to function at 100M+ edges:

**1. Pre-computed connection counts.** Replace the correlated subquery in `get_graph_vocabulary()` with a materialized stats table:

```sql
CREATE TABLE graph_node_stats (
    node_id TEXT PRIMARY KEY,
    connection_count INT NOT NULL
);
CREATE INDEX idx_stats_conn ON graph_node_stats(connection_count DESC);
```

Update the stats table after each batch of inserts. The vocabulary query becomes a simple indexed lookup instead of N subqueries on the edges table.

**2. Composite indexes.** The current schema has single-column indexes on `graph_edges.source`, `graph_edges.target`, and `graph_edges.edge_type`. Queries that filter on combinations (which is most of them) need composite indexes:

```sql
CREATE INDEX idx_edges_src_tgt ON graph_edges(source, target, edge_type);
CREATE INDEX idx_edges_tgt_src ON graph_edges(target, source, edge_type);
```

**3. Subgraph-scoped visualization.** The explorer's `extract_graph_data()` currently fetches the entire graph. At 100M edges, this needs to be replaced with a starting-node query that fetches the connected component within N hops, not the full graph.

### Multilingual dedup strategy

For a corpus where the same content exists in multiple languages, the recommended approach is to store each unique semantic chunk once and tag it with language metadata, rather than storing 12 separate copies. Content-hash-based deduplication (normalize text, hash, check for duplicates before extraction) across languages dramatically reduces the entry count — potentially from 1.8M to ~150K unique semantic units plus language metadata.

### Embedding at scale

For millions of entries, local Ollama on a single machine is insufficient. Production approaches include:

- **Multiple embedding model replicas** across GPU nodes (reported 9x speedup with 20 replicas)
- **Batch processing** via Ray, Spark, or similar distributed compute frameworks
- **COPY-based vector loading** into pgvector with indexes disabled during load

The embedding step is embarrassingly parallel (each entry is independent) and benefits most from horizontal scaling.

## Summary

| Scale | Architecture | Key constraints |
|-------|-------------|-----------------|
| **Personal** (100s of entries) | SQLite, single-file pipeline, all stages enabled | Current architecture, works as-is |
| **Small team** (1000s of entries) | SQLite or Postgres, single-file pipeline, optimizations 1-5 applied | Batch enrichment + embeddings, prefix index, transaction batching |
| **Corpus** (millions of entries) | Postgres required, parallel workers, bulk loading, deferred stages | Skip summarize/dedup/enrichment on initial load, COPY FROM, distributed embedding, graph schema changes |

The five personal/team optimizations (batch enrichment, batch embeddings, vocab cache fix, transaction batching, prefix index) are incremental improvements to the existing codebase. Corpus-scale ingestion is an architectural change that requires a parallel orchestrator, Postgres bulk loading support, graph schema changes, and a distributed embedding strategy.
