# Personal Knowledge MCP Server — Design Ideas

**Captured**: 2026-02-24
**Status**: Initial scope complete (v0.3.0)

---

## Core Concept

A personal knowledge MCP server that extends Claude's built-in memory with structured retrieval, a knowledge graph, and multi-granularity summaries. Enables true experiential learning across Claude Code sessions while remaining token-efficient.

## Knowledge Entry Schema

### Fields
- **project_ref**: project tag/category (knowledge is global but filterable by project)
- **short_title**: brief identifier
- **long_title**: descriptive title
- **knowledge_details**: full content
- **entry_type**: factual_reference | decision | pattern_convention | lesson_learned
- **timestamps**: created_at, updated_at
- **source_context**: which conversation/file/session spawned this
- **confidence_level**: initial confidence score (decays over time)
- **tags**: freeform categorization

### Entry Types and Their Characteristics
- **Factual references**: version numbers, API endpoints, config values — fast confidence decay
- **Decisions**: "chose X because Y" — slow decay, history is critical ("how we got here")
- **Patterns/conventions**: coding standards, workflow preferences — very slow decay
- **Lessons learned**: mistakes, debugging insights — almost no decay

### Versioning
- Track full history, especially for decision-type entries
- History provides the narrative of "how we got here" vs just "you are here"

### Confidence Decay Model
```
effective_confidence = base_confidence * decay_factor(age, entry_type)
```
- Entries below a confidence threshold get flagged for review or returned with staleness warnings
- Analogous to how humans naturally fact-check older memories

## Storage Layer

### SQLite as unified store
- **sqlite-vec** for vector embeddings
- **FTS5** for BM25/full-text search
- Single-file database, easy to back up and manage

### Hybrid Search
- Combine vector similarity (semantic) with BM25/FTS (keyword/exact match)
- Pure vector misses exact matches; hybrid gives best of both

### Embedding Model
- Local Ollama model (e.g., `nomic-embed-text` or `all-MiniLM-L6-v2`)
- Keeps everything local and fast

## Knowledge Graph + Dynamic Ontology

### Design Principles
- Entities: projects, concepts, decisions, files, people, tools
- Typed edges: uses, depends_on, supersedes, relates_to, etc.
- Ontology emerges from data rather than being predefined
- New entity types and relationship types proposed as knowledge grows

### Graph Building — Backend Responsibility (NOT Claude Code)
Claude Code instances should be knowledge producers and consumers, not graph architects. Reasons:
- **Consistency**: single backend process produces coherent graph vs N sessions with different interpretations
- **Token budget**: graph building tokens are wasted on non-task work
- **Separation of concerns**: working Claude is a task expert; graph builder is a knowledge organization expert

### Structured Hints from Claude
At the point of `kb_store`, Claude passes hints the backend can use:
```python
hints={
    "supersedes": "kb-00142",
    "related_entities": ["pfsense", "firewall-rules"],
    "decision_context": "chose X over Y because Z"
}
```
These are suggestions, not commands. The backend consumes them as strong signals but applies its own ontology rules.

### Tiered Graph Building
1. **Deterministic extraction** (always runs): NER, tag co-occurrence, explicit hints, temporal relationships
2. **Local LLM enrichment** (Ollama): relationship classification, entity disambiguation, ontology proposals
3. **Cloud LLM escalation** (via provider abstraction): complex ontology restructuring, conflict detection, high-quality summarization

### LLM Provider Abstraction
```python
class LLMProvider(Protocol):
    async def complete(self, prompt: str, ...) -> str: ...
```
Implementations for Ollama, Bedrock, Anthropic, OpenAI. Graph builder picks provider based on task complexity and availability.

### Graph Storage
- NetworkX for in-memory manipulation, persisted to SQLite
- KuzuDB as a future option if Cypher queries become valuable

## MCP Tools

### Core Tools (all Claude instances)
- **`kb_store`** — add/update knowledge entry with optional structured hints
- **`kb_search`** — hybrid semantic + keyword search, returns top-N with relevance scores
- **`kb_ask`** — natural language question, backend does multi-hop graph traversal + synthesis
- **`kb_summarize`** — compressed overview of a topic/project at requested granularity

### `kb_ask` — Phased Approach
**Near-term**: structured query with predefined strategies
```python
kb_ask(
    question="Why did we choose PostgreSQL for project X?",
    strategy="decision_trace",  # predefined query pattern
    scope="project:infra-rebuild"
)
```
Strategies: decision_trace, timeline, dependency_map, related_concepts

**Future**: full natural language understanding where the backend translates arbitrary questions to graph traversals using a capable LLM

### Maintenance Tools (KB_MANAGER=TRUE env var only)
Surfaced only to a dedicated maintenance Claude with a specialized CLAUDE.md:
- **`kb_merge`** — combine duplicate/overlapping entries
- **`kb_deprecate`** — mark entries as superseded with a reason
- **`kb_rebuild_graph`** — trigger full graph reconstruction
- **`kb_review_stale`** — surface entries below confidence threshold
- **`kb_ontology_review`** — review and approve proposed ontology changes

## Project Scoping

### Knowledge is Global
- Not partitioned per-project, but tagged with project references
- Enables cross-project knowledge (patterns, lessons) while allowing scoped search
- Example: troubleshooting pfsense doesn't need to comb through Python project knowledge

## Summaries and Compression

### Multi-Granularity Summaries
- Just-in-time context: compressed representations Claude can "look through" to decide if full entry is needed
- Think mermaid diagrams — dense information in a few lines
- Multiple levels: one-liner, paragraph summary, full entry

## Ingestion

### From Claude Code Sessions
- Claude calls `kb_store` during normal work when it learns something worth remembering

### From Disk (Background)
- Point at directories, run background ingestion tasks
- Reads markdown, code, config files
- LLM pass for non-trivial extraction
- Tags with source path and project reference
- Feeds into same graph building pipeline

## Architecture Overview

```
Claude Code instances
    |
    v (MCP tools: kb_store, kb_search, kb_ask, kb_summarize)
+---------------------------+
|   FastMCP Server          |
|   +-- knowledge_store     |--> SQLite (entries + FTS5 + sqlite-vec)
|   +-- search_engine       |      hybrid BM25 + vector
|   +-- query_engine        |      kb_ask / kb_summarize
|   +-- maintenance_tools   |      (KB_MANAGER=TRUE only)
+----------+----------------+
           | (async events on store/update)
           v
+---------------------------+
|   Graph Builder           |
|   +-- deterministic       |--> entity extraction, hints, co-occurrence
|   +-- llm_enrichment      |--> Ollama / Cloud LLM (via provider abstraction)
|   +-- ontology_manager    |--> dynamic type/relationship management
+----------+----------------+
           v
     Knowledge Graph (SQLite or KuzuDB)
```

## Tech Stack
- **Python** + **FastMCP** for the MCP server
- **SQLite** + **sqlite-vec** + **FTS5** for storage and search
- **Ollama** for local embeddings and LLM enrichment
- **NetworkX** for in-memory graph manipulation
- **asyncio** throughout (FastMCP is async-native)
- **LLM provider abstraction** for flexibility (Ollama, Bedrock, Anthropic, OpenAI)

## Roadmap

See [ROADMAP.md](ROADMAP.md).
