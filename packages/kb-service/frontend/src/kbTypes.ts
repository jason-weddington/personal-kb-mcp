// KB domain types — verified against kb-core models (entry.py, search.py, etc.)
// These use snake_case to match the raw server JSON (KB routes bypass camelCase conversion).

export type EntryType =
  | 'factual_reference'
  | 'decision'
  | 'pattern_convention'
  | 'lesson_learned'
  | 'mental_map'

export type MatchSource = 'hybrid' | 'fts' | 'vector'

/** 22-field verified shape from kb-core src/kb_core/models/entry.py */
export interface KnowledgeEntry {
  id: string
  project_ref: string | null
  short_title: string
  long_title: string
  knowledge_details: string
  entry_type: EntryType
  source_context: string | null
  contributor: string | null
  team: string | null
  updated_by: string | null
  sensitivity: 'internal' | 'restricted' | 'public' | null
  confidence_level: number // 0.0–1.0, default 0.9
  tags: string[]
  hints: Record<string, unknown>
  created_at: string | null
  updated_at: string | null
  last_accessed: string | null
  expires_at: string | null
  superseded_by: string | null
  is_active: boolean
  has_embedding: boolean
  version: number
}

// ── Search ──────────────────────────────────────────────────────────────────

export interface SearchRequest {
  query: string
  project_ref?: string
  entry_type?: string
  tags?: string[]
  limit?: number
  include_stale?: boolean
  include_expired?: boolean
}

export interface SearchResult {
  entry: KnowledgeEntry
  score: number
  effective_confidence: number
  staleness_warning: string | null
  match_source: MatchSource
}

export interface SearchResponse {
  results: SearchResult[]
  filtered_count: number
}

// ── Get ─────────────────────────────────────────────────────────────────────

export interface PointerRotTarget {
  target_id: string
  superseded_by: string | null
}

export interface GetEntryResult {
  id: string
  found: boolean
  entry: KnowledgeEntry | null
  pointer_rot: PointerRotTarget[]
}

export interface GetResponse {
  results: GetEntryResult[]
}

// ── Projects ─────────────────────────────────────────────────────────────────

export interface KbProject {
  name: string
  entry_count: number
}

export interface KbListResponse {
  items: KbProject[]
}

// ── Maps ─────────────────────────────────────────────────────────────────────

export interface MapRef {
  id: string
  short_title: string
  long_title: string
}

export interface ProjectMaps {
  project_ref: string
  maps: MapRef[]
}

export interface MapsIndexResponse {
  projects: ProjectMaps[]
}

// ── Graph ────────────────────────────────────────────────────────────────────

export interface GraphNeighbor {
  neighbor_id: string
  edge_type: string
  direction: string
}

export interface GraphNeighborsResponse {
  neighbors: GraphNeighbor[]
}

export interface GraphFullNode {
  id: string
  label: string
  type: string
  val: number
  properties: Record<string, unknown>
}

export interface GraphFullEdge {
  source: string
  target: string
  type: string
  properties: Record<string, unknown>
}

export interface GraphFullStats {
  node_count: number
  edge_count: number
}

export interface GraphFullResponse {
  nodes: GraphFullNode[]
  edges: GraphFullEdge[]
  stats: GraphFullStats
}

// ── Chat REST ────────────────────────────────────────────────────────────────

export interface ChatListItem {
  id: string
  title: string
  mode: string
  updated_at: string
}

export interface ChatMessageItem {
  role: string
  content: string
}

export interface ChatOkResponse {
  ok: boolean
  id: string | null
}

// ── SSE event payload types (Ask / query/stream) ─────────────────────────────

export interface ClassifiedEvent {
  mode: string
}

export interface StatusEvent {
  message: string
}

export interface SynthesisResultEvent {
  answer: string
  question: string
  entry_ids: string[]
}

export interface AskEntry {
  id: string
  short_title: string
  entry_type: EntryType
  tags: string[]
  context: string
}

export interface EntriesEvent {
  entries: AskEntry[]
  turns_used: number
}

// ── SSE event payload types (Chat / chat/stream) ──────────────────────────────

export interface ChatSessionEvent {
  session_id: string
}

export interface ChatResponseEvent {
  answer: string
  session_id: string
}

export interface ChatDoneEvent {
  new_entries: string[]
}

export interface ChatToolResultEvent {
  tool: string
  success: boolean
  entry_ids: string[]
}
