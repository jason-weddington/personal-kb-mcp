// Pure model for the Graph page's query mode: SSE events from
// POST /api/kb/query/stream become queued reveal steps that are applied one
// at a time so the agent's traversal animates onto the graph.
import type { AskEntry } from '../kbTypes'
import { EMPTY_TRAVERSAL } from './explorerModel'
import type { NodeState, QueryMode, TraversalState } from './explorerModel'

export const REVEAL_DELAY_MS = 250
export const VISITED_FIT_MS = 500
export const VISITED_FIT_PADDING = 60
export const RESULTS_FIT_MS = 600
export const RESULTS_FIT_PADDING = 40
export const STATUS_CLASSIFYING = 'Classifying query...'
export const STATUS_EXPLORING = 'Exploring knowledge graph...'
export const STATUS_SUMMARIZING = 'Preparing answer...'
export const NOT_IN_GRAPH_SAMPLE = 5

export type QueryPhase = 'idle' | 'streaming' | 'done' | 'error'

export type RevealStep =
  { kind: 'mark'; nodeId: string; state: NodeState } | { kind: 'fitResults' } | { kind: 'end' }

export interface QueryDiag {
  eventCounts: Readonly<Record<string, number>>
  unknownEvents: readonly string[]
  malformed: Readonly<Record<string, number>>
  marksVisited: number
  marksResult: number
  marksDeduped: number
  turnsUsed: number | null
  topScore: number | null
}

export interface QueryState {
  phase: QueryPhase
  question: string
  queryMode: QueryMode
  statusMessage: string
  nodeStates: ReadonlyMap<string, NodeState>
  pulseStarts: ReadonlyMap<string, number>
  pending: readonly RevealStep[]
  lastApplied: { seq: number; step: RevealStep } | null
  answer: string | null
  sourceIds: string[]
  entries: AskEntry[] | null
  error: string | null
  streamEnded: boolean
  diag: QueryDiag
}

export type QueryAction =
  | { type: 'start'; question: string }
  | { type: 'event'; event: string; data: Record<string, unknown> }
  | { type: 'tick'; now: number }
  | { type: 'reveal'; ids: string[] }
  | { type: 'streamClosed' }
  | { type: 'fail'; message: string }
  | { type: 'close' }

export const initialQueryDiag: QueryDiag = {
  eventCounts: {},
  unknownEvents: [],
  malformed: {},
  marksVisited: 0,
  marksResult: 0,
  marksDeduped: 0,
  turnsUsed: null,
  topScore: null,
}

export const initialQueryState: QueryState = {
  phase: 'idle',
  question: '',
  queryMode: null,
  statusMessage: '',
  nodeStates: new Map(),
  pulseStarts: new Map(),
  pending: [],
  lastApplied: null,
  answer: null,
  sourceIds: [],
  entries: null,
  error: null,
  streamEnded: false,
  diag: initialQueryDiag,
}

const IGNORED_EVENTS: ReadonlySet<string> = new Set([
  'agent_started',
  'thinking',
  'tool_dedup',
  'parse_error',
  'synthesis_started',
  'synthesis_done',
])
const HANDLED_EVENTS: ReadonlySet<string> = new Set([
  'classified',
  'status',
  'tool_call',
  'tool_result',
  'fast_path',
  'agent_done',
  'entries',
  'synthesis_result',
  'error',
  'stream_end',
])

// Same pattern as kb_core graph agent _ENTRY_ID_RE.
const ENTRY_ID_RE = /kb-\d{5}/g

const RANK: Record<NodeState, number> = { visited: 1, result: 2 }

function isNonEmptyString(v: unknown): v is string {
  return typeof v === 'string' && v !== ''
}

function parseIds(v: unknown): { ids: string[]; malformed: boolean } {
  if (!Array.isArray(v)) return { ids: [], malformed: true }
  const ids: string[] = []
  const seen = new Set<string>()
  let malformed = false
  for (const item of v) {
    if (!isNonEmptyString(item)) {
      malformed = true
      continue
    }
    if (seen.has(item)) continue
    seen.add(item)
    ids.push(item)
  }
  return { ids, malformed }
}

function uniq(ids: readonly string[]): string[] {
  return Array.from(new Set(ids))
}

/** Mutable working copy of the parts of a state the reducer changes. */
interface Draft {
  pending: RevealStep[]
  nodeStates: ReadonlyMap<string, NodeState>
  diag: {
    eventCounts: Record<string, number>
    unknownEvents: string[]
    malformed: Record<string, number>
    marksVisited: number
    marksResult: number
    marksDeduped: number
    turnsUsed: number | null
    topScore: number | null
  }
}

function makeDraft(state: QueryState): Draft {
  return {
    pending: [...state.pending],
    nodeStates: state.nodeStates,
    diag: {
      ...state.diag,
      eventCounts: { ...state.diag.eventCounts },
      unknownEvents: [...state.diag.unknownEvents],
      malformed: { ...state.diag.malformed },
    },
  }
}

function bumpMalformed(d: Draft, key: string): void {
  d.diag.malformed[key] = (d.diag.malformed[key] ?? 0) + 1
}

function enqueueMark(d: Draft, id: string, s: NodeState): void {
  let held = d.nodeStates.get(id)
  for (const step of d.pending) {
    if (step.kind === 'mark' && step.nodeId === id) {
      if (held === undefined || RANK[step.state] > RANK[held]) held = step.state
    }
  }
  if (held !== undefined && RANK[held] >= RANK[s]) {
    d.diag.marksDeduped += 1
    return
  }
  d.pending.push({ kind: 'mark', nodeId: id, state: s })
  if (s === 'visited') d.diag.marksVisited += 1
  else d.diag.marksResult += 1
}

function reduceEvent(state: QueryState, event: string, data: Record<string, unknown>): QueryState {
  const d = makeDraft(state)
  d.diag.eventCounts[event] = (d.diag.eventCounts[event] ?? 0) + 1
  if (!HANDLED_EVENTS.has(event) && !IGNORED_EVENTS.has(event)) {
    if (!d.diag.unknownEvents.includes(event)) d.diag.unknownEvents.push(event)
  }

  let queryMode = state.queryMode
  let statusMessage = state.statusMessage
  let phase = state.phase
  let answer = state.answer
  let sourceIds = state.sourceIds
  let entries = state.entries
  let error = state.error
  let streamEnded = state.streamEnded

  switch (event) {
    case 'classified': {
      queryMode = data.mode === 'summarize' ? 'summarize' : 'explore'
      statusMessage = queryMode === 'summarize' ? STATUS_SUMMARIZING : STATUS_EXPLORING
      if (data.mode !== 'explore' && data.mode !== 'summarize') bumpMalformed(d, 'classified:mode')
      break
    }
    case 'status': {
      if (typeof data.message === 'string') statusMessage = data.message
      else bumpMalformed(d, 'status:message')
      break
    }
    case 'tool_call': {
      const tool = data.tool
      if (tool === 'graph_neighbors' || tool === 'decision_chain') {
        const args =
          typeof data.args === 'object' && data.args !== null
            ? (data.args as Record<string, unknown>)
            : null
        const target =
          args === null ? undefined : args[tool === 'graph_neighbors' ? 'node_id' : 'entry_id']
        if (isNonEmptyString(target)) enqueueMark(d, target, 'visited')
        else bumpMalformed(d, 'tool_call:no_target')
      }
      break
    }
    case 'tool_result': {
      const p = parseIds(data.entry_ids)
      for (const id of p.ids) enqueueMark(d, id, 'visited')
      if (p.malformed) bumpMalformed(d, 'tool_result:entry_ids')
      break
    }
    case 'fast_path': {
      const p = parseIds(data.entry_ids)
      for (const id of p.ids) enqueueMark(d, id, 'result')
      if (p.malformed) bumpMalformed(d, 'fast_path:entry_ids')
      if (typeof data.top_score === 'number') d.diag.topScore = data.top_score
      break
    }
    case 'agent_done': {
      const p = parseIds(data.entry_ids)
      for (const id of p.ids) enqueueMark(d, id, 'result')
      if (p.malformed) bumpMalformed(d, 'agent_done:entry_ids')
      if (typeof data.turns_used === 'number') d.diag.turnsUsed = data.turns_used
      break
    }
    case 'entries': {
      if (!Array.isArray(data.entries)) {
        entries = []
        bumpMalformed(d, 'entries:not_array')
      } else {
        const kept: AskEntry[] = []
        const seen = new Set<string>()
        let bad = 0
        for (const item of data.entries as unknown[]) {
          if (
            typeof item !== 'object' ||
            item === null ||
            !isNonEmptyString((item as AskEntry).id)
          ) {
            bad += 1
            continue
          }
          const entry = item as AskEntry
          if (seen.has(entry.id)) continue
          seen.add(entry.id)
          kept.push(entry)
        }
        if (bad > 0)
          d.diag.malformed['entries:bad_item'] = (d.diag.malformed['entries:bad_item'] ?? 0) + bad
        entries = kept
      }
      if (typeof data.turns_used === 'number') d.diag.turnsUsed = data.turns_used
      for (const e of entries) enqueueMark(d, e.id, 'result')
      d.pending.push({ kind: 'fitResults' })
      break
    }
    case 'synthesis_result': {
      if (typeof data.answer === 'string') answer = data.answer
      else {
        answer = ''
        bumpMalformed(d, 'synthesis_result:answer')
      }
      const p = parseIds(data.entry_ids)
      if (p.malformed) bumpMalformed(d, 'synthesis_result:entry_ids')
      const citeIds = answer.match(ENTRY_ID_RE) ?? []
      sourceIds = uniq([...p.ids, ...citeIds])
      for (const id of sourceIds) enqueueMark(d, id, 'result')
      d.pending.push({ kind: 'fitResults' })
      break
    }
    case 'error': {
      phase = 'error'
      if (isNonEmptyString(data.message)) error = data.message
      else {
        error = 'Query failed'
        bumpMalformed(d, 'error:message')
      }
      statusMessage = ''
      break
    }
    case 'stream_end': {
      if (!streamEnded) {
        streamEnded = true
        d.pending.push({ kind: 'end' })
      }
      break
    }
    default:
      break
  }

  return {
    ...state,
    phase,
    queryMode,
    statusMessage,
    answer,
    sourceIds,
    entries,
    error,
    streamEnded,
    // Steps are only ever appended, so an equal length means nothing was queued.
    pending: d.pending.length === state.pending.length ? state.pending : d.pending,
    diag: d.diag,
  }
}

export function queryReducer(state: QueryState, action: QueryAction): QueryState {
  switch (action.type) {
    case 'start':
      return {
        ...initialQueryState,
        phase: 'streaming',
        question: action.question,
        queryMode: 'pending',
        statusMessage: STATUS_CLASSIFYING,
      }
    case 'event':
      if (state.phase === 'idle') return state
      return reduceEvent(state, action.event, action.data)
    case 'tick': {
      if (state.pending.length === 0) return state
      const [step, ...rest] = state.pending
      const lastApplied = { seq: (state.lastApplied?.seq ?? 0) + 1, step }
      const next: QueryState = { ...state, pending: rest, lastApplied }
      if (step.kind === 'mark') {
        const nodeStates = new Map(state.nodeStates)
        const pulseStarts = new Map(state.pulseStarts)
        nodeStates.set(
          step.nodeId,
          step.state === 'result' || state.nodeStates.get(step.nodeId) === 'result'
            ? 'result'
            : 'visited',
        )
        pulseStarts.set(step.nodeId, action.now)
        return { ...next, nodeStates, pulseStarts }
      }
      if (step.kind === 'end') {
        return {
          ...next,
          statusMessage: '',
          phase: state.phase === 'error' ? 'error' : 'done',
        }
      }
      return next
    }
    case 'reveal': {
      if (state.phase === 'idle') return state
      const ids = parseIds(action.ids).ids
      if (ids.length === 0) return state
      const d = makeDraft(state)
      for (const id of ids) enqueueMark(d, id, 'result')
      d.pending.push({ kind: 'fitResults' })
      return { ...state, pending: d.pending, diag: d.diag }
    }
    case 'streamClosed':
      if (state.phase === 'idle' || state.streamEnded) return state
      return {
        ...state,
        streamEnded: true,
        pending: [...state.pending, { kind: 'end' }],
      }
    case 'fail':
      if (state.phase === 'idle' || state.streamEnded) return state
      return {
        ...state,
        phase: 'error',
        error: action.message,
        statusMessage: '',
        streamEnded: true,
      }
    case 'close':
      return initialQueryState
  }
}

export function stepEffect(
  step: RevealStep,
  nodeStates: ReadonlyMap<string, NodeState>,
): {
  particlesFor: ReadonlySet<string> | null
  fit: { ids: ReadonlySet<string>; durationMs: number; padding: number } | null
} {
  if (step.kind === 'mark') {
    return {
      particlesFor: new Set([step.nodeId]),
      fit: {
        ids: new Set(nodeStates.keys()),
        durationMs: VISITED_FIT_MS,
        padding: VISITED_FIT_PADDING,
      },
    }
  }
  if (step.kind === 'fitResults') {
    const ids = new Set<string>()
    for (const [id, s] of nodeStates) if (s === 'result') ids.add(id)
    return {
      particlesFor: null,
      fit: ids.size > 0 ? { ids, durationMs: RESULTS_FIT_MS, padding: RESULTS_FIT_PADDING } : null,
    }
  }
  return { particlesFor: null, fit: null }
}

export function toTraversal(
  queryMode: QueryMode,
  nodeStates: ReadonlyMap<string, NodeState>,
  pulseStarts: ReadonlyMap<string, number>,
): TraversalState {
  if (queryMode === null && nodeStates.size === 0) return EMPTY_TRAVERSAL
  return { queryMode, nodeStates, pulseStarts }
}

/**
 * What a follow-up chat is seeded with. `answer` is always non-empty and is
 * what consumers send as seed_answer, because /api/chat/stream seeds a session
 * only when seed_question and seed_answer are both non-null.
 */
export type QuerySeed = {
  question: string
  mode: 'explore' | 'summarize'
  answer: string
  entryIds: string[]
}

export type QuerySeedInput = Pick<QueryState, 'question' | 'answer' | 'sourceIds' | 'entries'>

export function exploreSeedAnswer(entryIds: readonly string[]): string {
  const n = entryIds.length
  return `Found ${n} ${n === 1 ? 'entry' : 'entries'}: ${entryIds.map((id) => `[${id}]`).join(', ')}. Ask me anything about these results.`
}

export function querySeed(state: QuerySeedInput): QuerySeed | null {
  if (typeof state.answer === 'string' && state.answer !== '') {
    return {
      question: state.question,
      mode: 'summarize',
      answer: state.answer,
      entryIds: state.sourceIds,
    }
  }
  const entries = state.entries ?? []
  if (entries.length >= 1) {
    const ids = entries.map((e) => e.id)
    return {
      question: state.question,
      mode: 'explore',
      answer: exploreSeedAnswer(ids),
      entryIds: ids,
    }
  }
  return null
}

export interface QuerySummary {
  outcome: 'done' | 'error' | 'aborted'
  abort_reason: 'close' | 'superseded' | 'unmount' | null
  error_source: 'server' | 'transport' | null
  error: string | null
  mode: QueryMode
  path: 'fast_path' | 'agent' | 'no_agent'
  turns_used: number | null
  top_score: number | null
  events: number
  event_counts: Record<string, number>
  unknown_events: string[]
  malformed: Record<string, number>
  marks_visited: number
  marks_result: number
  marks_deduped: number
  marked_nodes: number
  marked_rendered: number
  marked_hidden: number
  marked_not_in_graph: number
  not_in_graph_sample: string[]
  results: number
  results_hidden: number
  results_not_in_graph: number
}

export function summarizeQuery(
  state: QueryState,
  ctx: {
    outcome: QuerySummary['outcome']
    abortReason: QuerySummary['abort_reason']
    renderedIds: ReadonlySet<string>
    graphIds: ReadonlySet<string>
  },
): QuerySummary {
  const counts = state.diag.eventCounts
  const marked = uniq([
    ...state.nodeStates.keys(),
    ...state.pending.flatMap((s) => (s.kind === 'mark' ? [s.nodeId] : [])),
  ])
  const split = (ids: readonly string[]) => {
    const rendered = ids.filter((id) => ctx.renderedIds.has(id))
    const hidden = ids.filter((id) => ctx.graphIds.has(id) && !ctx.renderedIds.has(id))
    const missing = ids.filter((id) => !ctx.graphIds.has(id))
    return { rendered, hidden, missing }
  }
  const m = split(marked)
  const resultIds = state.answer !== null ? state.sourceIds : (state.entries ?? []).map((e) => e.id)
  const r = split(resultIds)
  return {
    outcome: ctx.outcome,
    abort_reason: ctx.abortReason,
    error_source: (counts.error ?? 0) > 0 ? 'server' : state.phase === 'error' ? 'transport' : null,
    error: state.error,
    mode: state.queryMode,
    path:
      (counts.fast_path ?? 0) > 0
        ? 'fast_path'
        : (counts.agent_started ?? 0) > 0
          ? 'agent'
          : 'no_agent',
    turns_used: state.diag.turnsUsed,
    top_score: state.diag.topScore,
    events: Object.values(counts).reduce((a, b) => a + b, 0),
    event_counts: { ...counts },
    unknown_events: [...state.diag.unknownEvents],
    malformed: { ...state.diag.malformed },
    marks_visited: state.diag.marksVisited,
    marks_result: state.diag.marksResult,
    marks_deduped: state.diag.marksDeduped,
    marked_nodes: marked.length,
    marked_rendered: m.rendered.length,
    marked_hidden: m.hidden.length,
    marked_not_in_graph: m.missing.length,
    not_in_graph_sample: m.missing.slice(0, NOT_IN_GRAPH_SAMPLE),
    results: resultIds.length,
    results_hidden: r.hidden.length,
    results_not_in_graph: r.missing.length,
  }
}
