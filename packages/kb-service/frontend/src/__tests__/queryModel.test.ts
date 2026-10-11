import { describe, it, expect } from 'vitest'
import { EMPTY_TRAVERSAL } from '../graph/explorerModel'
import {
  exploreSeedAnswer,
  initialQueryState,
  queryReducer,
  querySeed,
  stepEffect,
  summarizeQuery,
  toTraversal,
} from '../graph/queryModel'
import type { QueryAction, QueryState } from '../graph/queryModel'
import {
  DRIFT_EVENTS,
  ERROR_EVENTS,
  EXPLORE_AGENT_EVENTS,
  EXPLORE_EMPTY_EVENTS,
  EXPLORE_NO_LLM_EVENTS,
  SUMMARIZE_FAST_PATH_EVENTS,
  SUMMARIZE_PLANNER_EVENTS,
} from './fixtures/queryStreamEvents'
import type { RecordedEvent } from './fixtures/queryStreamEvents'

const withoutDiag = (s: QueryState) => ({ ...s, diag: null })
const R = new Set(['kb-00001', 'kb-00002', 'tag:alpha', 'tag:beta'])
const ctx = {
  outcome: 'done' as const,
  abortReason: null,
  renderedIds: R,
  graphIds: R,
}

function run(state: QueryState, actions: QueryAction[]): QueryState {
  return actions.reduce(queryReducer, state)
}
const ev = (e: RecordedEvent): QueryAction => ({ type: 'event', event: e.event, data: e.data })
const started = (q = 'q') => queryReducer(initialQueryState, { type: 'start', question: q })
function feed(events: RecordedEvent[], q = 'q'): QueryState {
  return run(started(q), events.map(ev))
}
function drain(state: QueryState): QueryState {
  let s = state
  let now = 1000
  while (s.pending.length > 0) {
    s = queryReducer(s, { type: 'tick', now })
    now += 250
  }
  return s
}
const mark = (nodeId: string, state: 'visited' | 'result') => ({ kind: 'mark', nodeId, state })
const evt = (event: string, data: Record<string, unknown>): QueryAction => ({
  type: 'event',
  event,
  data,
})

describe('queryReducer start and classification', () => {
  it('start enters streaming and classifying', () => {
    const s = started('why alpha?')
    expect(s.phase).toBe('streaming')
    expect(s.queryMode).toBe('pending')
    expect(s.statusMessage).toBe('Classifying query...')
    expect(s.question).toBe('why alpha?')
  })

  it('classified sets the mode and status', () => {
    const sum = queryReducer(started(), evt('classified', { mode: 'summarize' }))
    expect(sum.queryMode).toBe('summarize')
    expect(sum.statusMessage).toBe('Preparing answer...')
    const exp = queryReducer(started(), evt('classified', { mode: 'explore' }))
    expect(exp.queryMode).toBe('explore')
    expect(exp.statusMessage).toBe('Exploring knowledge graph...')
  })

  it('an unknown or missing mode falls back to explore and is counted', () => {
    for (const data of [{ mode: 'weird' }, {}]) {
      const s = queryReducer(started(), evt('classified', data))
      expect(s.queryMode).toBe('explore')
      expect(s.diag.malformed).toEqual({ 'classified:mode': 1 })
    }
  })
})

describe('recorded streams', () => {
  it('EXPLORE_AGENT queues marks in order', () => {
    const s = feed(EXPLORE_AGENT_EVENTS, 'why alpha?')
    expect(s.pending).toEqual([
      mark('kb-00001', 'visited'),
      mark('kb-00002', 'visited'),
      mark('tag:alpha', 'visited'),
      mark('kb-00003', 'visited'),
      mark('kb-00001', 'result'),
      mark('kb-00003', 'result'),
      { kind: 'fitResults' },
      { kind: 'end' },
    ])
    expect(s.statusMessage).toBe('Thinking (turn 3)...')
    expect(s.diag.marksVisited).toBe(4)
    expect(s.diag.marksResult).toBe(2)
    expect(s.diag.marksDeduped).toBe(3)
  })

  it('EXPLORE_AGENT drains to the final node states', () => {
    let s = feed(EXPLORE_AGENT_EVENTS)
    let now = 1000
    for (let i = 0; i < 8; i++) {
      s = queryReducer(s, { type: 'tick', now })
      now += 250
    }
    expect(Object.fromEntries(s.nodeStates)).toEqual({
      'kb-00001': 'result',
      'kb-00002': 'visited',
      'tag:alpha': 'visited',
      'kb-00003': 'result',
    })
    expect(Object.fromEntries(s.pulseStarts)).toEqual({
      'kb-00001': 2000,
      'kb-00002': 1250,
      'tag:alpha': 1500,
      'kb-00003': 2250,
    })
    expect(s.phase).toBe('done')
    expect(s.statusMessage).toBe('')
    expect(s.entries).toHaveLength(2)
    expect(s.lastApplied?.seq).toBe(8)
  })

  it('SUMMARIZE_FAST_PATH', () => {
    const s = feed(SUMMARIZE_FAST_PATH_EVENTS)
    expect(s.pending).toEqual([
      mark('kb-00002', 'result'),
      mark('kb-00001', 'result'),
      { kind: 'fitResults' },
      { kind: 'end' },
    ])
    expect(s.answer?.startsWith('We chose it')).toBe(true)
    expect(s.sourceIds).toEqual(['kb-00002', 'kb-00001'])
  })

  it('SUMMARIZE_PLANNER takes the cited id', () => {
    const s = feed(SUMMARIZE_PLANNER_EVENTS)
    expect(s.pending).toEqual([mark('kb-00001', 'result'), { kind: 'fitResults' }, { kind: 'end' }])
    expect(s.sourceIds).toEqual(['kb-00001'])
  })

  it('EXPLORE_NO_LLM and EXPLORE_EMPTY', () => {
    expect(feed(EXPLORE_NO_LLM_EVENTS).pending).toEqual([
      mark('kb-00001', 'result'),
      { kind: 'fitResults' },
      { kind: 'end' },
    ])
    const empty = feed(EXPLORE_EMPTY_EVENTS)
    expect(empty.pending).toEqual([{ kind: 'fitResults' }, { kind: 'end' }])
    const done = drain(empty)
    expect(done.entries).toEqual([])
    expect(done.phase).toBe('done')
  })

  it('ERROR drains to the error phase', () => {
    const s = drain(feed(ERROR_EVENTS))
    expect(s.phase).toBe('error')
    expect(s.error).toBe('RuntimeError: boom')
    expect(s.statusMessage).toBe('')
  })
})

describe('dedup and malformed input', () => {
  const mid = () => queryReducer(started(), evt('classified', { mode: 'explore' }))

  it('does not queue a visited mark for a result node', () => {
    const s = drain(run(mid(), [evt('agent_done', { entry_ids: ['kb-00001'] })]))
    const next = queryReducer(s, evt('tool_result', { entry_ids: ['kb-00001'] }))
    expect(next.pending).toEqual([])
    expect(next.diag.marksDeduped).toBe(1)
  })

  it('queues a result mark for a visited node', () => {
    const s = drain(run(mid(), [evt('tool_result', { entry_ids: ['kb-00001'] })]))
    const next = queryReducer(s, evt('agent_done', { entry_ids: ['kb-00001'] }))
    expect(next.pending).toEqual([mark('kb-00001', 'result')])
  })

  it('agent_done duplicates queue one mark', () => {
    const s = queryReducer(mid(), evt('agent_done', { entry_ids: ['kb-00001', 'kb-00001'] }))
    expect(s.pending).toEqual([mark('kb-00001', 'result')])
  })

  it('tool_result skips non-string ids and counts it', () => {
    const s = queryReducer(mid(), evt('tool_result', { entry_ids: [1, null, '', 'kb-00009'] }))
    expect(s.pending).toEqual([mark('kb-00009', 'visited')])
    expect(s.diag.malformed).toEqual({ 'tool_result:entry_ids': 1 })
  })

  it('synthesis_result dedups entry ids', () => {
    const s = queryReducer(
      mid(),
      evt('synthesis_result', { answer: 'x', entry_ids: ['kb-00002', 'kb-00002'] }),
    )
    expect(s.sourceIds).toEqual(['kb-00002'])
  })

  it('entries keeps the first of a duplicate id', () => {
    const e = { id: 'kb-00001', short_title: 't', entry_type: 'decision', tags: [], context: '' }
    const s = queryReducer(mid(), evt('entries', { entries: [e, e] }))
    expect(s.entries).toHaveLength(1)
  })

  it('entries with a non-array value', () => {
    const s = queryReducer(mid(), evt('entries', { entries: 'nope' }))
    expect(s.entries).toEqual([])
    expect(s.pending).toEqual([{ kind: 'fitResults' }])
    expect(s.diag.malformed).toEqual({ 'entries:not_array': 1 })
  })

  it('decision_chain entry_id is a visited mark', () => {
    const s = queryReducer(
      mid(),
      evt('tool_call', { tool: 'decision_chain', args: { entry_id: 'kb-00005' } }),
    )
    expect(s.pending).toEqual([mark('kb-00005', 'visited')])
  })

  it('graph_neighbors without args is malformed', () => {
    const before = mid()
    const s = queryReducer(before, evt('tool_call', { tool: 'graph_neighbors' }))
    expect(s.diag.malformed).toEqual({ 'tool_call:no_target': 1 })
    expect(s.pending).toBe(before.pending)
  })
})

describe('ignored and unknown events', () => {
  it('hybrid_search tool_call only counts', () => {
    const prev = started()
    const next = queryReducer(prev, evt('tool_call', { tool: 'hybrid_search', args: {} }))
    expect(withoutDiag(next)).toEqual({ ...withoutDiag(prev), pending: prev.pending })
    expect(next.pending).toBe(prev.pending)
    expect(next.diag.eventCounts.tool_call).toBe(1)
  })

  it.each([
    'agent_started',
    'thinking',
    'tool_dedup',
    'parse_error',
    'synthesis_started',
    'synthesis_done',
  ])('%s has no effect', (name) => {
    const prev = started()
    const next = queryReducer(prev, evt(name, {}))
    expect(withoutDiag(next)).toEqual(withoutDiag(prev))
    expect(next.pending).toBe(prev.pending)
    expect(next.diag.unknownEvents).toEqual([])
  })

  it('an unknown event is recorded once', () => {
    const s = run(started(), [evt('tool_progress', {}), evt('tool_progress', {})])
    expect(s.diag.unknownEvents).toEqual(['tool_progress'])
    expect(s.diag.eventCounts.tool_progress).toBe(2)
  })
})

describe('same-object no-ops and other actions', () => {
  it('returns the same state', () => {
    expect(queryReducer(initialQueryState, { type: 'tick', now: 1 })).toBe(initialQueryState)
    expect(queryReducer(initialQueryState, evt('classified', { mode: 'explore' }))).toBe(
      initialQueryState,
    )
    expect(queryReducer(initialQueryState, { type: 'reveal', ids: ['kb-00001'] })).toBe(
      initialQueryState,
    )
    const s = started()
    expect(queryReducer(s, { type: 'reveal', ids: [] })).toBe(s)
    const ended = feed(SUMMARIZE_PLANNER_EVENTS)
    expect(queryReducer(ended, { type: 'streamClosed' })).toBe(ended)
    expect(queryReducer(ended, { type: 'fail', message: 'x' })).toBe(ended)
  })

  it('streamClosed without stream_end queues end', () => {
    const s = queryReducer(started(), { type: 'streamClosed' })
    expect(s.pending).toEqual([{ kind: 'end' }])
    expect(s.streamEnded).toBe(true)
  })

  it('fail mid-stream errors out', () => {
    const s = queryReducer(started(), { type: 'fail', message: 'x' })
    expect(s.phase).toBe('error')
    expect(s.error).toBe('x')
    expect(s.streamEnded).toBe(true)
  })

  it('close returns the initial state', () => {
    expect(queryReducer(started(), { type: 'close' })).toBe(initialQueryState)
  })

  it('tick does not mutate the previous maps', () => {
    const s = feed(EXPLORE_AGENT_EVENTS)
    const before = queryReducer(s, { type: 'tick', now: 1 })
    const size = before.nodeStates.size
    queryReducer(before, { type: 'tick', now: 2 })
    expect(before.nodeStates.size).toBe(size)
    expect(before.nodeStates.get('kb-00002')).toBeUndefined()
    expect(before.pulseStarts.size).toBe(size)
  })

  it('reveal after draining queues only new result marks', () => {
    const done = drain(feed(EXPLORE_AGENT_EVENTS))
    const s = queryReducer(done, { type: 'reveal', ids: ['kb-00002', 'kb-00001'] })
    expect(s.pending).toEqual([mark('kb-00002', 'result'), { kind: 'fitResults' }])
    expect(s.phase).toBe('done')
  })
})

describe('stepEffect and toTraversal', () => {
  const states = new Map<string, 'visited' | 'result'>([
    ['a', 'visited'],
    ['b', 'result'],
  ])

  it('mark', () => {
    const fx = stepEffect({ kind: 'mark', nodeId: 'a', state: 'visited' }, states)
    expect(fx.particlesFor).toEqual(new Set(['a']))
    expect(fx.fit).toEqual({ ids: new Set(['a', 'b']), durationMs: 500, padding: 60 })
  })

  it('fitResults', () => {
    expect(stepEffect({ kind: 'fitResults' }, states).fit).toEqual({
      ids: new Set(['b']),
      durationMs: 600,
      padding: 40,
    })
    expect(stepEffect({ kind: 'fitResults' }, new Map([['a', 'visited']])).fit).toBeNull()
  })

  it('end', () => {
    expect(stepEffect({ kind: 'end' }, states)).toEqual({ particlesFor: null, fit: null })
  })

  it('toTraversal', () => {
    expect(toTraversal(null, new Map(), new Map())).toBe(EMPTY_TRAVERSAL)
    const t = toTraversal('pending', new Map(), new Map())
    expect(t).toEqual({ queryMode: 'pending', nodeStates: new Map(), pulseStarts: new Map() })
  })
})

describe('querySeed', () => {
  it('is null while idle and for an empty result', () => {
    expect(querySeed(initialQueryState)).toBeNull()
    expect(querySeed(drain(feed(EXPLORE_EMPTY_EVENTS)))).toBeNull()
  })

  it('seeds a summarize answer', () => {
    expect(querySeed(drain(feed(SUMMARIZE_FAST_PATH_EVENTS, 'why beta?')))).toEqual({
      question: 'why beta?',
      mode: 'summarize',
      answer: 'We chose it for async support [kb-00002], see also [kb-00001].',
      entryIds: ['kb-00002', 'kb-00001'],
    })
  })

  it('seeds explore entries', () => {
    expect(querySeed(drain(feed(EXPLORE_AGENT_EVENTS, 'why alpha?')))).toEqual({
      question: 'why alpha?',
      mode: 'explore',
      answer: 'Found 2 entries: [kb-00001], [kb-00003]. Ask me anything about these results.',
      entryIds: ['kb-00001', 'kb-00003'],
    })
  })

  it('seeds a single no-LLM explore entry', () => {
    expect(querySeed(drain(feed(EXPLORE_NO_LLM_EVENTS, 'q')))).toMatchObject({
      answer: 'Found 1 entry: [kb-00001]. Ask me anything about these results.',
      entryIds: ['kb-00001'],
    })
  })

  it('exploreSeedAnswer pluralises', () => {
    expect(exploreSeedAnswer(['kb-00001'])).toBe(
      'Found 1 entry: [kb-00001]. Ask me anything about these results.',
    )
    expect(exploreSeedAnswer(['kb-00001', 'kb-00002'])).toBe(
      'Found 2 entries: [kb-00001], [kb-00002]. Ask me anything about these results.',
    )
  })

  it('accepts a plain seed input', () => {
    expect(
      querySeed({ question: 'Q', answer: 'A', sourceIds: ['kb-00001'], entries: null }),
    ).toEqual({ question: 'Q', mode: 'summarize', answer: 'A', entryIds: ['kb-00001'] })
  })
})

describe('summarizeQuery', () => {
  it('EXPLORE_AGENT', () => {
    const s = summarizeQuery(drain(feed(EXPLORE_AGENT_EVENTS)), ctx)
    expect(s).toMatchObject({
      events: 18,
      path: 'agent',
      turns_used: 3,
      marks_visited: 4,
      marks_result: 2,
      marks_deduped: 3,
      marked_nodes: 4,
      marked_rendered: 3,
      marked_hidden: 0,
      marked_not_in_graph: 1,
      not_in_graph_sample: ['kb-00003'],
      results: 2,
      results_not_in_graph: 1,
      unknown_events: [],
      malformed: {},
      error_source: null,
    })
    expect(s.event_counts.status).toBe(6)
  })

  it('SUMMARIZE_FAST_PATH', () => {
    const state = drain(feed(SUMMARIZE_FAST_PATH_EVENTS))
    expect(summarizeQuery(state, ctx)).toMatchObject({
      path: 'fast_path',
      top_score: 0.91,
      marks_result: 2,
      marks_deduped: 2,
      turns_used: null,
    })
    expect(summarizeQuery(state, { ...ctx, renderedIds: new Set(['kb-00001']) })).toMatchObject({
      marked_hidden: 1,
      results_hidden: 1,
    })
  })

  it('SUMMARIZE_PLANNER has no agent path', () => {
    expect(summarizeQuery(drain(feed(SUMMARIZE_PLANNER_EVENTS)), ctx).path).toBe('no_agent')
  })

  it('error sources', () => {
    expect(
      summarizeQuery(drain(feed(ERROR_EVENTS)), { ...ctx, outcome: 'error' }).error_source,
    ).toBe('server')
    const failed = queryReducer(started(), { type: 'fail', message: 'x' })
    expect(summarizeQuery(failed, { ...ctx, outcome: 'error' }).error_source).toBe('transport')
  })

  it('DRIFT', () => {
    expect(summarizeQuery(drain(feed(DRIFT_EVENTS)), ctx)).toMatchObject({
      unknown_events: ['tool_progress'],
      malformed: { 'tool_result:entry_ids': 1 },
      marked_nodes: 1,
      marked_rendered: 0,
      marked_not_in_graph: 1,
      not_in_graph_sample: ['alpha'],
      turns_used: 1,
    })
  })
})
