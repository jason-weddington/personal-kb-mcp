// Recorded /api/kb/query/stream frames, mirroring query_routes.py frame for
// frame. Not a test file: vitest only picks up *.test.* and *.spec.* names.
export type RecordedEvent = { event: string; data: Record<string, unknown> }

export const EXPLORE_AGENT_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'explore' } },
  { event: 'agent_started', data: { type: 'agent_started', seed_results: 2, max_turns: 4 } },
  { event: 'status', data: { message: 'Searching knowledge base...' } },
  { event: 'thinking', data: { type: 'thinking', turn: 1 } },
  { event: 'status', data: { message: 'Thinking (turn 1)...' } },
  {
    event: 'tool_call',
    data: { type: 'tool_call', turn: 1, tool: 'hybrid_search', args: { query: 'alpha' } },
  },
  { event: 'status', data: { message: 'Searching: alpha...' } },
  {
    event: 'tool_result',
    data: {
      type: 'tool_result',
      turn: 1,
      tool: 'hybrid_search',
      entry_ids: ['kb-00001', 'kb-00002'],
    },
  },
  { event: 'thinking', data: { type: 'thinking', turn: 2 } },
  { event: 'status', data: { message: 'Thinking (turn 2)...' } },
  {
    event: 'tool_call',
    data: { type: 'tool_call', turn: 2, tool: 'graph_neighbors', args: { node_id: 'tag:alpha' } },
  },
  { event: 'status', data: { message: 'Exploring neighbors of tag:alpha...' } },
  {
    event: 'tool_result',
    data: {
      type: 'tool_result',
      turn: 2,
      tool: 'graph_neighbors',
      entry_ids: ['kb-00001', 'kb-00003'],
    },
  },
  { event: 'thinking', data: { type: 'thinking', turn: 3 } },
  { event: 'status', data: { message: 'Thinking (turn 3)...' } },
  {
    event: 'agent_done',
    data: {
      type: 'agent_done',
      entry_ids: ['kb-00001', 'kb-00003'],
      reasoning: 'both cover alpha',
      turns_used: 3,
    },
  },
  {
    event: 'entries',
    data: {
      entries: [
        {
          id: 'kb-00001',
          short_title: 'Alpha decision',
          entry_type: 'decision',
          tags: ['alpha'],
          context: 'agent selected',
        },
        {
          id: 'kb-00003',
          short_title: 'Gamma pattern',
          entry_type: 'pattern_convention',
          tags: [],
          context: 'agent selected',
        },
      ],
      turns_used: 3,
    },
  },
  { event: 'stream_end', data: {} },
]

export const SUMMARIZE_FAST_PATH_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'summarize' } },
  {
    event: 'fast_path',
    data: { type: 'fast_path', entry_ids: ['kb-00002', 'kb-00001'], top_score: 0.91 },
  },
  { event: 'status', data: { message: 'Found strong matches...' } },
  { event: 'synthesis_started', data: { type: 'synthesis_started', entry_count: 2 } },
  { event: 'status', data: { message: 'Synthesizing answer from 2 entries...' } },
  { event: 'synthesis_done', data: { type: 'synthesis_done' } },
  {
    event: 'synthesis_result',
    data: {
      answer: 'We chose it for async support [kb-00002], see also [kb-00001].',
      question: 'why beta?',
      entry_ids: ['kb-00002', 'kb-00001'],
    },
  },
  { event: 'stream_end', data: {} },
]

// The KB_AGENTIC_QUERY=FALSE path: no engine retrieval events, empty entry_ids.
export const SUMMARIZE_PLANNER_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'summarize' } },
  { event: 'synthesis_started', data: { type: 'synthesis_started', entry_count: 1 } },
  { event: 'status', data: { message: 'Synthesizing answer from 1 entries...' } },
  { event: 'synthesis_done', data: { type: 'synthesis_done' } },
  { event: 'synthesis_result', data: { answer: 'See [kb-00001].', question: 'q', entry_ids: [] } },
  { event: 'stream_end', data: {} },
]

export const EXPLORE_NO_LLM_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'explore' } },
  {
    event: 'entries',
    data: {
      entries: [
        {
          id: 'kb-00001',
          short_title: 'Alpha decision',
          entry_type: 'decision',
          tags: ['alpha'],
          context: 'search match',
        },
      ],
      turns_used: 0,
    },
  },
  { event: 'stream_end', data: {} },
]

export const EXPLORE_EMPTY_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'explore' } },
  { event: 'entries', data: { entries: [], turns_used: 0 } },
  { event: 'stream_end', data: {} },
]

export const ERROR_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'explore' } },
  { event: 'error', data: { message: 'RuntimeError: boom' } },
  { event: 'stream_end', data: {} },
]

export const DRIFT_EVENTS: RecordedEvent[] = [
  { event: 'classified', data: { mode: 'explore' } },
  { event: 'tool_progress', data: { type: 'tool_progress' } },
  {
    event: 'tool_result',
    data: { type: 'tool_result', tool: 'hybrid_search', entry_ids: 'kb-00001' },
  },
  {
    event: 'tool_call',
    data: { type: 'tool_call', tool: 'graph_neighbors', args: { node_id: 'alpha' } },
  },
  { event: 'entries', data: { entries: [], turns_used: 1 } },
  { event: 'stream_end', data: {} },
]
