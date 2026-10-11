import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, useImperativeHandle } from 'react'
import type { Ref } from 'react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Graph } from '../pages/Graph'
import type { ExplorerNode } from '../graph/explorerModel'
import type { GraphFullResponse } from '../kbTypes'
import {
  DRIFT_EVENTS,
  ERROR_EVENTS,
  EXPLORE_AGENT_EVENTS,
  EXPLORE_EMPTY_EVENTS,
  EXPLORE_NO_LLM_EVENTS,
  SUMMARIZE_FAST_PATH_EVENTS,
} from './fixtures/queryStreamEvents'
import type { RecordedEvent } from './fixtures/queryStreamEvents'

/* eslint-disable @typescript-eslint/no-explicit-any */

const h = vi.hoisted(() => {
  const fgState = { cx: 0, cy: 0, k: 1 }
  const fgMethods: Record<string, any> = {}
  fgMethods.centerAt = vi.fn((x?: number, y?: number) => {
    if (x === undefined) return { x: fgState.cx, y: fgState.cy }
    fgState.cx = x
    fgState.cy = y ?? 0
    return fgMethods
  })
  fgMethods.zoom = vi.fn((k?: number) => {
    if (k === undefined) return fgState.k
    fgState.k = k
    return fgMethods
  })
  fgMethods.zoomToFit = vi.fn(() => fgMethods)
  fgMethods.emitParticle = vi.fn(() => fgMethods)
  return {
    fgState,
    fgMethods,
    lastProps: {} as Record<string, any>,
    mockOpenEntry: vi.fn(),
    getGraphFull: vi.fn(),
    getToken: vi.fn((): string | null => 'tok'),
    mockStreamSSE: vi.fn<(...a: any[]) => Promise<void>>(() => new Promise<void>(() => {})),
  }
})

vi.mock('react-force-graph-2d', () => ({
  default: function ForceGraphStub({
    ref,
    ...props
  }: { ref?: Ref<unknown> } & Record<string, any>) {
    h.lastProps = props
    useImperativeHandle(ref, () => h.fgMethods)
    return <div data-testid="fg-stub" />
  },
}))

vi.mock('../api', () => ({ getGraphFull: h.getGraphFull, getToken: h.getToken }))
vi.mock('../streaming', () => ({ streamSSE: h.mockStreamSSE }))
vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: () => ({ openEntry: h.mockOpenEntry, closeEntry: vi.fn() }),
}))

class ResizeObserverStub {
  private cb: ResizeObserverCallback
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb
  }
  observe(el: Element) {
    this.cb(
      [{ target: el, contentRect: { width: 1200, height: 900 } } as unknown as ResizeObserverEntry],
      this as unknown as ResizeObserver,
    )
  }
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver

function fixture(): GraphFullResponse {
  return {
    nodes: [
      {
        id: 'kb-00001',
        type: 'entry',
        label: 'Alpha decision',
        val: 2,
        properties: { project_ref: 'p1' },
        x: 10,
        y: 20,
      },
      {
        id: 'kb-00002',
        type: 'entry',
        label: 'Beta lesson',
        val: 2,
        properties: { project_ref: 'p2' },
        x: 30,
        y: 40,
      },
      { id: 'tag:alpha', type: 'tag', label: 'alpha', val: 1, properties: {}, x: 50, y: 60 },
      { id: 'tag:beta', type: 'tag', label: 'beta', val: 1, properties: {}, x: 70, y: 80 },
    ] as ExplorerNode[],
    edges: [
      { source: 'kb-00001', target: 'tag:alpha', type: 'has_tag', properties: {} },
      { source: 'kb-00002', target: 'tag:beta', type: 'has_tag', properties: {} },
      { source: 'kb-00001', target: 'kb-00002', type: 'related_to', properties: {} },
    ],
    stats: { node_count: 4, edge_count: 3 },
  }
}

const fg = h.fgMethods
const clearFg = () => Object.values(fg).forEach((f) => f.mockClear())
const links = () => h.lastProps.graphData.links as any[]
const link = (s: string, t: string) => links().find((l) => l.source === s && l.target === t)
const advance = (ms: number) =>
  act(() => {
    vi.advanceTimersByTime(ms)
  })
const drain = () => {
  for (let i = 0; i < 12; i++) advance(250)
}
const feedCall = (events: RecordedEvent[], call = 0) =>
  act(() => {
    const onEvent = h.mockStreamSSE.mock.calls[call][2]
    for (const e of events) onEvent(e.event, e.data)
  })
const signalOf = (call = 0) => h.mockStreamSSE.mock.calls[call][3] as AbortSignal
const askBox = () => screen.getByRole('textbox', { name: 'Ask the knowledge base' })
const searchBox = () => screen.getByPlaceholderText('Search nodes or ask a question...')
const region = () => screen.queryByRole('region', { name: 'Query results' })
const cameraMoves = () =>
  fg.centerAt.mock.calls.filter((c: any[]) => c.length > 0).length +
  fg.zoom.mock.calls.filter((c: any[]) => c.length > 0).length +
  fg.zoomToFit.mock.calls.length
const lastFit = (ms: number, pad: number) =>
  [...fg.zoomToFit.mock.calls].reverse().find((c: any[]) => c[0] === ms && c[1] === pad)

// RTL's asyncWrapper only advances fake timers when a `jest` global exists, so
// user-event would wait on a fake setTimeout forever under vitest.
;(globalThis as any).jest = { advanceTimersByTime: (ms: number) => vi.advanceTimersByTime(ms) }

let user: ReturnType<typeof userEvent.setup>
let infoSpy: ReturnType<typeof vi.spyOn>
let warnSpy: ReturnType<typeof vi.spyOn>
const infoCalls = () => infoSpy.mock.calls.filter((c: any[]) => c[0] === '[graph-query]')

async function setup() {
  const view = render(<Graph />)
  await screen.findByText('4 of 4 nodes · 3 of 3 edges')
  await vi.waitFor(() => expect(fg.zoomToFit).toHaveBeenCalledWith(400, 40, expect.any(Function)), {
    timeout: 3000,
  })
  await new Promise((r) => setTimeout(r, 500))
  clearFg()
  vi.useFakeTimers()
  user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
  return view
}

async function ask(text: string) {
  await user.type(askBox(), `${text}{Enter}`)
}

beforeEach(() => {
  h.fgState.cx = 0
  h.fgState.cy = 0
  h.fgState.k = 1
  h.lastProps = {}
  h.mockOpenEntry.mockClear()
  h.mockStreamSSE.mockReset()
  h.mockStreamSSE.mockImplementation(() => new Promise<void>(() => {}))
  h.getToken.mockReset()
  h.getToken.mockReturnValue('tok')
  clearFg()
  h.getGraphFull.mockImplementation(() => Promise.resolve(fixture()))
  infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {})
  warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('Graph query mode', { timeout: 15000 }, () => {
  it('(a) asking streams to the query endpoint', async () => {
    await setup()
    await ask('why alpha?')
    expect(h.mockStreamSSE).toHaveBeenCalledTimes(1)
    expect(h.mockStreamSSE).toHaveBeenCalledWith(
      '/api/kb/query/stream?token=tok',
      { question: 'why alpha?' },
      expect.any(Function),
      expect.any(AbortSignal),
    )
    const panel = screen.getByRole('region', { name: 'Query results' })
    expect(within(panel).getByText('classifying')).toBeInTheDocument()
    expect(within(panel).getByText('Classifying query...')).toBeInTheDocument()
  })

  it('(a2) without a token the url has no query string', async () => {
    h.getToken.mockReturnValue(null)
    await setup()
    await ask('q')
    expect(h.mockStreamSSE.mock.calls[0][0]).toBe('/api/kb/query/stream')
  })

  it('(b) free text asks, a highlighted suggestion flies', async () => {
    await setup()
    await user.type(searchBox(), 'zzz{Enter}')
    expect(h.mockStreamSSE).toHaveBeenCalledTimes(1)
    expect(h.mockStreamSSE.mock.calls[0][1]).toEqual({ question: 'zzz' })
    h.mockStreamSSE.mockClear()
    await user.type(searchBox(), 'alpha{ArrowDown}{Enter}')
    expect(fg.centerAt).toHaveBeenCalledWith(50, 60, 800)
    expect(h.mockStreamSSE).not.toHaveBeenCalled()
  })

  it('(c) reveals one step per 250 ms with particles, fits and colours', async () => {
    await setup()
    await ask('why alpha?')
    feedCall(EXPLORE_AGENT_EVENTS)
    const toAlpha = link('kb-00001', 'tag:alpha')
    const toBeta = link('kb-00001', 'kb-00002')

    advance(0)
    expect(fg.emitParticle).toHaveBeenCalledTimes(2)
    expect(fg.emitParticle).toHaveBeenCalledWith(toAlpha)
    expect(fg.emitParticle).toHaveBeenCalledWith(toBeta)
    const fit0 = lastFit(500, 60)
    expect(fit0).toBeDefined()
    expect(fit0![2]({ id: 'kb-00001' })).toBe(true)
    expect(fit0![2]({ id: 'kb-00002' })).toBe(false)

    advance(250)
    expect(lastFit(500, 60)![2]({ id: 'kb-00002' })).toBe(true)

    advance(1249)
    expect(lastFit(600, 40)).toBeUndefined()
    advance(1)
    const res = lastFit(600, 40)
    expect(res).toBeDefined()
    expect(res![2]({ id: 'kb-00001' })).toBe(true)
    expect(res![2]({ id: 'kb-00003' })).toBe(true)
    expect(res![2]({ id: 'kb-00002' })).toBe(false)
    expect(res![2]({ id: 'tag:alpha' })).toBe(false)

    advance(249)
    expect(screen.getByTestId('query-status')).toBeInTheDocument()
    advance(1)
    expect(screen.queryByTestId('query-status')).toBeNull()
    expect(screen.queryByTestId('query-progress')).toBeNull()
    expect(h.lastProps.linkColor(link('kb-00001', 'tag:alpha'))).toBe('rgba(0,255,136,0.5)')
    expect(h.lastProps.linkColor(link('kb-00002', 'tag:beta'))).toBe('rgba(0,255,136,0.25)')
  })

  it('(d) explore list flies to rendered rows and opens the drawer for others', async () => {
    await setup()
    await ask('why alpha?')
    feedCall(EXPLORE_AGENT_EVENTS)
    drain()
    const panel = screen.getByRole('region', { name: 'Query results' })
    expect(within(panel).getByText('Found 2 entries')).toBeInTheDocument()
    expect(within(panel).getByText('Alpha decision')).toBeInTheDocument()
    expect(within(panel).getByText('Gamma pattern')).toBeInTheDocument()
    expect(within(panel).getByText('not in graph')).toBeInTheDocument()
    expect(within(panel).getByText('1 of 2 results not shown on the graph')).toBeInTheDocument()

    fg.centerAt.mockClear()
    fireEvent.click(within(panel).getByText('Alpha decision'))
    expect(fg.centerAt).toHaveBeenCalledWith(90, 20, 800)
    advance(849)
    expect(h.mockOpenEntry).not.toHaveBeenCalled()
    advance(1)
    expect(h.mockOpenEntry).toHaveBeenCalledWith('kb-00001')

    h.mockOpenEntry.mockClear()
    fg.centerAt.mockClear()
    warnSpy.mockClear()
    fireEvent.click(within(panel).getByText('Gamma pattern'))
    expect(h.mockOpenEntry).toHaveBeenCalledWith('kb-00003')
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(
      warnSpy.mock.calls.filter((c: any[]) => c[0] === '[graph] flyToNode: node not rendered'),
    ).toHaveLength(1)
  })

  it('(e) summarize shows the answer and labels hidden sources', async () => {
    await setup()
    fireEvent.mouseDown(screen.getByRole('combobox', { name: /Scope/ }))
    fireEvent.click(screen.getByRole('option', { name: 'p1' }))
    await ask('why beta?')
    feedCall(SUMMARIZE_FAST_PATH_EVENTS)
    drain()
    const panel = screen.getByRole('region', { name: 'Query results' })
    expect(within(panel).getByText('summarize')).toBeInTheDocument()
    expect(within(panel).getByText(/We chose it for async support/)).toBeInTheDocument()
    expect(within(panel).getByText('Sources')).toBeInTheDocument()
    expect(within(panel).getByText('1 of 2 results not shown on the graph')).toBeInTheDocument()
    expect(within(panel).getByText('Beta lesson')).toBeInTheDocument()
    expect(within(panel).getByText('hidden by filters')).toBeInTheDocument()
    expect(within(panel).getByText('Alpha decision')).toBeInTheDocument()
  })

  it('(f) Escape closes the query, aborts and refits', async () => {
    await setup()
    await ask('q')
    const signal = signalOf()
    expect(fg.zoomToFit).not.toHaveBeenCalledWith(400, 40, expect.any(Function))
    fireEvent.keyDown(document.body, { key: 'Escape' })
    expect(fg.zoomToFit).toHaveBeenCalledWith(400, 40, expect.any(Function))
    expect(region()).toBeNull()
    expect(signal.aborted).toBe(true)
    for (const l of links()) {
      expect(h.lastProps.linkColor(l)).toBe('rgba(255,255,255,0.08)')
    }
  })

  it('(g) Close query aborts and logs once', async () => {
    await setup()
    await ask('q')
    const signal = signalOf()
    fireEvent.click(screen.getByRole('button', { name: 'Close query' }))
    expect(region()).toBeNull()
    expect(signal.aborted).toBe(true)
    expect(infoCalls()).toHaveLength(1)
    expect(infoCalls()[0][1]).toEqual(
      expect.objectContaining({ outcome: 'aborted', abort_reason: 'close' }),
    )
  })

  it('(h) errors show in an Alert', async () => {
    await setup()
    await ask('q')
    feedCall(ERROR_EVENTS)
    drain()
    expect(within(screen.getByRole('alert')).getByText('RuntimeError: boom')).toBeInTheDocument()
  })

  it('(h1) a transport failure shows in an Alert', async () => {
    h.mockStreamSSE.mockImplementation(() => Promise.reject(new Error('401: Not authenticated')))
    await setup()
    await ask('q')
    await act(async () => {})
    expect(
      within(screen.getByRole('alert')).getByText('401: Not authenticated'),
    ).toBeInTheDocument()
  })

  it('(h2) a late rejection of a superseded run is ignored', async () => {
    let rejectFirst: (e: Error) => void = () => {}
    h.mockStreamSSE.mockImplementationOnce(
      () =>
        new Promise<void>((_res, rej) => {
          rejectFirst = rej
        }),
    )
    await setup()
    await ask('q1')
    await ask('q2')
    rejectFirst(new Error('late'))
    await act(async () => {})
    const panel = screen.getByRole('region', { name: 'Query results' })
    expect(within(panel).getByText('q2')).toBeInTheDocument()
    expect(within(panel).queryByRole('alert')).toBeNull()
    expect(screen.getByTestId('query-progress')).toBeInTheDocument()
  })

  it('(i) a second submit supersedes the first', async () => {
    await setup()
    await ask('q1')
    await ask('q2')
    expect(signalOf(0).aborted).toBe(true)
    feedCall([{ event: 'classified', data: { mode: 'summarize' } }], 0)
    const panel = screen.getByRole('region', { name: 'Query results' })
    expect(within(panel).getByText('classifying')).toBeInTheDocument()
    expect(infoCalls().map((c: any[]) => c[1])).toContainEqual(
      expect.objectContaining({ outcome: 'aborted', abort_reason: 'superseded' }),
    )
  })

  it('(j) asking clears solo without moving the camera', async () => {
    await setup()
    const btn = screen.getByRole('button', { name: 'Solo tag' })
    fireEvent.click(btn)
    expect(btn).toHaveAttribute('aria-pressed', 'true')
    advance(600)
    clearFg()
    await ask('q')
    expect(screen.getByRole('button', { name: 'Solo tag' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    expect(cameraMoves()).toBe(0)
  })

  it('(k) unmounting aborts the stream and logs', async () => {
    const view = await setup()
    await ask('q')
    const signal = signalOf()
    view.unmount()
    expect(signal.aborted).toBe(true)
    expect(infoCalls().map((c: any[]) => c[1])).toContainEqual(
      expect.objectContaining({ outcome: 'aborted', abort_reason: 'unmount' }),
    )
  })

  it('(l) entries without an agent show at once', async () => {
    await setup()
    await ask('q')
    feedCall(EXPLORE_NO_LLM_EVENTS)
    expect(screen.getByText('Found 1 entry')).toBeInTheDocument()
  })

  it('(m) an empty result shows the empty state', async () => {
    await setup()
    await ask('q')
    feedCall(EXPLORE_EMPTY_EVENTS)
    advance(0)
    advance(250)
    advance(250)
    expect(screen.getByText('No matching entries found.')).toBeInTheDocument()
  })

  it('(n) an answer without sources shows no Sources heading', async () => {
    await setup()
    await ask('q')
    feedCall([
      { event: 'classified', data: { mode: 'summarize' } },
      { event: 'synthesis_result', data: { answer: 'x', question: 'q', entry_ids: [] } },
      { event: 'stream_end', data: {} },
    ])
    drain()
    expect(screen.getByText('x')).toBeInTheDocument()
    expect(screen.queryByText('Sources')).toBeNull()
    expect(screen.queryByText('No matching entries found.')).toBeNull()
  })

  it('(o) the ask box: disabled submit, Escape clears and blurs only the input', async () => {
    await setup()
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    await ask('q')
    expect(region()).not.toBeNull()
    await user.type(askBox(), 'abc')
    await user.keyboard('{Escape}')
    expect((askBox() as HTMLInputElement).value).toBe('')
    expect(document.activeElement).not.toBe(askBox())
    expect(region()).not.toBeNull()
  })

  it('(p) blank text in the search box asks nothing', async () => {
    await setup()
    await user.type(searchBox(), '   {Enter}')
    expect(h.mockStreamSSE).not.toHaveBeenCalled()
  })

  it('(q) drift is warned about once each', async () => {
    await setup()
    await ask('q')
    feedCall(DRIFT_EVENTS)
    drain()
    const warns = warnSpy.mock.calls
    const count = (a: unknown, b: unknown) =>
      warns.filter((c: any[]) => c[0] === a && JSON.stringify(c[1]) === JSON.stringify(b)).length
    expect(count('[graph-query] unhandled event', { event: 'tool_progress' })).toBe(1)
    expect(count('[graph-query] malformed event', { key: 'tool_result:entry_ids', count: 1 })).toBe(
      1,
    )
    expect(
      count('[graph-query] no marked node is in the graph', { marked_nodes: 1, sample: ['alpha'] }),
    ).toBe(1)
  })

  it('(r) one summary per run', async () => {
    await setup()
    await ask('q')
    feedCall(EXPLORE_AGENT_EVENTS)
    drain()
    expect(infoCalls()).toHaveLength(1)
    expect(infoCalls()[0][1]).toEqual(
      expect.objectContaining({
        outcome: 'done',
        abort_reason: null,
        error_source: null,
        mode: 'explore',
        path: 'agent',
        turns_used: 3,
        marked_nodes: 4,
        marked_rendered: 3,
        marked_not_in_graph: 1,
        results: 2,
        results_not_in_graph: 1,
      }),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Close query' }))
    expect(infoCalls()).toHaveLength(1)
  })
})
