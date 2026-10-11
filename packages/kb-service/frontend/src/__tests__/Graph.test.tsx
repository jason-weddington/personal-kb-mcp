import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, createRef, useImperativeHandle } from 'react'
import type { Ref } from 'react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { Graph } from '../pages/Graph'
import { GraphExplorer } from '../components/GraphExplorer'
import type { GraphExplorerHandle } from '../components/GraphExplorer'
import { NODE_COLORS, filterGraph } from '../graph/explorerModel'
import type { ExplorerLink, ExplorerNode } from '../graph/explorerModel'
import type { GraphFullResponse } from '../kbTypes'

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
  fgMethods.emitParticle = vi.fn(() => fgMethods)
  return {
    fgState,
    fgMethods,
    lastProps: {} as Record<string, any>,
    mockOpenEntry: vi.fn(),
    getGraphFull: vi.fn(),
    mockStreamSSE: vi.fn(() => new Promise<void>(() => {})),
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

vi.mock('../api', () => ({ getGraphFull: h.getGraphFull, getToken: () => null }))

vi.mock('../streaming', () => ({ streamSSE: h.mockStreamSSE }))

const authState = vi.hoisted(() => ({ authMode: 'none' as 'jwt' | 'none' }))
vi.mock('../contexts/AuthContext', () => ({
  useAuth: () => ({
    authMode: authState.authMode,
    isAuthenticated: true,
    user: null,
    loading: false,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  }),
}))

vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: () => ({ openEntry: h.mockOpenEntry, closeEntry: vi.fn() }),
}))

const roState = {
  size: { width: 1200, height: 900 },
  observers: [] as { cb: ResizeObserverCallback; el: Element; obs: ResizeObserver }[],
}
class ResizeObserverStub {
  private cb: ResizeObserverCallback
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb
  }
  observe(el: Element) {
    roState.observers.push({ cb: this.cb, el, obs: this as unknown as ResizeObserver })
    this.cb(
      [{ target: el, contentRect: { ...roState.size } } as unknown as ResizeObserverEntry],
      this as unknown as ResizeObserver,
    )
  }
  unobserve() {}
  disconnect() {}
}
const fireResize = (width: number) => {
  roState.size.width = width
  act(() => {
    for (const { cb, el, obs } of roState.observers) {
      cb(
        [{ target: el, contentRect: { width, height: 900 } } as unknown as ResizeObserverEntry],
        obs,
      )
    }
  })
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
const nodeOf = (id: string) =>
  (h.lastProps.graphData.nodes as ExplorerNode[]).find((n) => n.id === id)!
const click = (id: string) => act(() => h.lastProps.onNodeClick(nodeOf(id)))
const advance = (ms: number) => act(() => vi.advanceTimersByTime(ms))
const visible = (id: string) => h.lastProps.nodeVisibility(nodeOf(id))
const clearFg = () => Object.values(fg).forEach((f) => f.mockClear())

async function renderPage({ fake = true } = {}) {
  render(<Graph />)
  await screen.findByText('4 of 4 nodes · 3 of 3 edges')
  clearFg()
  if (fake) vi.useFakeTimers()
}

function chooseScope(value: string) {
  fireEvent.mouseDown(screen.getByRole('combobox', { name: /Scope/ }))
  fireEvent.click(screen.getByRole('option', { name: value }))
}

beforeEach(() => {
  authState.authMode = 'none'
  roState.size = { width: 1200, height: 900 }
  roState.observers = []
  h.fgState.cx = 0
  h.fgState.cy = 0
  h.fgState.k = 1
  h.lastProps = {}
  h.mockOpenEntry.mockClear()
  h.mockStreamSSE.mockClear()
  clearFg()
  h.getGraphFull.mockImplementation(() => Promise.resolve(fixture()))
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  window.history.replaceState({}, '', '/')
})

describe('Graph page', () => {
  it('(a, b) renders stats and the tuned force-graph props', async () => {
    await renderPage({ fake: false })
    expect(screen.getByText('4 of 4 nodes · 3 of 3 edges')).toBeInTheDocument()
    expect(h.lastProps.backgroundColor).toBe('#0a0a0f')
    expect(h.lastProps.maxZoom).toBe(12)
    expect(h.lastProps.linkDirectionalParticles).toBe(2)
    expect(h.lastProps.autoPauseRedraw).toBe(false)
    expect(h.lastProps.height).toBe(900)
    expect(h.lastProps.cooldownTicks).toBeUndefined()
    expect(h.lastProps.warmupTicks).toBeUndefined()
  })

  it('(c) a chip toggle after force-graph mutated links keeps fresh string links', async () => {
    await renderPage({ fake: false })
    const nodes = h.lastProps.graphData.nodes as ExplorerNode[]
    for (const l of h.lastProps.graphData.links as any[]) {
      l.source = nodes.find((n) => n.id === l.source)
      l.target = nodes.find((n) => n.id === l.target)
    }
    const group = screen.getByRole('group', { name: 'Node type filters' })
    fireEvent.click(within(group).getByRole('button', { name: 'tag' }))
    expect(h.lastProps.graphData.links).toEqual([
      { source: 'kb-00001', target: 'kb-00002', type: 'related_to' },
    ])
    expect(screen.getByText('2 of 4 nodes · 1 of 3 edges')).toBeInTheDocument()
  })

  it('(d) Scope narrows to the project and its one-hop neighbours', async () => {
    await renderPage({ fake: false })
    chooseScope('p1')
    expect((h.lastProps.graphData.nodes as ExplorerNode[]).map((n) => n.id)).toEqual([
      'kb-00001',
      'tag:alpha',
    ])
    expect(h.lastProps.graphData.links).toHaveLength(1)
  })

  it('(e) node click focuses and opens the drawer at 850 ms', async () => {
    await renderPage()
    click('kb-00001')
    expect(fg.centerAt).toHaveBeenCalledWith(90, 20, 800)
    expect(fg.zoom).toHaveBeenCalledWith(4, 800)
    advance(849)
    expect(h.mockOpenEntry).not.toHaveBeenCalled()
    advance(1)
    expect(h.mockOpenEntry).toHaveBeenCalledWith('kb-00001')

    click('tag:alpha')
    expect(fg.centerAt).toHaveBeenCalledWith(50, 60, 800)
    advance(849)
    expect(screen.queryByRole('region', { name: 'Node details' })).toBeNull()
    advance(1)
    expect(
      within(screen.getByRole('region', { name: 'Node details' })).getAllByText('alpha').length,
    ).toBeGreaterThan(0)
  })

  it('(f) Escape fits and clamps at 450 ms', async () => {
    await renderPage()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    h.fgState.k = 8
    advance(449)
    expect(fg.zoom).not.toHaveBeenCalledWith(6, 300)
    advance(1)
    expect(fg.zoom).toHaveBeenCalledWith(6, 300)
  })

  it('(g) the solo legend shows one type and hides links', async () => {
    await renderPage()
    const btn = screen.getByRole('button', { name: 'Solo tag' })
    fireEvent.click(btn)
    expect(btn).toHaveAttribute('aria-pressed', 'true')
    expect(visible('tag:alpha')).toBe(true)
    expect(visible('kb-00001')).toBe(false)
    expect(h.lastProps.linkVisibility(h.lastProps.graphData.links[0])).toBe(false)
  })

  it('(h) search ranks the exact label first; Enter asks, ArrowDown+Enter flies', async () => {
    await renderPage()
    const input = screen.getByPlaceholderText('Search nodes or ask a question...')
    fireEvent.change(input, { target: { value: 'alpha' } })
    const options = screen.getAllByRole('option')
    expect(options[0]).toHaveTextContent('alpha')
    expect(options[1]).toHaveTextContent('Alpha decision')
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(h.mockStreamSSE).toHaveBeenCalledWith(
      '/api/kb/query/stream',
      { question: 'alpha' },
      expect.any(Function),
      expect.any(AbortSignal),
    )
    expect(fg.centerAt).not.toHaveBeenCalledWith(50, 60, 800)

    fireEvent.change(input, { target: { value: 'alpha' } })
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(fg.centerAt).toHaveBeenCalledWith(50, 60, 800)
  })

  it('(h2) one Escape in search clears and blurs without moving the graph', async () => {
    await renderPage()
    const input = screen.getByPlaceholderText(
      'Search nodes or ask a question...',
    ) as HTMLInputElement
    input.focus()
    fireEvent.change(input, { target: { value: 'alpha' } })
    const before = fg.zoom.mock.calls.length
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(fg.zoom.mock.calls.length).toBe(before)
    expect(input.value).toBe('')
    expect(document.activeElement).not.toBe(input)
  })

  it('(i) ArrowLeft returns to the previous focused node and camera', async () => {
    await renderPage()
    click('kb-00001')
    click('tag:alpha')
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(90, 20, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    expect(fg.centerAt).not.toHaveBeenCalledWith(0, 0, 600)
    advance(650)
    expect(h.mockOpenEntry).toHaveBeenCalledWith('kb-00001')
  })

  it('(k) the node panel is non-modal', async () => {
    await renderPage()
    click('tag:alpha')
    advance(850)
    expect(screen.getByRole('region', { name: 'Node details' })).toBeInTheDocument()
    click('kb-00002')
    expect(fg.centerAt).toHaveBeenCalledWith(110, 40, 800)
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenCalledWith(expect.anything(), expect.anything(), 600)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    expect(screen.queryByRole('region', { name: 'Node details' })).toBeNull()
  })

  it('(m) narrowing Scope drops a focus on an unrendered node', async () => {
    await renderPage()
    click('kb-00002')
    chooseScope('p1')
    expect(visible('kb-00001')).toBe(true)
    expect(visible('tag:alpha')).toBe(true)
    fg.centerAt.mockClear()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).not.toHaveBeenCalled()
  })

  it('(n) all chips off shows the empty state', async () => {
    await renderPage({ fake: false })
    const group = screen.getByRole('group', { name: 'Node type filters' })
    for (const type of Object.keys(NODE_COLORS)) {
      fireEvent.click(within(group).getByRole('button', { name: type }))
    }
    expect(screen.getByText('No nodes match the current filters')).toBeInTheDocument()
  })

  it('(p) ?perf=1 enables the perf hooks and readout', async () => {
    window.history.replaceState({}, '', '/graph?perf=1')
    await renderPage({ fake: false })
    expect(typeof h.lastProps.onRenderFramePost).toBe('function')
    expect(typeof h.lastProps.onEngineStop).toBe('function')
    expect(screen.getByTestId('graph-perf')).toBeInTheDocument()
  })

  it('(p) without the flag there are no perf hooks', async () => {
    await renderPage({ fake: false })
    expect(h.lastProps.onRenderFramePost).toBeUndefined()
    expect(h.lastProps.onEngineStop).toBeUndefined()
    expect(screen.queryByTestId('graph-perf')).toBeNull()
  })
})

describe('GraphExplorer handle', () => {
  function renderExplorer(
    onEscape?: () => boolean,
    extra: Partial<React.ComponentProps<typeof GraphExplorer>> = {},
  ) {
    const f = fixture()
    const graph: { nodes: ExplorerNode[]; links: ExplorerLink[] } = filterGraph(
      f.nodes,
      f.edges,
      new Set(Object.keys(NODE_COLORS)),
      '',
    )
    const ref = createRef<GraphExplorerHandle>()
    vi.useFakeTimers()
    render(<GraphExplorer ref={ref} graph={graph} onEscape={onEscape} {...extra} />)
    clearFg()
    return { ref, graph }
  }

  it('(j1) flyToNode on an unrendered node logs at debug and returns false', () => {
    const { ref } = renderExplorer()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {})
    let ok: boolean | undefined
    act(() => {
      ok = ref.current!.flyToNode('kb-99999')
    })
    expect(ok).toBe(false)
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(warn).not.toHaveBeenCalled()
    expect(debug).toHaveBeenCalledTimes(1)
  })

  it('(j2) flyToNode clears solo and focuses', () => {
    const { ref } = renderExplorer()
    fireEvent.click(screen.getByRole('button', { name: 'Solo tag' }))
    let ok: boolean | undefined
    act(() => {
      ok = ref.current!.flyToNode('kb-00001')
    })
    expect(ok).toBe(true)
    expect(visible('kb-00001')).toBe(true)
    expect(visible('tag:alpha')).toBe(true)
    expect(visible('kb-00002')).toBe(true)
    expect(visible('tag:beta')).toBe(false)
    expect(fg.centerAt).toHaveBeenCalledWith(90, 20, 800)
  })

  it('(j3) fitNodes fits a subset and clamps at duration + 50 ms', () => {
    const { ref } = renderExplorer()
    act(() => ref.current!.fitNodes(new Set(['kb-00001']), 500, 60))
    expect(fg.zoom).toHaveBeenCalledWith(6, 500)
    expect(fg.centerAt).toHaveBeenCalledWith(10, 20, 500)
    h.fgState.k = 8
    advance(549)
    expect(fg.zoom).not.toHaveBeenCalledWith(6, 300)
    advance(1)
    expect(fg.zoom).toHaveBeenCalledWith(6, 300)
  })

  it('(j4) emitParticlesFor emits on touching links', () => {
    const { ref, graph } = renderExplorer()
    act(() => ref.current!.emitParticlesFor(new Set(['tag:alpha'])))
    expect(fg.emitParticle).toHaveBeenCalledTimes(1)
    expect(fg.emitParticle.mock.calls[0][0]).toBe(graph.links[0])
  })

  it('(j5) clearSelection drops solo without moving the camera', () => {
    const { ref } = renderExplorer()
    fireEvent.click(screen.getByRole('button', { name: 'Solo tag' }))
    clearFg()
    act(() => ref.current!.clearSelection())
    for (const id of ['kb-00001', 'kb-00002', 'tag:alpha', 'tag:beta']) {
      expect(visible(id)).toBe(true)
    }
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
  })

  it('(j6) resetAll fits and clears solo', () => {
    const { ref } = renderExplorer()
    const btn = screen.getByRole('button', { name: 'Solo tag' })
    fireEvent.click(btn)
    act(() => ref.current!.resetAll())
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    expect(btn).toHaveAttribute('aria-pressed', 'false')
  })

  it('(j7) onEscape returning true consumes Escape', () => {
    renderExplorer(() => true)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).not.toHaveBeenCalled()
  })

  it('(j7) onEscape returning false lets Escape reset', () => {
    renderExplorer(() => false)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
  })

  it('(j8) rightInset narrows the entry drawer shift', () => {
    const { ref } = renderExplorer(undefined, { rightInset: 420 })
    act(() => {
      ref.current!.flyToNode('kb-00001')
    })
    expect(fg.centerAt).toHaveBeenCalledWith(37.5, 20, 800)
  })

  it('(j9) a width change during a traversal refits the results once', () => {
    const traversal = {
      queryMode: 'summarize' as const,
      nodeStates: new Map([['kb-00001', 'result' as const]]),
      pulseStarts: new Map<string, number>(),
    }
    renderExplorer(undefined, { traversal })
    advance(2000)
    clearFg()
    fireResize(780)
    advance(2000)
    const fits = () => fg.zoom.mock.calls.filter((c: any[]) => c[1] === 600).length
    expect(fits()).toBe(1)
    expect(fg.zoom).toHaveBeenCalledWith(6, 600)
    expect(fg.centerAt).toHaveBeenCalledWith(10, 20, 600)
    fireResize(780)
    advance(2000)
    expect(fits()).toBe(1)
  })

  it('(j9b) without a traversal the resize refit does not fit results', () => {
    renderExplorer()
    advance(2000)
    clearFg()
    fireResize(700)
    advance(2000)
    expect(fg.zoom.mock.calls.some((c: any[]) => c[1] === 600)).toBe(false)
  })

  it('(j10) backtrack keeps the inset-aware focus', () => {
    renderExplorer(undefined, { rightInset: 420 })
    click('kb-00001')
    click('tag:alpha')
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(37.5, 20, 600)
  })
})
