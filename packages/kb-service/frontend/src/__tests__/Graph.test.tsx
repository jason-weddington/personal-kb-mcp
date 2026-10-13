import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, createRef, useImperativeHandle } from 'react'
import type { Ref } from 'react'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { Graph } from '../pages/Graph'
import { GraphExplorer } from '../components/GraphExplorer'
import type { GraphExplorerHandle } from '../components/GraphExplorer'
import { getEntries, getGraphNeighbors } from '../api'
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
    mockCloseEntry: vi.fn(),
    nodeHandler: null as ((id: string) => void) | null,
    getGraphFull: vi.fn(),
    getEntries: vi.fn(),
    getGraphNeighbors: vi.fn(),
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

vi.mock('../api', () => ({
  getGraphFull: h.getGraphFull,
  getToken: () => null,
  getEntries: h.getEntries,
  getGraphNeighbors: h.getGraphNeighbors,
}))

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
  useEntryDrawer: () => ({
    openEntry: h.mockOpenEntry,
    closeEntry: h.mockCloseEntry,
    openNode: vi.fn(),
    setNodeHandler: (fn: ((id: string) => void) | null) => {
      h.nodeHandler = fn
    },
  }),
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
        properties: {
          project_ref: 'p1',
          short_title: 'Alpha decision',
          long_title: 'Alpha decision, the long form',
          entry_type: 'decision',
          tags: ['alpha'],
          confidence_level: 0.9,
        },
        x: 10,
        y: 20,
      },
      {
        id: 'kb-00002',
        type: 'entry',
        label: 'Beta lesson',
        val: 2,
        properties: {
          project_ref: 'p2',
          short_title: 'Beta lesson',
          long_title: 'Beta lesson, the long form',
          entry_type: 'lesson_learned',
          tags: ['beta'],
          confidence_level: 0.9,
        },
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

function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="loc">{loc.pathname + loc.search}</div>
}

async function renderPage({ fake = true, url = '/graph' } = {}) {
  render(
    <MemoryRouter initialEntries={[url]}>
      <Graph />
      <LocationProbe />
    </MemoryRouter>,
  )
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
  h.mockCloseEntry.mockClear()
  h.nodeHandler = null
  h.mockStreamSSE.mockClear()
  vi.mocked(getEntries).mockReset()
  // The entry inspector's Details opens expanded and fetches on open: a
  // found default entry keeps every focused entry inspector quiet.
  vi.mocked(getEntries).mockImplementation((ids: string[]) =>
    Promise.resolve({
      results: ids.map((id) => ({
        id,
        found: true,
        entry: inspectorEntry(id),
        pointer_rot: [],
      })),
    }),
  )
  vi.mocked(getGraphNeighbors).mockReset()
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

  it('(e) node click focuses and opens the inspector at 850 ms', async () => {
    await renderPage()
    click('kb-00001')
    expect(fg.centerAt).toHaveBeenCalledWith(65, 20, 800)
    expect(fg.zoom).toHaveBeenCalledWith(4, 800)
    advance(849)
    expect(screen.queryByRole('region', { name: 'Node inspector' })).toBeNull()
    advance(1)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Alpha decision')
        .length,
    ).toBeGreaterThan(0)
    expect(h.mockOpenEntry).not.toHaveBeenCalled()

    click('tag:alpha')
    expect(fg.centerAt).toHaveBeenCalledWith(50, 60, 800)
    advance(850)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('alpha').length,
    ).toBeGreaterThan(0)
    expect(h.mockOpenEntry).not.toHaveBeenCalled()
  })

  it('(f) Escape keeps the history: Back restores the node and camera you left', async () => {
    await renderPage()
    click('kb-00001')
    click('kb-00002')
    // Back rewrites kb-00002's record with the live camera and restores
    // kb-00001's record: the pre-move camera (4, 65, 20)
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Alpha decision')
        .length,
    ).toBeGreaterThan(0)
    // the first Escape closes ONLY the inspector; cursor 0 keeps Back disabled
    clearFg()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('region', { name: 'Node inspector' })).toBeNull()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled()
    // the next Escape unfocuses (history and cursor kept) and fits + clamps
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    h.fgState.k = 8
    advance(449)
    expect(fg.zoom).not.toHaveBeenCalledWith(6, 300)
    advance(1)
    expect(fg.zoom).toHaveBeenCalledWith(6, 300)
    expect(screen.getByRole('button', { name: 'Back' })).toBeEnabled()
    // Back while nothing is focused re-focuses the record at the cursor
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Alpha decision')
        .length,
    ).toBeGreaterThan(0)
  })

  it('(g) no solo legend exists; the type-chip row drives filterGraph', async () => {
    await renderPage()
    // the solo legend and its per-type buttons are gone for good
    expect(screen.queryByRole('group', { name: 'Solo node type' })).toBeNull()
    expect(screen.queryByRole('button', { name: /Solo / })).toBeNull()
    const group = screen.getByRole('group', { name: 'Node type filters' })
    // clicking tag off drops that type's nodes and their links
    fireEvent.click(within(group).getByRole('button', { name: 'tag' }))
    const tagChip = within(group).getByRole('button', { name: 'tag' })
    // an OFF chip reads as grey: outlined, never a coloured border or fill
    expect(tagChip).toHaveClass('MuiChip-outlined')
    expect(tagChip).not.toHaveStyle({ backgroundColor: NODE_COLORS.tag })
    expect((h.lastProps.graphData.nodes as ExplorerNode[]).map((n) => n.id)).toEqual([
      'kb-00001',
      'kb-00002',
    ])
    expect(h.lastProps.graphData.links).toEqual([
      { source: 'kb-00001', target: 'kb-00002', type: 'related_to' },
    ])
    expect(screen.getByText('2 of 4 nodes · 1 of 3 edges')).toBeInTheDocument()
    // clicking it back on returns them
    fireEvent.click(within(group).getByRole('button', { name: 'tag' }))
    expect(visible('tag:alpha')).toBe(true)
    expect(visible('tag:beta')).toBe(true)
    expect(visible('kb-00001')).toBe(true)
    expect(h.lastProps.graphData.links).toHaveLength(3)
    expect(screen.getByText('4 of 4 nodes · 3 of 3 edges')).toBeInTheDocument()
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
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    expect(fg.centerAt).not.toHaveBeenCalledWith(0, 0, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Alpha decision')
        .length,
    ).toBeGreaterThan(0)
    expect(h.mockOpenEntry).not.toHaveBeenCalled()
  })

  it('(k) the inspector is non-modal and Escape closes it before resetting', async () => {
    await renderPage()
    click('tag:alpha')
    advance(850)
    expect(screen.getByRole('region', { name: 'Node inspector' })).toBeInTheDocument()
    click('kb-00002')
    expect(fg.centerAt).toHaveBeenCalledWith(85, 40, 800)
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    // tag:alpha's leave-camera: a non-entry focus has no camera shift
    expect(fg.centerAt).toHaveBeenLastCalledWith(50, 60, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('alpha').length,
    ).toBeGreaterThan(0)
    // the first Escape closes ONLY the inspector
    clearFg()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('region', { name: 'Node inspector' })).toBeNull()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(fg.centerAt).not.toHaveBeenCalled()
    // the next Escape unfocuses (history kept) and fits + clamps
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    h.fgState.k = 8
    advance(449)
    expect(fg.zoom).not.toHaveBeenCalledWith(6, 300)
    advance(1)
    expect(fg.zoom).toHaveBeenCalledWith(6, 300)
    // nothing focused: Back re-focuses the record at the cursor (tag:alpha)
    expect(screen.getByRole('button', { name: 'Back' })).toBeEnabled()
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(50, 60, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('alpha').length,
    ).toBeGreaterThan(0)
  })

  it('(m) narrowing Scope drops a focus on an unrendered node', async () => {
    await renderPage()
    click('kb-00002')
    chooseScope('p1')
    expect(visible('kb-00001')).toBe(true)
    expect(visible('tag:alpha')).toBe(true)
    // the focused node is unrendered: the whole history clears, so Back is
    // disabled and ArrowLeft produces no camera call at all
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled()
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
  })

  it('(b1) the Back and Forward buttons mirror the history cursor', async () => {
    await renderPage()
    const back = screen.getByRole('button', { name: 'Back' })
    const forward = screen.getByRole('button', { name: 'Forward' })
    expect(back).toBeDisabled()
    expect(forward).toBeDisabled()
    click('kb-00001')
    advance(850)
    // focused at cursor 0: nothing to walk back to yet
    expect(back).toBeDisabled()
    expect(forward).toBeDisabled()
    click('kb-00002')
    expect(back).toBeEnabled()
    expect(forward).toBeDisabled()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    expect(back).toBeDisabled()
    expect(forward).toBeEnabled()
    fireEvent.keyDown(document, { key: 'ArrowRight' })
    advance(650)
    expect(back).toBeEnabled()
    expect(forward).toBeDisabled()
  })

  it('(b2) Back restores the camera and the inspector of the node you left', async () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    await renderPage()
    click('kb-00001')
    advance(850)
    click('kb-00002')
    advance(850)
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    // kb-00001's record was rewritten with the pre-move camera (4, 65, 20)
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Alpha decision')
        .length,
    ).toBeGreaterThan(0)
    expect(h.mockOpenEntry).not.toHaveBeenCalled()
    // telemetry: the walk and the restore, with the pinned payload shape
    const selectCalls = info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-history] select')
    expect(selectCalls).toHaveLength(2)
    expect(selectCalls[0][1]).toMatchObject({
      node_id: 'kb-00001',
      cursor: 0,
      history_len: 1,
      truncated: 0,
      dropped_oldest: false,
      can_back: false,
      can_forward: false,
      visited_count: 1,
    })
    expect(selectCalls[1][1]).toMatchObject({
      node_id: 'kb-00002',
      cursor: 1,
      history_len: 2,
      truncated: 0,
      can_back: true,
      can_forward: false,
      visited_count: 2,
    })
    const backCalls = info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-history] back')
    expect(backCalls).toHaveLength(1)
    expect(backCalls[0][1]).toMatchObject({
      node_id: 'kb-00001',
      cursor: 0,
      history_len: 2,
      can_back: false,
      can_forward: true,
      visited_count: 2,
    })
  })

  it('(b3) Forward restores the Back-leave snapshot, by arrow key and by button', async () => {
    await renderPage()
    click('kb-00001')
    advance(850)
    click('kb-00002')
    advance(850)
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    // Forward by arrow key: kb-00001's record takes the outgoing camera
    // (65, 20, 4) and kb-00002's record restores (85, 40)
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowRight' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(85, 40, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Beta lesson')
        .length,
    ).toBeGreaterThan(0)
    // Forward by button click, from cursor 0 again
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    clearFg()
    fireEvent.click(screen.getByRole('button', { name: 'Forward' }))
    expect(fg.centerAt).toHaveBeenLastCalledWith(85, 40, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Beta lesson')
        .length,
    ).toBeGreaterThan(0)
    // back to cursor 0, then a further Back is a reducer no-op with no fg call
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
  })

  it('(h3) arrow keys in the search input never walk the history', async () => {
    await renderPage()
    const input = screen.getByPlaceholderText(
      'Search nodes or ask a question...',
    ) as HTMLInputElement
    input.focus()
    clearFg()
    fireEvent.keyDown(input, { key: 'ArrowLeft' })
    fireEvent.keyDown(input, { key: 'ArrowRight' })
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
  })

  it('(h4) a search pick truncates everything forward of the cursor', async () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    await renderPage()
    click('kb-00001')
    click('kb-00002')
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    // back at cursor 0 on kb-00001; the search pick flies to tag:beta and
    // drops the forward record (kb-00002) from the history
    const input = screen.getByPlaceholderText('Search nodes or ask a question...')
    fireEvent.change(input, { target: { value: 'beta' } })
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(fg.centerAt).toHaveBeenCalledWith(70, 80, 800)
    expect(screen.getByRole('button', { name: 'Forward' })).toBeDisabled()
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowRight' })
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
    const selectCalls = info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-history] select')
    expect(selectCalls.at(-1)![1]).toMatchObject({
      node_id: 'tag:beta',
      cursor: 1,
      history_len: 2,
      truncated: 1,
      can_back: true,
      can_forward: false,
    })
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

function inspectorEntry(id: string) {
  return {
    id,
    project_ref: 'p1',
    short_title: 'Alpha decision',
    long_title: 'Alpha decision, the long form',
    knowledge_details: `Body for ${id}`,
    entry_type: 'decision' as const,
    source_context: null,
    contributor: null,
    team: null,
    updated_by: null,
    sensitivity: null,
    confidence_level: 0.9,
    tags: ['alpha'],
    hints: {},
    created_at: null,
    updated_at: '2026-03-03T12:00:00Z',
    last_accessed: null,
    expires_at: null,
    superseded_by: null,
    is_active: true,
    has_embedding: true,
    version: 1,
  }
}

describe('Graph page — node inspector', () => {
  it('(n1) emits the [graph-inspector] open info event once per target', async () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    await renderPage()
    click('tag:alpha')
    advance(850)
    const calls = info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] open')
    expect(calls).toHaveLength(1)
    expect(calls[0][1]).toMatchObject({
      node_id: 'tag:alpha',
      node_type: 'tag',
      member_total: 1,
      groups: [{ key: 'has_tag:in', count: 1 }],
    })
  })

  it('(n2) connections come from the payload; Details is expanded by default and fetches the entry exactly once', async () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    vi.mocked(getEntries).mockResolvedValue({
      results: [{ id: 'kb-00001', found: true, entry: inspectorEntry('kb-00001'), pointer_rot: [] }],
    })
    await renderPage()
    click('kb-00001')
    advance(850)
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledTimes(0)
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(1)
    expect(vi.mocked(getEntries)).toHaveBeenCalledWith(['kb-00001'], { touch: false })
    // exactly one details-fetch info line for the shown id
    expect(
      info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] details fetch'),
    ).toHaveLength(1)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    expect(within(region).getByText('p1 · 0.90')).toBeInTheDocument()
    // the header tag chip and project chip ARE the has_tag:out/in_project:out groups
    expect(within(region).getByRole('button', { name: 'alpha' })).toBeInTheDocument()
    expect(within(region).queryByText(/has_tag:out/)).toBeNull()
    await act(async () => {})
    const date = new Date('2026-03-03T12:00:00Z').toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
    expect(within(region).getByText(`p1 · 0.90 · updated ${date}`)).toBeInTheDocument()
    // toggling Details closed and open again does not refetch
    fireEvent.click(within(region).getByText('Details'))
    fireEvent.click(within(region).getByText('Details'))
    await act(async () => {})
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(1)
    expect(
      info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] details fetch'),
    ).toHaveLength(1)
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledTimes(0)
  })

  it('(n3) the first Escape closes only the inspector; the second resets the view', async () => {
    await renderPage()
    click('tag:alpha')
    advance(850)
    expect(screen.getByRole('region', { name: 'Node inspector' })).toBeInTheDocument()
    clearFg()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('region', { name: 'Node inspector' })).toBeNull()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(fg.centerAt).not.toHaveBeenCalled()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    h.fgState.k = 8
    advance(449)
    expect(fg.zoom).not.toHaveBeenCalledWith(6, 300)
    advance(1)
    expect(fg.zoom).toHaveBeenCalledWith(6, 300)
  })

  it('(n4) a background click closes the inspector and fits after 100 ms', async () => {
    await renderPage()
    click('tag:alpha')
    advance(850)
    expect(screen.getByRole('region', { name: 'Node inspector' })).toBeInTheDocument()
    clearFg()
    act(() => h.lastProps.onBackgroundClick())
    expect(screen.queryByRole('region', { name: 'Node inspector' })).toBeNull()
    // the unfocus reads the leave-camera, but no camera move happens before the fit
    advance(99)
    expect(fg.zoom).not.toHaveBeenCalledWith(expect.any(Number), 400)
    advance(1)
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
  })

  it('(n5) the canvas stays interactive: re-targeting keeps the region mounted', async () => {
    await renderPage()
    click('tag:alpha')
    advance(850)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    click('kb-00002')
    expect(fg.centerAt).toHaveBeenCalledWith(85, 40, 800)
    advance(850)
    expect(region).toBeInTheDocument()
    expect(within(region).getAllByText('Beta lesson').length).toBeGreaterThan(0)
  })

  it('(n6) a header tag chip click walks to the tag like a canvas click', async () => {
    await renderPage()
    click('kb-00001')
    advance(850)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    clearFg()
    fireEvent.click(within(region).getByRole('button', { name: 'alpha' }))
    expect(fg.centerAt).toHaveBeenCalledWith(50, 60, 800)
    // history pushed: ArrowLeft returns to the entry's camera
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
  })

  it('(n7) a row click walks to the member like a canvas click on it', async () => {
    const nodes: ExplorerNode[] = [
      {
        id: 'kb-00001',
        type: 'entry',
        label: 'Hub',
        val: 2,
        properties: { project_ref: 'p1' },
        x: 10,
        y: 20,
      },
      ...Array.from({ length: 13 }, (_, i) => ({
        id: `kb-1000${i}`,
        type: 'entry',
        label: `Row ${i}`,
        val: 1,
        properties: { project_ref: 'p2' },
        x: 30,
        y: 40 + i,
      })),
    ]
    const links = Array.from({ length: 13 }, (_, i) => ({
      source: 'kb-00001',
      target: `kb-1000${i}`,
      type: 'related_to',
    }))
    vi.useFakeTimers()
    render(<GraphExplorer graph={{ nodes, links }} />)
    clearFg()
    act(() => h.lastProps.onNodeClick(nodes[0]))
    advance(850)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    fireEvent.click(within(region).getByRole('button', { name: 'related to → 13' }))
    expect(within(region).getAllByText(/Row \d+/).length).toBe(13)
    clearFg()
    fireEvent.click(within(region).getByRole('button', { name: /Row 5/ }))
    // the same camera call a canvas click on kb-10005 produces
    expect(fg.centerAt).toHaveBeenCalledWith(85, 45, 800)
    // history pushed: ArrowLeft returns to the hub's camera
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
  })

  it('(n8) the inspector scroll container reports scrollTop and the reset restores it', async () => {
    await renderPage()
    click('tag:alpha')
    advance(850)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    const scroll = within(region).getByTestId('inspector-scroll')
    act(() => {
      scroll.scrollTop = 120
    })
    fireEvent.scroll(scroll)
    expect(scroll.scrollTop).toBe(120)
    // re-targeting resets the lifted view state, so the scroll container
    // snaps back to the top
    click('kb-00002')
    advance(850)
    expect(region).toBeInTheDocument()
    expect(within(region).getByTestId('inspector-scroll').scrollTop).toBe(0)
  })

  it('(v1) visited members carry a trailing row check that survives an unfocus', async () => {
    await renderPage()
    click('kb-00001')
    advance(850)
    click('kb-00002')
    advance(850)
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    // kb-00002 was focused this session: its related_to:out member row shows
    // the trailing check (the row's accessible name is composite)
    const betaRow = within(region).getByRole('button', { name: /Beta lesson/ })
    expect(within(betaRow).getByTestId('visited-check')).toBeInTheDocument()
    // the entry-header tag chip is excluded from visited marks (and tag:alpha
    // was never walked)
    const alphaChip = within(region).getByRole('button', { name: 'alpha' })
    expect(within(alphaChip).queryByTestId('visited-check')).toBeNull()
    // Escape-unfocus (inspector close, then reset) keeps the session set
    fireEvent.keyDown(document, { key: 'Escape' })
    fireEvent.keyDown(document, { key: 'Escape' })
    advance(500)
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    advance(650)
    const reopened = screen.getByRole('region', { name: 'Node inspector' })
    const betaRowAgain = within(reopened).getByRole('button', { name: /Beta lesson/ })
    expect(within(betaRowAgain).getByTestId('visited-check')).toBeInTheDocument()
  })
})

// select() reads centerAt() then moves it: one fly is one call with coordinates
const flies = () => fg.centerAt.mock.calls.filter((c: unknown[]) => c.length > 0).length

describe('Graph page — connection chips', () => {
  it('(q1) registers a node handler that closes the drawer and flies', async () => {
    await renderPage({ fake: false })
    expect(h.nodeHandler).not.toBeNull()
    act(() => h.nodeHandler!('tag:alpha'))
    expect(h.mockCloseEntry).toHaveBeenCalledTimes(1)
    expect(fg.centerAt).toHaveBeenCalled()
  })

  it('(q2) a filtered-out node resets filters and retries once', async () => {
    await renderPage({ fake: false })
    const group = screen.getByRole('group', { name: 'Node type filters' })
    fireEvent.click(within(group).getByRole('button', { name: 'tag' }))
    expect(screen.getByText('2 of 4 nodes · 1 of 3 edges')).toBeInTheDocument()
    clearFg()
    act(() => h.nodeHandler!('tag:alpha'))
    expect(screen.getByText('4 of 4 nodes · 3 of 3 edges')).toBeInTheDocument()
    expect(flies()).toBe(1)
  })

  it('(q3) an unknown id logs at debug and does nothing', async () => {
    await renderPage({ fake: false })
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {})
    clearFg()
    act(() => h.nodeHandler!('tag:nope'))
    expect(debug).toHaveBeenCalledWith('[graph] openNode: node not rendered', {
      nodeId: 'tag:nope',
    })
    expect(fg.centerAt).not.toHaveBeenCalled()
  })

  it('(q4) ?focus flies to the node once and clears the param', async () => {
    render(
      <MemoryRouter initialEntries={['/graph?focus=tag%3Aalpha']}>
        <Graph />
        <LocationProbe />
      </MemoryRouter>,
    )
    await screen.findByText('4 of 4 nodes · 3 of 3 edges')
    expect(flies()).toBe(1)
    await waitFor(() => expect(screen.getByTestId('loc').textContent).toBe('/graph'))
    // the fly is a walk: Back is enabled and restores the walked entry
    vi.useFakeTimers()
    click('kb-00001')
    click('kb-00002')
    expect(screen.getByRole('button', { name: 'Back' })).toBeEnabled()
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(65, 20, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Alpha decision')
        .length,
    ).toBeGreaterThan(0)
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

  it('(j2) flyToNode focuses the node and its neighbourhood', () => {
    const { ref } = renderExplorer()
    let ok: boolean | undefined
    act(() => {
      ok = ref.current!.flyToNode('kb-00001')
    })
    expect(ok).toBe(true)
    expect(visible('kb-00001')).toBe(true)
    expect(visible('tag:alpha')).toBe(true)
    expect(visible('kb-00002')).toBe(true)
    expect(visible('tag:beta')).toBe(false)
    expect(fg.centerAt).toHaveBeenCalledWith(65, 20, 800)
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

  it('(j5) clearSelection drops focus without moving the camera', () => {
    const { ref } = renderExplorer()
    click('kb-00001')
    clearFg()
    act(() => ref.current!.clearSelection())
    for (const id of ['kb-00001', 'kb-00002', 'tag:alpha', 'tag:beta']) {
      expect(visible(id)).toBe(true)
    }
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
  })

  it('(j6) resetAll fits and clears focus', () => {
    const { ref } = renderExplorer()
    click('kb-00001')
    act(() => ref.current!.resetAll())
    expect(fg.zoom).toHaveBeenCalledWith(expect.any(Number), 400)
    for (const id of ['kb-00001', 'kb-00002', 'tag:alpha', 'tag:beta']) {
      expect(visible(id)).toBe(true)
    }
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
    expect(fg.centerAt).toHaveBeenCalledWith(12.5, 20, 800)
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
    expect(fg.centerAt).toHaveBeenLastCalledWith(12.5, 20, 600)
  })

  it('(j11) a background click unfocuses but keeps the history for Back', () => {
    const info: Array<{ canBack: boolean; canForward: boolean }> = []
    renderExplorer(undefined, {
      onHistoryChange: (i) => info.push({ canBack: i.canBack, canForward: i.canForward }),
    })
    click('kb-00001')
    click('kb-00002')
    clearFg()
    act(() => h.lastProps.onBackgroundClick())
    // nothing focused with history behind: Back can re-focus the node just left
    expect(info.at(-1)).toEqual({ canBack: true, canForward: false })
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    expect(fg.centerAt).toHaveBeenLastCalledWith(85, 40, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Beta lesson')
        .length,
    ).toBeGreaterThan(0)
  })

  it('(j12) resetAll keeps the history: Back re-focuses the captured camera', () => {
    const { ref } = renderExplorer()
    click('kb-00001')
    click('kb-00002')
    advance(850)
    expect(screen.getByRole('region', { name: 'Node inspector' })).toBeInTheDocument()
    act(() => ref.current!.resetAll())
    expect(screen.queryByRole('region', { name: 'Node inspector' })).toBeNull()
    clearFg()
    let ok: boolean | undefined
    act(() => {
      ok = ref.current!.back()
    })
    expect(ok).toBe(true)
    expect(fg.centerAt).toHaveBeenCalledWith(85, 40, 600)
    expect(fg.zoom).toHaveBeenCalledWith(4, 600)
    // a second Back walks to cursor 0, and Forward returns true and restores
    // the kb-00002 record again
    let okBack: boolean | undefined
    let okForward: boolean | undefined
    clearFg()
    act(() => {
      okBack = ref.current!.back()
    })
    expect(okBack).toBe(true)
    clearFg()
    act(() => {
      okForward = ref.current!.forward()
    })
    expect(okForward).toBe(true)
    expect(fg.centerAt).toHaveBeenLastCalledWith(85, 40, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    expect(
      within(screen.getByRole('region', { name: 'Node inspector' })).getAllByText('Beta lesson')
        .length,
    ).toBeGreaterThan(0)
  })

  it('(j13) back and forward return false and stay silent on an empty history', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    const { ref } = renderExplorer()
    let back: boolean | undefined
    let forward: boolean | undefined
    act(() => {
      back = ref.current!.back()
    })
    act(() => {
      forward = ref.current!.forward()
    })
    expect(back).toBe(false)
    expect(forward).toBe(false)
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(
      info.mock.calls.filter((c: unknown[]) => String(c[0]).startsWith('[graph-history]')),
    ).toHaveLength(0)
  })
})

// ── Group filters: narrowing a large connection group (item C) ─────────────

function fixtureNarrow(): GraphFullResponse {
  const nodes: ExplorerNode[] = [
    { id: 'tag:mass', type: 'tag', label: 'mass', val: 15, properties: {}, x: 40, y: 70 },
  ]
  const edges = []
  for (let i = 0; i < 15; i++) {
    const n = String(i).padStart(2, '0')
    const deep = i === 0 ? 'Deep background for mass entry 0' : `Mass entry ${i} detail`
    nodes.push({
      id: `kb-001${n}`,
      type: 'entry',
      label: `Mass entry ${i}`,
      val: 1,
      properties: {
        project_ref: i <= 5 ? 'p1' : 'p2',
        short_title: `Mass entry ${i}`,
        long_title: deep,
        entry_type: i <= 5 ? 'decision' : i <= 12 ? 'factual_reference' : 'lesson_learned',
      },
      x: 20 + i * 2,
      y: 70,
    })
    edges.push({ source: `kb-001${n}`, target: 'tag:mass', type: 'has_tag', properties: {} })
  }
  return {
    nodes,
    edges,
    stats: { node_count: 16, edge_count: 15 },
  }
}

describe('GraphExplorer — group filters', () => {
  const narrowId = (i: number) => `kb-001${String(i).padStart(2, '0')}`
  const p1Ids = Array.from({ length: 6 }, (_, i) => narrowId(i))
  const nonP1Ids = Array.from({ length: 9 }, (_, i) => narrowId(i + 6))
  const allNarrowIds = [...p1Ids, ...nonP1Ids]

  function renderNarrow() {
    const f = fixtureNarrow()
    const graph = filterGraph(f.nodes, f.edges, new Set(Object.keys(NODE_COLORS)), '')
    vi.useFakeTimers()
    render(<GraphExplorer graph={graph} />)
    clearFg()
    return graph
  }

  /** Focus tag:mass, wait for the inspector, expand its has_tag:in group. */
  function openNarrowInspector() {
    act(() => h.lastProps.onNodeClick(nodeOf('tag:mass')))
    advance(850)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    fireEvent.click(within(region).getByRole('button', { name: 'has tag ← 15' }))
    return region
  }

  it('(n1b) the open event carries facet_rows for groups over the open threshold', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    renderNarrow()
    act(() => h.lastProps.onNodeClick(nodeOf('tag:mass')))
    advance(850)
    const calls = info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] open')
    expect(calls).toHaveLength(1)
    // the static whole-group counts, cut to FACET_ROW_TOP_N per labelled row
    expect(calls[0][1].facet_rows).toEqual([
      {
        label: 'Type',
        total: 3,
        shown: 3,
        facets: [
          { key: 'entry_type:factual_reference', count: 7 },
          { key: 'entry_type:decision', count: 6 },
          { key: 'entry_type:lesson_learned', count: 2 },
        ],
      },
      {
        label: 'Project',
        total: 2,
        shown: 2,
        facets: [
          { key: 'project:p2', count: 9 },
          { key: 'project:p1', count: 6 },
        ],
      },
    ])
  })

  it('(c1) expanding the has_tag:in group shows the filter input and the pinned labelled facet rows', () => {
    renderNarrow()
    const region = openNarrowInspector()
    const input = within(region).getByTestId('group-filter-has_tag:in')
    expect(input).toBeInTheDocument()
    expect(within(region).getByTestId('group-filter-has_tag:in')).toHaveAccessibleName(
      'Filter has_tag by title',
    )
    expect(
      within(region)
        .queryAllByRole('button', { name: /\S.* \d+$/ })
        .filter((c) => c.getAttribute('aria-pressed') !== null)
        .map((c) => c.textContent),
    ).toEqual(['factual reference 7', 'decision 6', 'lesson learned 2', 'p2 9', 'p1 6'])
    // every member's only non-entry neighbour is tag:mass itself, excluded as
    // the inspected node — so there are no node-facet rows and no +N more
    expect(within(region).queryByRole('button', { name: /\+.* more$/ })).toBeNull()
  })

  it('(c2) hiding the p1 facet narrows the canvas, the rows and the header; Clear restores', () => {
    const graph = renderNarrow()
    const region = openNarrowInspector()
    for (const id of allNarrowIds) expect(visible(id)).toBe(true)
    // clicking 'p1 6' switches that facet OFF: the 6 p1 members hide
    fireEvent.click(within(region).getByRole('button', { name: 'p1 6' }))
    for (const id of nonP1Ids) expect(visible(id)).toBe(true)
    for (const id of p1Ids) expect(visible(id)).toBe(false)
    expect(visible('tag:mass')).toBe(true)
    for (const l of graph.links) {
      const linkVisible = h.lastProps.linkVisibility(l)
      if (p1Ids.includes(l.source)) expect(linkVisible).toBe(false)
      else expect(linkVisible).toBe(true)
    }
    expect(within(region).getByText('has tag ← 9 of 15')).toBeInTheDocument()
    expect(within(region).getAllByRole('button', { name: /Mass entry \d+/ })).toHaveLength(9)
    // Clear restores every member and its links
    fireEvent.click(within(region).getByTestId('group-clear-has_tag:in'))
    for (const id of allNarrowIds) expect(visible(id)).toBe(true)
    for (const l of graph.links) expect(h.lastProps.linkVisibility(l)).toBe(true)
    expect(within(region).queryByText(/^\d+ of \d+$/)).toBeNull()
    expect(within(region).getAllByRole('button', { name: /Mass entry \d+/ })).toHaveLength(15)
  })

  it('(c3) emptying the title text restores every member', () => {
    const graph = renderNarrow()
    const region = openNarrowInspector()
    const input = within(region).getByTestId('group-filter-has_tag:in') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'deep background' } })
    for (const id of allNarrowIds) if (id !== 'kb-00100') expect(visible(id)).toBe(false)
    expect(visible('kb-00100')).toBe(true)
    expect(within(region).getAllByRole('button', { name: /Mass entry \d+/ })).toHaveLength(1)
    fireEvent.change(input, { target: { value: '' } })
    for (const id of allNarrowIds) expect(visible(id)).toBe(true)
    for (const l of graph.links) expect(h.lastProps.linkVisibility(l)).toBe(true)
  })

  it('(c4) a facet click never walks: no camera calls and no re-open', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    renderNarrow()
    const region = openNarrowInspector()
    clearFg()
    fireEvent.click(within(region).getByRole('button', { name: 'decision 6' }))
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(
      info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] open'),
    ).toHaveLength(1)
  })

  it('(c5) Escape with focus in the filter input keeps the inspector and the focus', () => {
    renderNarrow()
    const region = openNarrowInspector()
    const input = within(region).getByTestId('group-filter-has_tag:in') as HTMLInputElement
    input.focus()
    clearFg()
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(screen.getByRole('region', { name: 'Node inspector' })).toBeInTheDocument()
    expect(fg.zoom).not.toHaveBeenCalled()
    expect(fg.centerAt).not.toHaveBeenCalled()
    expect(document.activeElement).toBe(input)
  })

  it('(c6) filtering fetches nothing and emits the pinned filter line per change', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    renderNarrow()
    const region = openNarrowInspector()
    const input = within(region).getByTestId('group-filter-has_tag:in') as HTMLInputElement
    fireEvent.click(within(region).getByRole('button', { name: 'p1 6' }))
    fireEvent.change(input, { target: { value: 'mass' } })
    fireEvent.click(within(region).getByTestId('group-clear-has_tag:in'))
    const filterCalls = info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] filter')
    expect(filterCalls).toHaveLength(3)
    expect(filterCalls[0][1]).toEqual({
      node_id: 'tag:mass',
      group_key: 'has_tag:in',
      text: '',
      hidden: ['project:p1'],
      matching: 9,
      total: 15,
      canvas_excluded: 6,
    })
    expect(filterCalls[1][1]).toMatchObject({
      text: 'mass',
      hidden: ['project:p1'],
      matching: 9,
      canvas_excluded: 6,
    })
    expect(filterCalls[2][1]).toMatchObject({
      text: '',
      hidden: [],
      matching: 15,
      total: 15,
      canvas_excluded: 0,
    })
    expect(
      info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] open'),
    ).toHaveLength(1)
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledTimes(0)
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(0)
  })
})

// ── History: the inspector view snapshot round trip (item B) ────────────────

describe('GraphExplorer — history view-state round trip', () => {
  function renderNarrowExplorer() {
    const f = fixtureNarrow()
    const graph = filterGraph(f.nodes, f.edges, new Set(Object.keys(NODE_COLORS)), '')
    vi.useFakeTimers()
    render(<GraphExplorer graph={graph} />)
    clearFg()
    return graph
  }

  it('Back restores the camera, the expanded group, the filter and the scroll', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    renderNarrowExplorer()
    act(() => h.lastProps.onNodeClick(nodeOf('tag:mass')))
    advance(850)
    const region = screen.getByRole('region', { name: 'Node inspector' })
    fireEvent.click(within(region).getByRole('button', { name: 'has tag ← 15' }))
    fireEvent.click(within(region).getByRole('button', { name: 'p1 6' }))
    expect(within(region).getByText('has tag ← 9 of 15')).toBeInTheDocument()
    // the same scroll container item A's scrollTop round-trip test uses
    // the same scroll container item A's scrollTop round-trip test uses
    const scroll = within(region).getByTestId('inspector-scroll')
    const filterCallsBeforeScroll = info.mock.calls.filter(
      (c: unknown[]) => c[0] === '[graph-inspector] filter',
    ).length
    act(() => {
      scroll.scrollTop = 120
    })
    fireEvent.scroll(scroll)
    // the live scroll position no longer flows through onInspectorViewChange:
    // scrolling emits no filter telemetry line
    expect(
      info.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] filter'),
    ).toHaveLength(filterCallsBeforeScroll)
    // walk to kb-00109 (a still-visible p2 member) through the shared select
    // path a row click takes
    fireEvent.click(within(region).getByRole('button', { name: /Mass entry 9/ }))
    clearFg()
    fireEvent.keyDown(document, { key: 'ArrowLeft' })
    // tag:mass's record kept the pre-move camera: a non-entry focus has no
    // camera shift, so (4, 40, 70)
    expect(fg.centerAt).toHaveBeenLastCalledWith(40, 70, 600)
    expect(fg.zoom).toHaveBeenLastCalledWith(4, 600)
    advance(650)
    const reopened = screen.getByRole('region', { name: 'Node inspector' })
    // the restored filter keeps the '<matching> of <total>' in the header
    expect(
      within(reopened).getByRole('button', { name: /has tag ← 9 of 15/ }),
    ).toHaveAttribute('aria-expanded', 'true')
    // aria-pressed means the facet is OFF/hidden
    expect(within(reopened).getByRole('button', { name: 'p1 6' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(within(reopened).getByText('has tag ← 9 of 15')).toBeInTheDocument()
    expect(within(reopened).getByTestId('inspector-scroll').scrollTop).toBe(120)
  })
})
