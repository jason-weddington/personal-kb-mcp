import { describe, it, expect, vi } from 'vitest'
import {
  ENTRY_DRAWER_W,
  FIT_PADDING,
  INSPECTOR_W,
  MAX_AUTO_ZOOM,
  MAX_ZOOM,
  boundingBox,
  fitView,
  EMPTY_TRAVERSAL,
  IDLE_LINK_STYLE,
  activeNeighborSet,
  explorerReducer,
  filterGraph,
  focusCenter,
  fpsFromFrameTimes,
  historyInvariantError,
  HISTORY_MAX,
  initialExplorerState,
  buildAdjacency,
  computeFocusSet,
  endpointId,
  isLinkVisible,
  isNodeVisible,
  labelOpacity,
  linkStyle,
  nodeRadius,
  pulseRadius,
  rankSearch,
  truncateLabel,
  zoomClampTarget,
  NODE_COLORS,
} from '../graph/explorerModel'
import type {
  ExplorerNode,
  ExplorerState,
  HistoryRecord,
  NavSnapshot,
  TraversalState,
} from '../graph/explorerModel'
import { DEFAULT_INSPECTOR_VIEW } from '../components/GraphNodeInspector'
import type { InspectorViewState } from '../components/GraphNodeInspector'

const base = {
  hovered: false,
  highlighted: false,
  nodeState: undefined,
  queryMode: null,
  inFocus: false,
  solo: false,
  globalScale: 1,
} as const

function node(id: string, type: string, extra: Partial<ExplorerNode> = {}): ExplorerNode {
  return { id, type, label: id, val: 1, properties: {}, ...extra }
}

describe('labelOpacity', () => {
  it('ramps 3.0 to 5.0 outside solo', () => {
    expect(labelOpacity({ ...base, globalScale: 2.0 })).toBe(0)
    expect(labelOpacity({ ...base, globalScale: 3.0 })).toBe(0)
    expect(labelOpacity({ ...base, globalScale: 4.0 })).toBe(0.5)
    expect(labelOpacity({ ...base, globalScale: 5.0 })).toBe(1)
    expect(labelOpacity({ ...base, globalScale: 8.0 })).toBe(1)
  })
  it('ramps 1.5 to 3.0 in solo', () => {
    expect(labelOpacity({ ...base, solo: true, globalScale: 1.5 })).toBe(0)
    expect(labelOpacity({ ...base, solo: true, globalScale: 2.25 })).toBe(0.5)
    expect(labelOpacity({ ...base, solo: true, globalScale: 3.0 })).toBe(1)
  })
  it('applies overrides in order', () => {
    expect(labelOpacity({ ...base, hovered: true, globalScale: 0.5 })).toBe(1)
    expect(labelOpacity({ ...base, highlighted: true, globalScale: 0.5 })).toBe(1)
    expect(labelOpacity({ ...base, nodeState: 'visited', globalScale: 0.5 })).toBe(1)
    expect(labelOpacity({ ...base, queryMode: 'explore', globalScale: 10 })).toBe(0)
    expect(labelOpacity({ ...base, queryMode: 'pending', globalScale: 10 })).toBe(0)
    expect(labelOpacity({ ...base, inFocus: true, globalScale: 0.5 })).toBe(1)
  })
})

describe('truncateLabel', () => {
  it('truncates long labels below scale 2', () => {
    expect(truncateLabel('abcdefghijklmnopqrstuvwxyz', 1.9)).toBe('abcdefghijklmnopqrs…')
    expect(truncateLabel('abcdefghijklmnopqrstuvwxyz', 2.0)).toBe('abcdefghijklmnopqrstuvwxyz')
    const s20 = 'a'.repeat(20)
    expect(truncateLabel(s20, 1.0)).toBe(s20)
    const s21 = 'abcdefghijklmnopqrstu'
    expect(truncateLabel(s21, 1.0)).toBe(s21.slice(0, 19) + '…')
  })
})

describe('nodeRadius and pulseRadius', () => {
  it('nodeRadius', () => {
    expect(nodeRadius(4, 0, 0)).toBe(5)
    expect(nodeRadius(0, 0, 0)).toBe(2.5)
    expect(nodeRadius(4, 6, 0)).toBe(11)
    expect(nodeRadius(4, 3, 0)).toBe(8)
    expect(nodeRadius(9, 0, 3)).toBe(10.5)
  })
  it('pulseRadius', () => {
    expect(pulseRadius(undefined, 5)).toBe(0)
    expect(pulseRadius(1000, 1000)).toBe(6)
    expect(pulseRadius(1000, 1300)).toBe(3)
    expect(pulseRadius(1000, 1600)).toBe(0)
    expect(pulseRadius(1000, 1601)).toBe(0)
  })
})

describe('endpoints and adjacency', () => {
  it('endpointId', () => {
    expect(endpointId('a')).toBe('a')
    expect(endpointId({ id: 'a' })).toBe('a')
    expect(endpointId(5)).toBe('5')
    expect(endpointId({})).toBe('')
  })
  it('buildAdjacency with object and string endpoints', () => {
    for (const links of [
      [{ source: { id: 'a' }, target: { id: 'b' }, type: 't' }],
      [{ source: 'a', target: 'b', type: 't' }],
    ]) {
      const adj = buildAdjacency(links)
      expect(adj.get('a')).toEqual([{ node: 'b', type: 't', dir: 'out' }])
      expect(adj.get('b')).toEqual([{ node: 'a', type: 't', dir: 'in' }])
    }
  })
  it('computeFocusSet', () => {
    const adj = buildAdjacency([
      { source: 'a', target: 'b', type: 't' },
      { source: 'c', target: 'a', type: 't' },
    ])
    expect(computeFocusSet(null, adj)).toBeNull()
    expect(computeFocusSet('a', adj)).toEqual(new Set(['a', 'b', 'c']))
    expect(computeFocusSet('z', adj)).toEqual(new Set(['z']))
  })
  it('isNodeVisible: focus beats solo', () => {
    const n = { id: 'a', type: 'entry' }
    expect(isNodeVisible(n, null, null)).toBe(true)
    expect(isNodeVisible(n, null, 'tag')).toBe(false)
    expect(isNodeVisible(n, null, 'entry')).toBe(true)
    expect(isNodeVisible(n, new Set(['a']), 'tag')).toBe(true)
    expect(isNodeVisible(n, new Set(['b']), null)).toBe(false)
  })
  it('isLinkVisible', () => {
    expect(isLinkVisible('a', 'b', null, null)).toBe(true)
    expect(isLinkVisible('a', 'b', null, 'tag')).toBe(false)
    expect(isLinkVisible('a', 'b', 'a', 'tag')).toBe(true)
    expect(isLinkVisible('a', 'b', 'b', null)).toBe(true)
    expect(isLinkVisible('a', 'b', 'c', null)).toBe(false)
  })
  it('activeNeighborSet', () => {
    const adj = buildAdjacency([
      { source: 'a', target: 'b', type: 't' },
      { source: 'b', target: 'c', type: 't' },
    ])
    expect(activeNeighborSet(new Map([['a', 'result']]), adj)).toEqual(new Set(['b']))
  })
})

describe('linkStyle', () => {
  it('returns the frozen idle style when not querying', () => {
    const a = linkStyle({ source: 'a', target: 'b' }, EMPTY_TRAVERSAL)
    const b = linkStyle({ source: 'c', target: 'd' }, EMPTY_TRAVERSAL)
    expect(a).toBe(IDLE_LINK_STYLE)
    expect(b).toBe(IDLE_LINK_STYLE)
  })
  it('styles active links in query mode', () => {
    const t: TraversalState = {
      queryMode: 'explore',
      nodeStates: new Map([
        ['a', 'result'],
        ['b', 'visited'],
      ]),
      pulseStarts: new Map(),
    }
    expect(linkStyle({ source: { id: 'a' }, target: 'b' }, t)).toEqual({
      color: 'rgba(0,255,136,0.5)',
      width: 1.5,
      particleWidth: 2.5,
      particleColor: 'rgba(0,255,136,0.6)',
      particleSpeed: 0.006,
    })
    expect(linkStyle({ source: 'a', target: 'z' }, t)).toEqual({
      color: 'rgba(0,255,136,0.25)',
      width: 1.5,
      particleWidth: 2.5,
      particleColor: 'rgba(0,255,136,0.6)',
      particleSpeed: 0.006,
    })
    expect(linkStyle({ source: 'y', target: 'z' }, t)).toEqual({
      color: 'rgba(255,255,255,0.03)',
      width: 0.3,
      particleWidth: 0,
      particleColor: 'rgba(0,0,0,0)',
      particleSpeed: 0.003,
    })
  })
})

describe('rankSearch', () => {
  it('ranks exact label first, then val desc', () => {
    const nodes = [
      node('kb-1', 'entry', { label: 'Alpha decision', val: 9 }),
      node('tag:alpha', 'tag', { label: 'alpha', val: 1 }),
      node('kb-2', 'entry', { label: 'alpha beta', val: 5 }),
    ]
    expect(rankSearch(nodes, 'ALPHA').map((n) => n.id)).toEqual(['tag:alpha', 'kb-1', 'kb-2'])
  })
  it('caps at 12 and ignores blank queries', () => {
    const nodes = Array.from({ length: 20 }, (_, i) => node(`n${i}`, 'tag', { label: `match ${i}` }))
    expect(rankSearch(nodes, 'match')).toHaveLength(12)
    expect(rankSearch(nodes, '   ')).toEqual([])
  })
  it('matches id, long_title, tags, entry_type and project_ref', () => {
    const n = node('kb-00042', 'entry', {
      label: 'x',
      properties: {
        long_title: 'Long Title Here',
        tags: 'sqlite,vec',
        entry_type: 'lesson_learned',
        project_ref: 'proj-zeta',
      },
    })
    for (const q of ['kb-00042', 'title here', 'SQLITE', 'lesson_learned', 'zeta']) {
      expect(rankSearch([n], q)).toEqual([n])
    }
    expect(rankSearch([n], 'nope')).toEqual([])
  })
})

describe('camera helpers', () => {
  it('zoomClampTarget', () => {
    expect(zoomClampTarget(6)).toBeNull()
    expect(zoomClampTarget(6.01)).toBe(6)
    expect(zoomClampTarget(12)).toBe(6)
    expect(zoomClampTarget(1)).toBeNull()
  })
  it('focusCenter', () => {
    const n = { x: 10, y: 20 }
    expect(focusCenter(n, true, 1200)).toEqual({ x: 65, y: 20 })
    expect(focusCenter(n, true, 959)).toEqual({ x: 10, y: 20 })
    expect(focusCenter(n, false, 1200)).toEqual({ x: 10, y: 20 })
  })
  it('focusCenter with a right inset', () => {
    const n = { x: 10, y: 20 }
    for (const w of [1200, 780, 959, 540]) {
      expect(focusCenter(n, true, w, 420)).toEqual({ x: 12.5, y: 20 })
    }
    expect(focusCenter(n, true, 539, 420)).toEqual({ x: 10, y: 20 })
    expect(focusCenter(n, true, 1200, 640)).toEqual({ x: 10, y: 20 })
    expect(focusCenter(n, true, 1200, 800)).toEqual({ x: 10, y: 20 })
    expect(focusCenter(n, false, 1200, 420)).toEqual({ x: 10, y: 20 })
    expect(focusCenter(n, true, 1200)).toEqual({ x: 65, y: 20 })
  })
  it('entry camera shift constants', () => {
    // the 440 px inspector drives the shift; the 640 px drawer stays for the
    // off-graph drawer
    expect(INSPECTOR_W).toBe(440)
    expect(ENTRY_DRAWER_W).toBe(640)
    expect((INSPECTOR_W - 0) / 2 / 4).toBe(55)
    expect((INSPECTOR_W - 420) / 2 / 4).toBe(2.5)
  })
  it('fpsFromFrameTimes', () => {
    const now = 10_000
    const times = Array.from({ length: 60 }, (_, i) => now - i * (1000 / 60))
    expect(fpsFromFrameTimes(times, now)).toBe(60)
    expect(fpsFromFrameTimes([], now)).toBe(0)
    expect(fpsFromFrameTimes([now - 2000, now - 1000], now)).toBe(0)
    const half = Array.from({ length: 30 }, (_, i) => now - i * 10)
    expect(fpsFromFrameTimes(half, now, 500)).toBe(60)
  })
})

describe('filterGraph', () => {
  const all = new Set(Object.keys(NODE_COLORS))
  const fixture = () => {
    const nodes: ExplorerNode[] = [
      node('kb-1', 'entry', { properties: { project_ref: 'p1' } }),
      node('kb-2', 'entry', { properties: { project_ref: 'p2' } }),
      node('tag:a', 'tag'),
      node('tag:b', 'tag'),
      node('concept:c', 'concept'),
      node('x:1', 'mystery'),
      node('x:2', 'mystery'),
    ]
    const edges = [
      { source: 'kb-1', target: 'tag:a', type: 'has_tag' },
      { source: 'kb-2', target: 'tag:b', type: 'has_tag' },
      { source: 'tag:a', target: 'concept:c', type: 'about' },
      { source: 'kb-2', target: 'x:1', type: 'odd' },
      { source: 'kb-1', target: 'kb-2', type: 'related_to' },
    ]
    return { nodes, edges }
  }

  it('returns everything as new string links for all types and scope All', () => {
    const { nodes, edges } = fixture()
    const g = filterGraph(nodes, edges, all, '')
    expect(g.nodes).toHaveLength(nodes.length)
    expect(g.links).toHaveLength(edges.length)
    g.links.forEach((l, i) => {
      expect(l).not.toBe(edges[i])
      expect(l).toEqual(edges[i])
    })
    g.nodes.forEach((n, i) => expect(n).toBe(nodes[i]))
  })

  it('drops tag nodes and links when the tag chip is off, keeping unknown types', () => {
    const { nodes, edges } = fixture()
    const types = new Set([...all].filter((t) => t !== 'tag'))
    const g = filterGraph(nodes, edges, types, '')
    expect(g.nodes.map((n) => n.id)).toEqual(['kb-1', 'kb-2', 'concept:c', 'x:1', 'x:2'])
    expect(g.links.map((l) => l.type)).toEqual(['odd', 'related_to'])
  })

  it('scopes to a project plus one-hop non-entry neighbours', () => {
    const { nodes, edges } = fixture()
    const g = filterGraph(nodes, edges, all, 'p1')
    expect(g.nodes.map((n) => n.id)).toEqual(['kb-1', 'tag:a', 'x:1', 'x:2'])
    expect(g.links).toEqual([{ source: 'kb-1', target: 'tag:a', type: 'has_tag' }])
  })

  it('does not re-add deactivated types under a scope', () => {
    const { nodes, edges } = fixture()
    const types = new Set([...all].filter((t) => t !== 'tag'))
    const g = filterGraph(nodes, edges, types, 'p1')
    expect(g.nodes.map((n) => n.id)).toEqual(['kb-1', 'x:1', 'x:2'])
  })

  it('reads mutated object endpoints and warns once', () => {
    const { nodes, edges } = fixture()
    const byId = new Map(nodes.map((n) => [n.id, n]))
    const mutated = edges.map((e) => ({
      source: byId.get(e.source)!,
      target: byId.get(e.target)!,
      type: e.type,
    }))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const g = filterGraph(nodes, mutated, all, '')
    expect(g.links).toHaveLength(filterGraph(nodes, edges, all, '').links.length)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(typeof g.links[0].source).toBe('string')
    warn.mockRestore()
  })
})


// ── Navigation reducer: one cursor-based browser history ────────────────────

describe('explorerReducer', () => {
  const view = (i: number): InspectorViewState => ({
    expanded: { [`g${i}`]: true },
    filters: { [`g${i}`]: { text: `t${i}`, facets: [`f${i}`] } },
    scrollTop: i * 10,
  })
  const v0 = view(0)
  const v1 = view(1)
  const v2 = view(2)
  const v4 = view(4)
  const v5 = view(5)
  const v6 = view(6)
  const snap = (zoom: number, cx: number, cy: number, v: InspectorViewState): NavSnapshot => ({
    zoom,
    cx,
    cy,
    view: v,
  })
  const rec = (
    nodeId: string,
    zoom: number,
    cx: number,
    cy: number,
    v: InspectorViewState,
  ): HistoryRecord => ({ nodeId, zoom, cx, cy, view: v })
  const D = DEFAULT_INSPECTOR_VIEW
  const walked = (): ExplorerState => ({
    focusedId: 'c',
    highlightedId: 'c',
    soloType: null,
    history: [
      rec('a', 4, 10, 20, v1),
      rec('b', 4, 30, 40, v2),
      rec('c', 4, 30, 40, D),
    ],
    cursor: 2,
  })

  it('select rewrites the record at the cursor, truncates forward and appends', () => {
    let s = explorerReducer(initialExplorerState, {
      type: 'select',
      nodeId: 'a',
      snapshot: snap(1, 0, 0, v0),
    })
    s = explorerReducer(s, { type: 'select', nodeId: 'b', snapshot: snap(4, 10, 20, v1) })
    s = explorerReducer(s, { type: 'select', nodeId: 'c', snapshot: snap(4, 30, 40, v2) })
    expect(s.focusedId).toBe('c')
    expect(s.highlightedId).toBe('c')
    expect(s.cursor).toBe(2)
    expect(s.history).toHaveLength(3)
    // a's record was rewritten by the select of b (the camera from before the move)
    expect(s.history[0]).toEqual(rec('a', 4, 10, 20, v1))
    // c's record carries the select-c snapshot camera and the default view
    expect(s.history[2]).toEqual(rec('c', 4, 30, 40, D))
    expect(historyInvariantError(s)).toBeNull()
  })

  it('back and forward rewrite the outgoing record and move the cursor', () => {
    const s = walked()
    const afterBack = explorerReducer(s, { type: 'back', snapshot: snap(1, 50, 50, v4) })
    expect(afterBack.history[2]).toEqual(rec('c', 1, 50, 50, v4))
    expect(afterBack.cursor).toBe(1)
    expect(afterBack.focusedId).toBe('b')
    expect(afterBack.highlightedId).toBe('b')
    expect(historyInvariantError(afterBack)).toBeNull()

    const afterForward = explorerReducer(afterBack, { type: 'forward', snapshot: snap(2, 60, 60, v5) })
    expect(afterForward.history[1]).toEqual(rec('b', 2, 60, 60, v5))
    expect(afterForward.cursor).toBe(2)
    expect(afterForward.focusedId).toBe('c')
    expect(historyInvariantError(afterForward)).toBeNull()

    // from the post-back state, a select truncates everything forward of the cursor
    const branched = explorerReducer(afterBack, {
      type: 'select',
      nodeId: 'd',
      snapshot: snap(5, 7, 9, v6),
    })
    expect(branched.history[1]).toEqual(rec('b', 5, 7, 9, v6))
    expect(branched.history.map((r) => r.nodeId)).toEqual(['a', 'b', 'd'])
    expect(branched.cursor).toBe(2)
    expect(branched.focusedId).toBe('d')
    expect(historyInvariantError(branched)).toBeNull()
  })

  it('caps history at HISTORY_MAX records, dropping the oldest', () => {
    expect(HISTORY_MAX).toBe(500)
    let s = initialExplorerState
    for (let i = 0; i < HISTORY_MAX; i++) {
      s = explorerReducer(s, {
        type: 'select',
        nodeId: `n${i}`,
        snapshot: snap(1, i, 0, D),
      })
    }
    expect(s.history).toHaveLength(HISTORY_MAX)
    expect(s.cursor).toBe(HISTORY_MAX - 1)
    s = explorerReducer(s, { type: 'select', nodeId: 'extra', snapshot: snap(1, 500, 0, D) })
    expect(s.history).toHaveLength(HISTORY_MAX)
    expect(s.history[0].nodeId).toBe('n1')
    expect(s.cursor).toBe(HISTORY_MAX - 1)
    expect(historyInvariantError(s)).toBeNull()
  })

  it('unfocus keeps history and cursor; a snapshot rewrites the record at the cursor', () => {
    const s = walked()
    const u = explorerReducer(s, { type: 'unfocus' })
    expect(u.focusedId).toBeNull()
    expect(u.highlightedId).toBeNull()
    expect(u.history).toEqual(s.history)
    expect(u.cursor).toBe(2)
    const withSnap = explorerReducer(s, { type: 'unfocus', snapshot: snap(9, 8, 7, v1) })
    expect(withSnap.focusedId).toBeNull()
    expect(withSnap.highlightedId).toBeNull()
    expect(withSnap.history[2]).toEqual(rec('c', 9, 8, 7, v1))
    expect(withSnap.history[0]).toEqual(rec('a', 4, 10, 20, v1))
    expect(withSnap.cursor).toBe(2)
  })

  it('toggleSolo and clearSolo keep history and cursor', () => {
    const s: ExplorerState = { ...walked(), soloType: 'tag' }
    expect(explorerReducer(s, { type: 'toggleSolo', nodeType: 'tag' })).toEqual({
      focusedId: null,
      highlightedId: null,
      soloType: null,
      history: s.history,
      cursor: 2,
    })
    expect(explorerReducer(s, { type: 'toggleSolo', nodeType: 'entry' }).soloType).toBe('entry')
    const solo = explorerReducer(s, { type: 'toggleSolo', nodeType: 'entry' })
    const cleared = explorerReducer(solo, { type: 'clearSolo' })
    expect(cleared.soloType).toBeNull()
    expect(cleared.history).toEqual(s.history)
    expect(cleared.cursor).toBe(2)
  })

  it('clearAll clears focus, highlight and solo but keeps history and cursor', () => {
    const s: ExplorerState = { ...walked(), soloType: 'tag' }
    const out = explorerReducer(s, { type: 'clearAll' })
    expect(out.focusedId).toBeNull()
    expect(out.highlightedId).toBeNull()
    expect(out.soloType).toBeNull()
    expect(out.history).toEqual(s.history)
    expect(out.cursor).toBe(2)
    const withSnap = explorerReducer(s, { type: 'clearAll', snapshot: snap(7, 1, 2, v2) })
    expect(withSnap.focusedId).toBeNull()
    expect(withSnap.soloType).toBeNull()
    expect(withSnap.history[2]).toEqual(rec('c', 7, 1, 2, v2))
    expect(withSnap.cursor).toBe(2)
  })

  it('back while unfocused re-focuses the record at the cursor', () => {
    const s: ExplorerState = {
      focusedId: null,
      highlightedId: null,
      soloType: null,
      history: [rec('a', 1, 0, 0, D), rec('b', 2, 1, 3, D)],
      cursor: 1,
    }
    const action = { type: 'back' as const, snapshot: snap(3, 4, 5, v0) }
    const out = explorerReducer(s, action)
    expect(out.focusedId).toBe('b')
    expect(out.highlightedId).toBe('b')
    expect(out.cursor).toBe(1)
    // the record at the cursor is NOT rewritten by the unfocused re-focus
    expect(out.history[1]).toEqual(rec('b', 2, 1, 3, D))
    // an identical dispatch against the same unfocused state lands the same
    // result: the re-focus is idempotent while nothing is focused
    expect(explorerReducer(s, action)).toEqual(out)
    expect(historyInvariantError(out)).toBeNull()
  })

  it('forward while unfocused moves the cursor and re-focuses', () => {
    const s: ExplorerState = {
      focusedId: null,
      highlightedId: null,
      soloType: null,
      history: [rec('a', 1, 0, 0, D), rec('b', 2, 1, 3, D)],
      cursor: 0,
    }
    const out = explorerReducer(s, { type: 'forward', snapshot: snap(3, 4, 5, v0) })
    expect(out.focusedId).toBe('b')
    expect(out.highlightedId).toBe('b')
    expect(out.cursor).toBe(1)
    expect(historyInvariantError(out)).toBeNull()
  })

  it('boundary and empty-history walks return the same state object', () => {
    const one: ExplorerState = {
      focusedId: 'a',
      highlightedId: 'a',
      soloType: null,
      history: [rec('a', 1, 0, 0, D)],
      cursor: 0,
    }
    expect(explorerReducer(one, { type: 'back', snapshot: snap(2, 1, 1, v0) })).toBe(one)
    const two: ExplorerState = {
      focusedId: 'b',
      highlightedId: 'b',
      soloType: null,
      history: [rec('a', 1, 0, 0, D), rec('b', 2, 1, 3, D)],
      cursor: 1,
    }
    expect(explorerReducer(two, { type: 'forward', snapshot: snap(2, 1, 1, v0) })).toBe(two)
    // forward while unfocused at the last record is also a no-op
    const atEnd: ExplorerState = { ...two, focusedId: null, highlightedId: null }
    expect(explorerReducer(atEnd, { type: 'forward', snapshot: snap(2, 1, 1, v0) })).toBe(atEnd)
    expect(explorerReducer(initialExplorerState, { type: 'back', snapshot: snap(2, 1, 1, v0) })).toBe(
      initialExplorerState,
    )
    expect(
      explorerReducer(initialExplorerState, { type: 'forward', snapshot: snap(2, 1, 1, v0) }),
    ).toBe(initialExplorerState)
  })

  it('prune keeps the cursor pointing at the same logical record', () => {
    const types = new Set(['entry', 'tag'])
    // focused node removed: the whole history clears (the one in-page clear)
    const focused: ExplorerState = {
      focusedId: 'b',
      highlightedId: 'b',
      soloType: 'tag',
      history: [rec('a', 1, 0, 0, D), rec('b', 1, 0, 0, D)],
      cursor: 1,
    }
    expect(
      explorerReducer(focused, { type: 'prune', renderedIds: new Set(['a']), renderedTypes: types }),
    ).toEqual({
      focusedId: null,
      highlightedId: null,
      soloType: 'tag',
      history: [],
      cursor: 0,
    })
    // records dropped while the record at the cursor survives: cursor shifts
    const s: ExplorerState = {
      focusedId: 'b',
      highlightedId: 'b',
      soloType: null,
      history: [rec('a', 1, 0, 0, D), rec('b', 1, 0, 0, D), rec('c', 1, 0, 0, D)],
      cursor: 1,
    }
    const shifted = explorerReducer(s, {
      type: 'prune',
      renderedIds: new Set(['b', 'c']),
      renderedTypes: types,
    })
    expect(shifted.history.map((r) => r.nodeId)).toEqual(['b', 'c'])
    expect(shifted.cursor).toBe(0)
    expect(shifted.focusedId).toBe('b')
    expect(historyInvariantError(shifted)).toBeNull()
    // record at the cursor dropped while unfocused: cursor clamps
    const u: ExplorerState = {
      focusedId: null,
      highlightedId: null,
      soloType: null,
      history: [rec('a', 1, 0, 0, D), rec('b', 1, 0, 0, D)],
      cursor: 1,
    }
    const clamped = explorerReducer(u, {
      type: 'prune',
      renderedIds: new Set(['a']),
      renderedTypes: types,
    })
    expect(clamped.history.map((r) => r.nodeId)).toEqual(['a'])
    expect(clamped.cursor).toBe(0)
    expect(clamped.focusedId).toBeNull()
    expect(historyInvariantError(clamped)).toBeNull()
    // solo type removed
    expect(
      explorerReducer(focused, {
        type: 'prune',
        renderedIds: new Set(['a', 'b']),
        renderedTypes: new Set(['entry']),
      }).soloType,
    ).toBeNull()
    // highlighted but unfocused node removed
    const hl: ExplorerState = { ...initialExplorerState, highlightedId: 'z' }
    expect(
      explorerReducer(hl, { type: 'prune', renderedIds: new Set(), renderedTypes: types })
        .highlightedId,
    ).toBeNull()
    // nothing changes -> same object
    expect(
      explorerReducer(s, {
        type: 'prune',
        renderedIds: new Set(['a', 'b', 'c']),
        renderedTypes: types,
      }),
    ).toBe(s)
  })

  it('historyInvariantError reports a corrupted cursor or an unrendered record', () => {
    expect(historyInvariantError(initialExplorerState)).toBeNull()
    const ok: ExplorerState = {
      focusedId: 'b',
      highlightedId: 'b',
      soloType: null,
      history: [rec('a', 1, 0, 0, D), rec('b', 2, 1, 3, D)],
      cursor: 1,
    }
    expect(historyInvariantError(ok)).toBeNull()
    expect(historyInvariantError(ok, { renderedIds: new Set(['a', 'b']) })).toBeNull()
    // cursor out of range
    expect(historyInvariantError({ ...ok, cursor: 2 })).toMatch(/cursor/)
    expect(historyInvariantError({ ...ok, cursor: -1 })).toMatch(/cursor/)
    // empty history with a nonzero cursor
    expect(
      historyInvariantError({ ...initialExplorerState, history: [], cursor: 1 }),
    ).toMatch(/cursor/)
    // a history record whose node is not rendered
    expect(historyInvariantError(ok, { renderedIds: new Set(['a']) })).toMatch(/not rendered/)
    // the record at the cursor must exist
    expect(historyInvariantError({ ...ok, history: [] })).toMatch(/cursor/)
  })
})


describe('fitView', () => {
  const caps = { maxZoom: MAX_ZOOM, maxAutoZoom: MAX_AUTO_ZOOM }
  const sq = { minX: -500, minY: -500, maxX: 500, maxY: 500 }

  it('keeps a square graph at least 70% of a 390x844 canvas width with a 160 px inset', () => {
    const v = fitView(sq, { width: 390, height: 844 }, 160, FIT_PADDING, caps)
    expect(1000 * v.zoom).toBeGreaterThanOrEqual(0.7 * 390)
  })

  it('at 1440x900 beats the four-sided padding fit', () => {
    const inset = 160
    const v = fitView(sq, { width: 1440, height: 900 }, inset, FIT_PADDING, caps)
    const old = Math.min(
      (1440 - 2 * (FIT_PADDING + inset)) / 1000,
      (900 - 2 * (FIT_PADDING + inset)) / 1000,
    )
    expect(v.zoom).toBeGreaterThan(old)
    expect(v.zoom).toBeCloseTo((900 - inset - 2 * FIT_PADDING) / 1000)
  })

  it('with no inset matches zoomToFit with FIT_PADDING, centred on the box', () => {
    const v = fitView(sq, { width: 1440, height: 900 }, 0, FIT_PADDING, caps)
    expect(v.zoom).toBeCloseTo((900 - 2 * FIT_PADDING) / 1000)
    expect(v.center).toEqual({ x: 0, y: 0 })
  })

  it('shifts the camera up by half the inset so the box sits below it', () => {
    const v = fitView(sq, { width: 1440, height: 900 }, 100, FIT_PADDING, caps)
    expect(v.center.y).toBeCloseTo(-50 / v.zoom)
  })

  it('caps the zoom for a single point', () => {
    const v = fitView(
      { minX: 5, minY: 5, maxX: 5, maxY: 5 },
      { width: 800, height: 600 },
      0,
      FIT_PADDING,
      caps,
    )
    expect(v.zoom).toBe(MAX_AUTO_ZOOM)
  })

  it('boundingBox skips nodes without a position', () => {
    expect(boundingBox([{ x: 1, y: 2 }, {}, { x: 3, y: -4 }])).toEqual({
      minX: 1,
      minY: -4,
      maxX: 3,
      maxY: 2,
    })
    expect(boundingBox([{}])).toBeNull()
  })
})
