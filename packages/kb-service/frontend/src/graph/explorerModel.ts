// Pure model for the Graph page explorer: constants, helpers, filtering and
// the navigation reducer. Ported from the tuned standalone explorer
// (personal_kb 47afec7 src/personal_kb/explorer/static/explorer.js).
import type { GraphFullNode } from '../kbTypes'
import { DEFAULT_INSPECTOR_VIEW } from '../components/GraphNodeInspector'
import type { InspectorViewState } from '../components/GraphNodeInspector'

// ── Constants ────────────────────────────────────────────────────────────────

/** Node type colours. Key order is the chip and legend order. */
export const NODE_COLORS = {
  entry: '#e0e0e0',
  tag: '#00bcd4',
  project: '#ff9800',
  person: '#ffc107',
  tool: '#4caf50',
  concept: '#9c27b0',
  technology: '#2196f3',
  note: '#78909c',
} as const satisfies Record<string, string>

export const NODE_TYPES = Object.keys(NODE_COLORS) as Array<keyof typeof NODE_COLORS>

export const FALLBACK_COLOR = '#666'
export const BACKGROUND = '#0a0a0f'
export const RESULT_COLOR = '#00ff88'
export const VISITED_COLOR = '#ffaa00'
export const DIMMED_FILL = '#555'
export const DIM_ALPHA = 0.15
export const LABEL_ZOOM_MIN = 3.0
export const LABEL_ZOOM_MAX = 5.0
export const LABEL_TRUNCATE_SCALE = 2.0
export const LABEL_TRUNCATE_LEN = 20
export const LABEL_MIN_OPACITY = 0.01
export const NODE_RADIUS_SCALE = 2.5
export const RESULT_EXTRA_R = 3
export const HIGHLIGHT_EXTRA_R = 6
export const PULSE_DURATION_MS = 600
export const PULSE_MAX_R = 6
export const GLOW_RESULT = 30
export const GLOW_VISITED = 20
export const GLOW_HIGHLIGHT = 25
export const MAX_ZOOM = 12
export const MAX_AUTO_ZOOM = 6
export const FOCUS_ZOOM = 4
export const FOCUS_MS = 800
export const DRAWER_DELAY_MS = 850
export const BACKTRACK_MS = 600
export const BACKTRACK_DRAWER_DELAY_MS = 650
export const FIT_MS = 400
export const FIT_PADDING = 40
export const CLAMP_MS = 300
/** The zoom clamp runs at fit duration + this, so 450 ms after a 400 ms fit. */
export const CLAMP_AFTER_FIT_MS = 50
export const RESET_FIT_DELAY_MS = 100
export const INITIAL_FIT_DELAY_MS = 1500
export const SEARCH_LIMIT = 12
export const LINK_PARTICLES = 2
export const ENTRY_DRAWER_W = 640
export const ENTRY_OFFSET_MIN_WIDTH = 960
export const INSPECTOR_W = 440
export const PERF_WINDOW_MS = 1000
export const PERF_TICK_MS = 500

export function nodeColor(type: string): string {
  return (NODE_COLORS as Record<string, string>)[type] ?? FALLBACK_COLOR
}

export function isKnownType(type: string): boolean {
  return Object.prototype.hasOwnProperty.call(NODE_COLORS, type)
}

// ── Types ────────────────────────────────────────────────────────────────────

export type ExplorerNode = GraphFullNode & { x?: number; y?: number }
export type ExplorerLink = { source: string; target: string; type: string }
export type LinkEnd = string | number | { id?: string | number }
export type Neighbor = { node: string; type: string; dir: 'out' | 'in' }
export type Adjacency = Map<string, Neighbor[]>

/** Traversal contract (rendered here, populated by query mode). */
export type QueryMode = 'pending' | 'explore' | 'summarize' | null
export type NodeState = 'visited' | 'result'
export interface TraversalState {
  queryMode: QueryMode
  nodeStates: ReadonlyMap<string, NodeState>
  /** performance.now() timestamps. */
  pulseStarts: ReadonlyMap<string, number>
}
export const EMPTY_TRAVERSAL: TraversalState = {
  queryMode: null,
  nodeStates: new Map(),
  pulseStarts: new Map(),
}

export type LinkStyle = {
  color: string
  width: number
  particleWidth: number
  particleColor: string
  particleSpeed: number
}

export const IDLE_LINK_STYLE: Readonly<LinkStyle> = Object.freeze({
  color: 'rgba(255,255,255,0.08)',
  width: 0.5,
  particleWidth: 1.5,
  particleColor: 'rgba(255,255,255,0.3)',
  particleSpeed: 0.003,
})

// ── Pure helpers ─────────────────────────────────────────────────────────────

/** Link endpoint id, whether or not force-graph replaced it with a node. */
export function endpointId(end: LinkEnd): string {
  if (typeof end === 'string') return end
  if (typeof end === 'number') return String(end)
  if (end && end.id !== undefined && end.id !== null) return String(end.id)
  return ''
}

export function buildAdjacency(
  links: ReadonlyArray<{ source: LinkEnd; target: LinkEnd; type: string }>,
): Adjacency {
  const adj: Adjacency = new Map()
  for (const l of links) {
    const s = endpointId(l.source)
    const t = endpointId(l.target)
    let sl = adj.get(s)
    if (!sl) {
      sl = []
      adj.set(s, sl)
    }
    let tl = adj.get(t)
    if (!tl) {
      tl = []
      adj.set(t, tl)
    }
    sl.push({ node: t, type: l.type, dir: 'out' })
    tl.push({ node: s, type: l.type, dir: 'in' })
  }
  return adj
}

export function computeFocusSet(
  focusedId: string | null,
  adjacency: ReadonlyMap<string, Neighbor[]>,
): Set<string> | null {
  if (focusedId === null) return null
  const set = new Set<string>([focusedId])
  for (const n of adjacency.get(focusedId) ?? []) set.add(n.node)
  return set
}

export function isNodeVisible(node: { id: string; type: string }, focusSet: Set<string> | null): boolean {
  if (focusSet !== null) return focusSet.has(node.id)
  return true
}

export function isLinkVisible(
  sourceId: string,
  targetId: string,
  focusedId: string | null,
): boolean {
  if (focusedId !== null) return sourceId === focusedId || targetId === focusedId
  return true
}

export function labelOpacity(a: {
  hovered: boolean
  highlighted: boolean
  nodeState: NodeState | undefined
  queryMode: QueryMode
  inFocus: boolean
  globalScale: number
}): number {
  if (a.hovered) return 1
  if (a.highlighted) return 1
  if (a.nodeState !== undefined) return 1
  if (a.queryMode !== null) return 0
  if (a.inFocus) return 1
  return Math.min(Math.max((a.globalScale - LABEL_ZOOM_MIN) / (LABEL_ZOOM_MAX - LABEL_ZOOM_MIN), 0), 1)
}

export function truncateLabel(text: string, globalScale: number): string {
  if (globalScale < LABEL_TRUNCATE_SCALE && text.length > LABEL_TRUNCATE_LEN) {
    return text.slice(0, LABEL_TRUNCATE_LEN - 1) + '…'
  }
  return text
}

export function nodeRadius(val: number, extraR: number, pulseR: number): number {
  return Math.sqrt(val || 1) * NODE_RADIUS_SCALE + extraR + pulseR
}

export function pulseRadius(start: number | undefined, now: number): number {
  if (start === undefined) return 0
  const elapsed = now - start
  if (elapsed > PULSE_DURATION_MS) return 0
  return (1 - elapsed / PULSE_DURATION_MS) * PULSE_MAX_R
}

export function activeNeighborSet(
  nodeStates: ReadonlyMap<string, NodeState>,
  adjacency: ReadonlyMap<string, Neighbor[]>,
): Set<string> {
  const out = new Set<string>()
  for (const id of nodeStates.keys()) {
    for (const n of adjacency.get(id) ?? []) out.add(n.node)
  }
  return out
}

export function linkStyle(
  link: { source: LinkEnd; target: LinkEnd },
  traversal: TraversalState,
): LinkStyle {
  if (traversal.queryMode === null) return IDLE_LINK_STYLE
  const sA = traversal.nodeStates.has(endpointId(link.source))
  const tA = traversal.nodeStates.has(endpointId(link.target))
  const either = sA || tA
  const both = sA && tA
  return {
    color: both
      ? 'rgba(0,255,136,0.5)'
      : either
        ? 'rgba(0,255,136,0.25)'
        : 'rgba(255,255,255,0.03)',
    width: either ? 1.5 : 0.3,
    particleWidth: either ? 2.5 : 0,
    particleColor: either ? 'rgba(0,255,136,0.6)' : 'rgba(0,0,0,0)',
    particleSpeed: either ? 0.006 : 0.003,
  }
}

export function rankSearch(
  nodes: ReadonlyArray<ExplorerNode>,
  query: string,
  limit = SEARCH_LIMIT,
): ExplorerNode[] {
  const q = query.trim().toLowerCase()
  if (!q) return []
  const matches = nodes.filter((n) => {
    const p = n.properties ?? {}
    const hay = [n.label, n.id, p.long_title, p.tags, p.entry_type, p.project_ref]
      .filter(Boolean)
      .map((v) => String(v))
      .join(' ')
      .toLowerCase()
    return hay.includes(q)
  })
  const exact = (n: ExplorerNode) => ((n.label || '').toLowerCase() === q ? 1 : 0)
  matches.sort((a, b) => {
    const d = exact(b) - exact(a)
    if (d !== 0) return d
    return (b.val || 0) - (a.val || 0)
  })
  return matches.slice(0, limit)
}

export function zoomClampTarget(zoom: number): number | null {
  return zoom > MAX_AUTO_ZOOM ? MAX_AUTO_ZOOM : null
}

export function focusCenter(
  node: { x?: number; y?: number },
  opensEntryDrawer: boolean,
  containerWidth: number,
  rightInset = 0,
): { x: number; y: number } {
  const overlap = Math.max(0, INSPECTOR_W - rightInset)
  const shift =
    opensEntryDrawer && overlap > 0 && containerWidth + rightInset >= ENTRY_OFFSET_MIN_WIDTH
      ? overlap / 2 / FOCUS_ZOOM
      : 0
  return { x: (node.x ?? 0) + shift, y: node.y ?? 0 }
}

export function fpsFromFrameTimes(
  times: readonly number[],
  now: number,
  windowMs = PERF_WINDOW_MS,
): number {
  return Math.round((times.filter((t) => t > now - windowMs).length * 1000) / windowMs)
}

// ── Filtering ────────────────────────────────────────────────────────────────

export function filterGraph(
  nodes: ExplorerNode[],
  edges: ReadonlyArray<{ source: LinkEnd; target: LinkEnd; type: string }>,
  activeTypes: ReadonlySet<string>,
  scope: string,
): { nodes: ExplorerNode[]; links: ExplorerLink[] } {
  let warned = false
  const ids: Array<[string, string]> = edges.map((e) => {
    if (!warned && (typeof e.source !== 'string' || typeof e.target !== 'string')) {
      warned = true
      console.warn('[graph] filterGraph: edge endpoints are objects, payload was mutated in place')
    }
    return [endpointId(e.source), endpointId(e.target)]
  })

  const kept = new Set<string>()
  if (scope === '') {
    for (const n of nodes) {
      if (!isKnownType(n.type) || activeTypes.has(n.type)) kept.add(n.id)
    }
  } else {
    const byId = new Map<string, ExplorerNode>()
    for (const n of nodes) {
      byId.set(n.id, n)
      if (!isKnownType(n.type)) kept.add(n.id)
      else if (
        n.type === 'entry' &&
        activeTypes.has('entry') &&
        n.properties?.project_ref === scope
      ) {
        kept.add(n.id)
      }
    }
    const scopedEntries = new Set<string>()
    for (const n of nodes) {
      if (n.type === 'entry' && kept.has(n.id)) scopedEntries.add(n.id)
    }
    const maybeAdd = (otherId: string) => {
      const other = byId.get(otherId)
      if (
        other &&
        other.type !== 'entry' &&
        isKnownType(other.type) &&
        activeTypes.has(other.type)
      ) {
        kept.add(otherId)
      }
    }
    for (const [s, t] of ids) {
      if (scopedEntries.has(s)) maybeAdd(t)
      if (scopedEntries.has(t)) maybeAdd(s)
    }
  }

  const outNodes = nodes.filter((n) => kept.has(n.id))
  const links: ExplorerLink[] = []
  edges.forEach((e, i) => {
    const [s, t] = ids[i]
    if (kept.has(s) && kept.has(t)) links.push({ source: s, target: t, type: e.type })
  })
  return { nodes: outNodes, links }
}

// ── Navigation reducer ───────────────────────────────────────────────────────

/**
 * Camera plus lifted inspector view state at the moment a walk leaves a node.
 * A history record stores the snapshot taken BEFORE moving away, so Back
 * restores what was on screen when you left that node.
 */
export type NavSnapshot = { zoom: number; cx: number; cy: number; view: InspectorViewState }
export type HistoryRecord = { nodeId: string } & NavSnapshot

/** Records kept in the history; beyond this the oldest record drops. */
export const HISTORY_MAX = 500

export type ExplorerState = {
  focusedId: string | null
  highlightedId: string | null
  history: HistoryRecord[]
  /** 0-based index into history of the record the walk is at. */
  cursor: number
}

export const initialExplorerState: ExplorerState = {
  focusedId: null,
  highlightedId: null,
  history: [],
  cursor: 0,
}

export type ExplorerAction =
  | { type: 'select'; nodeId: string; snapshot: NavSnapshot }
  | { type: 'back'; snapshot: NavSnapshot }
  | { type: 'forward'; snapshot: NavSnapshot }
  | { type: 'unfocus'; snapshot?: NavSnapshot }
  | { type: 'clearAll'; snapshot?: NavSnapshot }
  | { type: 'prune'; renderedIds: ReadonlySet<string>; renderedTypes: ReadonlySet<string> }

/** Rewrite the record at the cursor with the outgoing live snapshot. */
function rewriteAtCursor(state: ExplorerState, snapshot: NavSnapshot): HistoryRecord[] {
  if (state.history.length === 0) return state.history
  return state.history.map((r, i) => (i === state.cursor ? { ...r, ...snapshot } : r))
}

export function explorerReducer(state: ExplorerState, action: ExplorerAction): ExplorerState {
  switch (action.type) {
    case 'select': {
      // the record at the cursor keeps the camera + inspector view we are
      // leaving; everything forward of the cursor is dropped, then the new
      // node appends with the same (pre-move) camera and a fresh view.
      const rewritten = rewriteAtCursor(state, action.snapshot)
      const kept = rewritten.slice(0, state.cursor + 1)
      const appended: HistoryRecord = {
        nodeId: action.nodeId,
        zoom: action.snapshot.zoom,
        cx: action.snapshot.cx,
        cy: action.snapshot.cy,
        view: DEFAULT_INSPECTOR_VIEW,
      }
      let history = [...kept, appended]
      let cursor = history.length - 1
      if (history.length > HISTORY_MAX) {
        history = history.slice(1)
        cursor = history.length - 1
      }
      return { ...state, focusedId: action.nodeId, highlightedId: action.nodeId, history, cursor }
    }
    case 'back': {
      const { history, cursor, focusedId } = state
      if (history.length === 0) return state
      if (focusedId !== null) {
        if (cursor === 0) return state
        const next = rewriteAtCursor(state, action.snapshot)
        const nextCursor = cursor - 1
        return {
          ...state,
          focusedId: next[nextCursor].nodeId,
          highlightedId: next[nextCursor].nodeId,
          history: next,
          cursor: nextCursor,
        }
      }
      // nothing focused: Back re-focuses the record at the cursor (the node
      // you just left) without touching the record or moving the cursor.
      return {
        ...state,
        focusedId: history[cursor].nodeId,
        highlightedId: history[cursor].nodeId,
      }
    }
    case 'forward': {
      const { history, cursor, focusedId } = state
      if (history.length === 0) return state
      if (cursor === history.length - 1) return state
      if (focusedId !== null) {
        const next = rewriteAtCursor(state, action.snapshot)
        const nextCursor = cursor + 1
        return {
          ...state,
          focusedId: next[nextCursor].nodeId,
          highlightedId: next[nextCursor].nodeId,
          history: next,
          cursor: nextCursor,
        }
      }
      const nextCursor = cursor + 1
      return {
        ...state,
        focusedId: history[nextCursor].nodeId,
        highlightedId: history[nextCursor].nodeId,
        cursor: nextCursor,
      }
    }
    case 'unfocus': {
      const history =
        action.snapshot !== undefined ? rewriteAtCursor(state, action.snapshot) : state.history
      return { ...state, focusedId: null, highlightedId: null, history }
    }
    case 'clearAll': {
      // Only a page reload clears history (the prune branch is the one
      // in-page exception): focus and highlight clear, the walk stays.
      const history =
        action.snapshot !== undefined ? rewriteAtCursor(state, action.snapshot) : state.history
      return { focusedId: null, highlightedId: null, history, cursor: state.cursor }
    }
    case 'prune': {
      const { renderedIds } = action
      let { focusedId, highlightedId, history, cursor } = state
      if (focusedId !== null && !renderedIds.has(focusedId)) {
        focusedId = null
        highlightedId = null
        history = []
        cursor = 0
      } else {
        if (history.length > 0) {
          const atCursor = history[Math.min(cursor, history.length - 1)]
          const kept = history.filter((r) => renderedIds.has(r.nodeId))
          if (kept.length !== history.length) {
            if (renderedIds.has(atCursor.nodeId)) {
              // the record at the cursor survives: shift with its branch
              const droppedBefore = history
                .slice(0, cursor)
                .filter((r) => !renderedIds.has(r.nodeId)).length
              cursor = cursor - droppedBefore
            } else {
              cursor = kept.length === 0 ? 0 : Math.min(cursor, kept.length - 1)
            }
            history = kept
          }
        }
        if (highlightedId !== null && !renderedIds.has(highlightedId)) highlightedId = null
      }
      if (
        focusedId === state.focusedId &&
        highlightedId === state.highlightedId &&
        history === state.history &&
        cursor === state.cursor
      ) {
        return state
      }
      return { focusedId, highlightedId, history, cursor }
    }
  }
}

/**
 * Coherence tripwire for the history: null when the cursor is in range, the
 * record at the cursor exists and (when renderedIds is given) every history
 * record's node is still rendered; otherwise a human-readable reason.
 */
export function historyInvariantError(
  state: ExplorerState,
  opts?: { renderedIds?: ReadonlySet<string> },
): string | null {
  const { history, cursor } = state
  if (history.length === 0) {
    return cursor === 0 ? null : `cursor ${cursor} out of range for an empty history`
  }
  if (cursor < 0 || cursor > history.length - 1) {
    return `cursor ${cursor} out of range 0..${history.length - 1}`
  }
  if (history[cursor] === undefined) return `record at cursor ${cursor} is missing`
  const renderedIds = opts?.renderedIds
  if (renderedIds !== undefined) {
    for (const r of history) {
      if (!renderedIds.has(r.nodeId)) return `history node ${r.nodeId} not rendered`
    }
  }
  return null
}

export interface FitBox {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export interface FitCaps {
  maxZoom: number
  maxAutoZoom: number
}

/** Bounding box of the nodes that have a layout position, or null if none do. */
export function boundingBox(
  nodes: ReadonlyArray<{ x?: number; y?: number }>,
): FitBox | null {
  let box: FitBox | null = null
  for (const n of nodes) {
    if (typeof n.x !== 'number' || typeof n.y !== 'number') continue
    if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) continue
    if (box === null) box = { minX: n.x, minY: n.y, maxX: n.x, maxY: n.y }
    else {
      box.minX = Math.min(box.minX, n.x)
      box.minY = Math.min(box.minY, n.y)
      box.maxX = Math.max(box.maxX, n.x)
      box.maxY = Math.max(box.maxY, n.y)
    }
  }
  return box
}

/**
 * Camera for fitting `bbox` into the canvas with `pad` on every side and an extra
 * `inset` reserved at the TOP only (an overlay covers it). Centering shifts the box
 * down into the free area; the zoom is capped like a normal auto-fit.
 */
export function fitView(
  bbox: FitBox,
  canvas: { width: number; height: number },
  inset: number,
  pad: number,
  caps: FitCaps,
): { zoom: number; center: { x: number; y: number } } {
  const bw = bbox.maxX - bbox.minX
  const bh = bbox.maxY - bbox.minY
  const availW = Math.max(1, canvas.width - 2 * pad)
  const availH = Math.max(1, canvas.height - inset - 2 * pad)
  const raw = Math.min(bw > 0 ? availW / bw : Infinity, bh > 0 ? availH / bh : Infinity)
  const zoom = Math.min(raw, caps.maxZoom, caps.maxAutoZoom)
  const cx = (bbox.minX + bbox.maxX) / 2
  const cy = (bbox.minY + bbox.maxY) / 2
  return { zoom, center: { x: cx, y: cy - inset / 2 / zoom } }
}
