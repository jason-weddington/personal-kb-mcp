// The tuned graph explorer (ported from personal_kb 47afec7 explorer.js):
// labels by zoom, click-to-focus, a browser-style Back/Forward
// history walked with ArrowLeft/ArrowRight, Escape unfocus (the inspector
// closes first), the non-modal node inspector. Query-mode traversal is
// rendered from the `traversal` prop; it is populated elsewhere.
import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react'
import type { Ref, RefObject } from 'react'
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import ForceGraph2D from 'react-force-graph-2d'
import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d'
import { RESULTS_FIT_MS, RESULTS_FIT_PADDING } from '../graph/queryModel'
import { drawNode } from '../graph/drawNode'
import type { NodeRenderState } from '../graph/drawNode'
import { GraphNodeInspector, DEFAULT_INSPECTOR_VIEW, memberFromNode } from './GraphNodeInspector'
import type { InspectorMember, InspectorViewState } from './GraphNodeInspector'
import {
  excludedMemberIds,
  groupNeighbors,
  memberMatches,
  computeFacets,
  facetRows,
  INSPECTOR_GROUP_OPEN_MAX,
  FACET_ROW_TOP_N,
} from '../graph/inspectorModel'
import type { ConnInput, GroupFilter } from '../graph/inspectorModel'
import {
  BACKGROUND,
  BACKTRACK_DRAWER_DELAY_MS,
  BACKTRACK_MS,
  CLAMP_AFTER_FIT_MS,
  CLAMP_MS,
  DRAWER_DELAY_MS,
  EMPTY_TRAVERSAL,
  FIT_MS,
  FIT_PADDING,
  FOCUS_MS,
  FOCUS_ZOOM,
  INITIAL_FIT_DELAY_MS,
  LINK_PARTICLES,
  MAX_AUTO_ZOOM,
  MAX_ZOOM,
  NODE_COLORS,
  PERF_TICK_MS,
  PERF_WINDOW_MS,
  RESET_FIT_DELAY_MS,
  activeNeighborSet,
  buildAdjacency,
  computeFocusSet,
  endpointId,
  explorerReducer,
  focusCenter,
  boundingBox,
  fitView,
  fpsFromFrameTimes,
  historyInvariantError,
  HISTORY_MAX,
  initialExplorerState,
  isLinkVisible,
  isNodeVisible,
  linkStyle,
  nodeColor,
  zoomClampTarget,
} from '../graph/explorerModel'
import type {
  Adjacency,
  ExplorerAction,
  ExplorerLink,
  ExplorerNode,
  ExplorerState,
  HistoryRecord,
  NavSnapshot,
  TraversalState,
} from '../graph/explorerModel'

export type { InspectorViewState } from './GraphNodeInspector'

type FgNode = NodeObject<ExplorerNode>
type FgLink = LinkObject<ExplorerNode, ExplorerLink>
type FgMethods = ForceGraphMethods<FgNode, FgLink>

export interface GraphExplorerHandle {
  flyToNode(nodeId: string): boolean
  fitNodes(nodeIds: ReadonlySet<string> | null, durationMs: number, padding: number): void
  emitParticlesFor(nodeIds: ReadonlySet<string>): void
  resetView(): void
  clearSelection(): void
  resetAll(): void
  /** Walk one step back through the focus history; false at the boundary. */
  back(): boolean
  /** Walk one step forward through the focus history; false at the boundary. */
  forward(): boolean
}

export interface HistoryInfo {
  canBack: boolean
  canForward: boolean
}

export interface GraphExplorerProps {
  graph: { nodes: ExplorerNode[]; links: ExplorerLink[] }
  traversal?: TraversalState
  /** Called first on Escape; return true to consume it. */
  onEscape?: () => boolean
  /** Mirrors the Back/Forward enable state after every history change. */
  onHistoryChange?: (info: HistoryInfo) => void
  perf?: { loadMs: number | null }
  /** Overlay at the top of the canvas; fits are padded by its measured bottom edge. */
  fitInsetRef?: RefObject<HTMLElement | null>
  /** Width in px of a panel beside the canvas (the entry drawer overlap shrinks by it). */
  rightInset?: number
  ref?: Ref<GraphExplorerHandle>
}

const KEY_GUARD_SELECTOR =
  'input, textarea, select, [contenteditable="true"], [role="presentation"], [role="dialog"]'

const RESIZE_FIT_DELAY_MS = 200

/** Shared empty set: with no inspector target there is no narrowing. */
const EMPTY_EXCLUDED: ReadonlySet<string> = new Set()

function sameHidden(a: ReadonlyArray<string>, b: ReadonlyArray<string>): boolean {
  return a.length === b.length && a.every((k, i) => k === b[i])
}

/** True when a back/forward dispatch cannot change the state at all. */
function walkIsNoop(state: ExplorerState, dir: 'back' | 'forward'): boolean {
  if (state.history.length === 0) return true
  // while nothing is focused, Back re-focuses the record at the cursor
  if (dir === 'back') return state.focusedId !== null && state.cursor === 0
  return state.cursor === state.history.length - 1
}

// force-graph repaints its hit-test canvas every 800 ms for as long as the
// canvas keeps redrawing, which here is always (particles, autoPauseRedraw
// off). Painting every link into it (27k on the live KB) made a long frame
// and a major GC every repaint, the periodic hitch in the particle flow.
// Links have no click handler, so they stay out of hit-testing.
const NO_LINK_HIT_AREA = () => {}

type Latest = {
  state: ExplorerState
  graph: GraphExplorerProps['graph']
  nodeById: Map<string, ExplorerNode>
  adjacency: Adjacency
  width: number
  height: number
  panelNode: ExplorerNode | null
  onEscape: (() => boolean) | undefined
  perf: GraphExplorerProps['perf']
  rightInset: number
}

export function GraphExplorer({
  graph,
  traversal = EMPTY_TRAVERSAL,
  onEscape,
  onHistoryChange,
  perf,
  fitInsetRef,
  rightInset = 0,
  ref,
}: GraphExplorerProps) {
  const fgRef = useRef<FgMethods | undefined>(undefined)
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })
  const [state, dispatch] = useReducer(explorerReducer, initialExplorerState)
  const [panelNode, setPanelNode] = useState<ExplorerNode | null>(null)
  const [inspectorView, setInspectorView] = useState<InspectorViewState>(DEFAULT_INSPECTOR_VIEW)
  // Every node focused this page session (walk or history restore); never
  // cleared except by unmount. Distinct from the query-mode traversal
  // 'visited' NodeState, which drawNode paints on the canvas.
  const [visited, setVisited] = useState<ReadonlySet<string>>(() => new Set())
  const { focusedId, highlightedId, history, cursor } = state
  const canBack = focusedId !== null ? cursor > 0 : history.length > 0
  const canForward = cursor < history.length - 1

  useEffect(() => {
    onHistoryChange?.({ canBack, canForward })
  }, [onHistoryChange, focusedId, cursor, history.length, canBack, canForward])

  // Refs mirroring the lifted view state, so the walk handlers read the live
  // values without re-subscribing the document keydown listener on every
  // inspector filter/expand update.
  const inspectorViewRef = useRef(inspectorView)
  // The live inspector scroll position — a ref, not state, so scrolling it
  // never re-renders (the controlled scrollTop round-trip bounced).
  const inspectorScrollTopRef = useRef(0)
  const visitedRef = useRef(visited)
  useLayoutEffect(() => {
    inspectorViewRef.current = inspectorView
    visitedRef.current = visited
  })

  // Close the node panel when a filter change stops rendering its node
  // (adjusting state while rendering, the React-recommended alternative to
  // setState in an effect).
  const [panelGraph, setPanelGraph] = useState(graph)
  if (panelGraph !== graph) {
    setPanelGraph(graph)
    if (panelNode !== null && !graph.nodes.some((n) => n.id === panelNode.id)) setPanelNode(null)
  }

  // ── Derived values ───────────────────────────────────────────────────────
  const adjacency = useMemo(() => buildAdjacency(graph.links), [graph])
  const nodeById = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph])
  const focusSet = useMemo(() => computeFocusSet(focusedId, adjacency), [focusedId, adjacency])
  const activeNeighbors = useMemo(
    () => activeNeighborSet(traversal.nodeStates, adjacency),
    [traversal, adjacency],
  )

  const panelGroups = useMemo(() => {
    if (panelNode === null) return []
    const conns: ConnInput[] = (adjacency.get(panelNode.id) ?? []).map((n) => ({
      nodeId: n.node,
      edgeType: n.type,
      dir: n.dir,
    }))
    return groupNeighbors(conns)
  }, [panelNode, adjacency])

  // Member metadata for the panel target's groups, from the loaded payload:
  // the excludedMemberIds match callback and the filter telemetry both need
  // the same shape the inspector's rows are built from.
  const panelMembersById = useMemo(() => {
    const byNode = new Map<string, InspectorMember>()
    for (const g of panelGroups) {
      for (const m of g.members) {
        if (byNode.has(m.nodeId)) continue
        const n = nodeById.get(m.nodeId)
        if (n === undefined) continue // InspectorConnections warns and falls back
        byNode.set(
          m.nodeId,
          memberFromNode(n, { nodeId: m.nodeId, edgeType: g.edgeType, dir: g.dir }),
        )
      }
    }
    return byNode
  }, [panelGroups, nodeById])

  const neighborIdSet = useCallback(
    (nodeId: string) => new Set((adjacency.get(nodeId) ?? []).map((n) => n.node)),
    [adjacency],
  )

  const memberPasses = useCallback(
    (nodeId: string, groupKey: string, filters: Record<string, GroupFilter>): boolean => {
      const filter = filters[groupKey] ?? { text: '', hidden: [] }
      const member = panelMembersById.get(nodeId)
      const m: InspectorMember =
        member !== undefined
          ? member
          : {
              nodeId,
              label: nodeId,
              nodeType: 'unknown',
              edgeType: '',
              dir: 'in',
              entryType: null,
              projectRef: null,
            }
      return memberMatches(m, filter, neighborIdSet(nodeId))
    },
    [panelMembersById, neighborIdSet],
  )

  // Canvas narrowing (item C): while an inspector target is focused, the
  // members of its actively filtered groups that fail the group's filter drop
  // out of the visible focus neighbourhood — the focused node itself is
  // guarded inside excludedMemberIds and never drops out.
  const excludedIds = useMemo(() => {
    if (focusedId === null) return EMPTY_EXCLUDED
    return excludedMemberIds(focusedId, panelGroups, inspectorView.filters, (nodeId, groupKey) =>
      memberPasses(nodeId, groupKey, inspectorView.filters),
    )
  }, [focusedId, panelGroups, inspectorView.filters, memberPasses])

  // ── Refs read by handlers and timers ─────────────────────────────────────
  const renderRef = useRef<NodeRenderState>({
    hoveredId: null,
    highlightedId: null,
    focusSet: null,
    traversal: EMPTY_TRAVERSAL,
    activeNeighbors: new Set(),
    now: 0,
  })
  const latestRef = useRef<Latest>({
    state: initialExplorerState,
    graph,
    nodeById,
    adjacency,
    width: size.width,
    height: size.height,
    panelNode,
    onEscape,
    perf,
    rightInset,
  })
  useLayoutEffect(() => {
    const rs = renderRef.current
    rs.highlightedId = highlightedId
    rs.focusSet = focusSet
    rs.traversal = traversal
    rs.activeNeighbors = activeNeighbors
    latestRef.current = {
      state,
      graph,
      nodeById,
      adjacency,
      width: size.width,
      height: size.height,
      panelNode,
      onEscape,
      perf,
      rightInset,
    }
  })

  // ── Timers ───────────────────────────────────────────────────────────────
  const timersRef = useRef(new Set<ReturnType<typeof setTimeout>>())
  const drawerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const graphFitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const resizeFitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // While the layout is still settling (a focus right after load, or after a
  // filter change), the camera follows the focused node so it stays in view.
  // Any pointer or wheel input on the canvas hands the camera back.
  const followRef = useRef<{ id: string; dx: number; dy: number; after: number } | null>(null)
  const stopFollow = useCallback(() => {
    followRef.current = null
  }, [])

  const schedule = useCallback((fn: () => void, ms: number) => {
    const id = setTimeout(() => {
      timersRef.current.delete(id)
      fn()
    }, ms)
    timersRef.current.add(id)
  }, [])

  const clearDrawerTimer = useCallback(() => {
    if (drawerTimerRef.current !== null) {
      clearTimeout(drawerTimerRef.current)
      drawerTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    const timers = timersRef.current
    return () => {
      timers.forEach((id) => clearTimeout(id))
      timers.clear()
      if (drawerTimerRef.current !== null) clearTimeout(drawerTimerRef.current)
      if (graphFitTimerRef.current !== null) clearTimeout(graphFitTimerRef.current)
    }
  }, [])

  // ── Camera primitives ────────────────────────────────────────────────────
  const clampLater = useCallback(
    (afterMs: number) => {
      schedule(() => {
        const fg = fgRef.current
        if (fg && zoomClampTarget(fg.zoom()) !== null) fg.zoom(MAX_AUTO_ZOOM, CLAMP_MS)
      }, afterMs + CLAMP_AFTER_FIT_MS)
    },
    [schedule],
  )

  /** Fit `ids` (or the connected nodes when null) below the overlay's measured inset. */
  const fitTo = useCallback(
    (ids: ReadonlySet<string> | null, ms: number, pad: number) => {
      const fg = fgRef.current
      if (!fg) return
      let inset = 0
      const overlay = fitInsetRef?.current
      const container = containerRef.current
      if (overlay && container) {
        inset = Math.max(
          0,
          overlay.getBoundingClientRect().bottom - container.getBoundingClientRect().top,
        )
      }
      const { graph: g, adjacency: adj, width, height } = latestRef.current
      const nodes = g.nodes.filter((n) => (ids ? ids.has(n.id) : adj.size === 0 || adj.has(n.id)))
      const box = boundingBox(nodes as ReadonlyArray<{ x?: number; y?: number }>)
      if (box) {
        const v = fitView(box, { width, height }, inset, pad, {
          maxZoom: MAX_ZOOM,
          maxAutoZoom: MAX_AUTO_ZOOM,
        })
        fg.zoom(v.zoom, ms)
        fg.centerAt(v.center.x, v.center.y, ms)
      }
      clampLater(ms)
    },
    [clampLater, fitInsetRef],
  )

  const fit = useCallback((ms: number, pad: number) => fitTo(null, ms, pad), [fitTo])

  const fitRef = useRef(fit)
  useLayoutEffect(() => {
    fitRef.current = fit
  })

  const showDetails = useCallback((node: ExplorerNode, view: InspectorViewState = DEFAULT_INSPECTOR_VIEW) => {
    // One inspector for every rendered node type. The view state resets in the
    // same update that changes the target (not in an effect), so a history
    // restore passes its saved snapshot here and nothing overwrites it.
    setPanelNode(node)
    setInspectorView(view)
    inspectorScrollTopRef.current = view.scrollTop
    const { nodeById, adjacency } = latestRef.current
    const conns: ConnInput[] = (adjacency.get(node.id) ?? []).map((n) => ({
      nodeId: n.node,
      edgeType: n.type,
      dir: n.dir,
    }))
    const groups = groupNeighbors(conns)
    // Facet rows for the telemetry: computed per large group over its FULL
    // static facet set (the same counts the chips show), total before the
    // top-N cut, shown after it. Groups at or under the open threshold
    // contribute nothing.
    const facetRowsForGroup = (g: (typeof groups)[number]) => {
      const members = g.members.flatMap((m) => {
        const n = nodeById.get(m.nodeId)
        return n !== undefined
          ? [memberFromNode(n, { nodeId: m.nodeId, edgeType: g.edgeType, dir: g.dir })]
          : []
      })
      const neighborByMember = new Map(
        members.map((m) => [
          m.nodeId,
          (adjacency.get(m.nodeId) ?? []).map(({ node: id, type }) => ({ id, type })),
        ]),
      )
      const nodeTypeById = new Map([...nodeById.values()].map((n) => [n.id, n.type]))
      const nodeLabelById = new Map([...nodeById.values()].map((n) => [n.id, n.label || n.id]))
      const facets = computeFacets({
        inspectedNodeId: node.id,
        members,
        neighborByMember,
        nodeTypeById,
        nodeLabelById,
      })
      return facetRows(facets, Object.keys(NODE_COLORS)).map((row) => ({
        label: row.label,
        total: row.facets.length,
        shown: Math.min(row.facets.length, FACET_ROW_TOP_N),
        facets: row.facets
          .slice(0, FACET_ROW_TOP_N)
          .map((f) => ({ key: f.key, count: f.count })),
      }))
    }
    console.info('[graph-inspector] open', {
      node_id: node.id,
      node_type: node.type,
      member_total: groups.reduce((sum, g) => sum + g.members.length, 0),
      groups: groups.map((g) => ({ key: g.key, count: g.members.length })),
      facet_rows: groups
        .filter((g) => g.members.length > INSPECTOR_GROUP_OPEN_MAX)
        .flatMap((g) => facetRowsForGroup(g)),
    })
  }, [])

  // ── History helpers ──────────────────────────────────────────────────────

  /** The live camera plus the live lifted inspector view, at press time. */
  const liveSnapshot = useCallback((): NavSnapshot | null => {
    const fg = fgRef.current
    if (!fg) return null
    const c = fg.centerAt()
    // scrollTop comes from the scroll container's ref: view state stopped
    // tracking the live position when scrolling stopped updating state.
    return {
      zoom: fg.zoom(),
      cx: c.x,
      cy: c.y,
      view: { ...inspectorViewRef.current, scrollTop: inspectorScrollTopRef.current },
    }
  }, [])

  /**
   * Dispatch a navigation action and tripwire the resulting state: the pure
   * reducer stays side-effect-free, the warn fires from this dispatch site.
   */
  const dispatchNav = useCallback((action: ExplorerAction): ExplorerState => {
    const prev = latestRef.current.state
    const next = explorerReducer(prev, action)
    dispatch(action)
    if (next !== prev) {
      const opts =
        action.type === 'prune' ? { renderedIds: action.renderedIds } : undefined
      if (historyInvariantError(next, opts) !== null) {
        console.warn('[graph-history] invariant violated', {
          action: action.type,
          cursor: next.cursor,
          history_len: next.history.length,
        })
      }
    }
    return next
  }, [])

  const addVisited = useCallback((id: string) => {
    setVisited((prev) => (prev.has(id) ? prev : new Set([...prev, id])))
  }, [])

  /** Visited count as it stands once `id` is added (telemetry only). */
  const visitedCountWith = useCallback((id: string) => {
    const cur = visitedRef.current
    return cur.has(id) ? cur.size : cur.size + 1
  }, [])

  const select = useCallback(
    (node: ExplorerNode) => {
      clearDrawerTimer()
      const fg = fgRef.current
      if (!fg) return
      const snapshot = liveSnapshot()
      if (snapshot === null) return
      const prev = latestRef.current.state
      const truncated = Math.max(0, prev.history.length - (prev.cursor + 1))
      const next = dispatchNav({ type: 'select', nodeId: node.id, snapshot })
      addVisited(node.id)
      console.info('[graph-history] select', {
        node_id: node.id,
        cursor: next.cursor,
        history_len: next.history.length,
        truncated,
        dropped_oldest: prev.history.length - truncated + 1 > HISTORY_MAX,
        can_back: next.focusedId !== null ? next.cursor > 0 : next.history.length > 0,
        can_forward: next.cursor < next.history.length - 1,
        visited_count: visitedCountWith(node.id),
      })
      const fc = focusCenter(
        node,
        node.type === 'entry',
        latestRef.current.width,
        latestRef.current.rightInset,
      )
      fg.centerAt(fc.x, fc.y, FOCUS_MS)
      fg.zoom(FOCUS_ZOOM, FOCUS_MS)
      followRef.current = {
        id: node.id,
        dx: fc.x - (node.x ?? 0),
        dy: fc.y - (node.y ?? 0),
        after: performance.now() + FOCUS_MS,
      }
      drawerTimerRef.current = setTimeout(() => {
        drawerTimerRef.current = null
        showDetails(node)
      }, DRAWER_DELAY_MS)
    },
    [clearDrawerTimer, showDetails, liveSnapshot, dispatchNav, addVisited, visitedCountWith],
  )

  const resetView = useCallback(() => {
    clearDrawerTimer()
    stopFollow()
    // The leave-camera is read BEFORE the fit, so Back after Escape restores
    // what was on screen at Escape time.
    const snapshot = liveSnapshot()
    dispatchNav({ type: 'unfocus', snapshot: snapshot ?? undefined })
    setPanelNode(null)
    fit(FIT_MS, FIT_PADDING)
  }, [clearDrawerTimer, stopFollow, fit, liveSnapshot, dispatchNav])

  /**
   * One Back or Forward step. The outgoing snapshot rewrites the record at
   * the cursor, the restored record's camera and inspector view are applied
   * together, and a boundary (or an empty history) is a silent no-op that
   * never touches the camera.
   */
  const walkHistory = useCallback(
    (dir: 'back' | 'forward'): boolean => {
      if (walkIsNoop(latestRef.current.state, dir)) return false
      const fg = fgRef.current
      const snapshot = liveSnapshot()
      if (fg === undefined || snapshot === null) return false
      const prev = latestRef.current.state
      const next = dispatchNav({ type: dir, snapshot })
      if (next === prev) return false
      const record: HistoryRecord | undefined = next.history[next.cursor]
      const node = record !== undefined ? latestRef.current.nodeById.get(record.nodeId) : undefined
      if (record === undefined || node === undefined) {
        console.warn('[graph-history] restore target not in payload', {
          node_id: record?.nodeId ?? null,
          cursor: next.cursor,
        })
        return false
      }
      clearDrawerTimer()
      stopFollow()
      fg.centerAt(record.cx, record.cy, BACKTRACK_MS)
      fg.zoom(record.zoom, BACKTRACK_MS)
      addVisited(record.nodeId)
      drawerTimerRef.current = setTimeout(() => {
        drawerTimerRef.current = null
        showDetails(node, record.view)
      }, BACKTRACK_DRAWER_DELAY_MS)
      console.info(dir === 'back' ? '[graph-history] back' : '[graph-history] forward', {
        node_id: record.nodeId,
        cursor: next.cursor,
        history_len: next.history.length,
        can_back: next.focusedId !== null ? next.cursor > 0 : next.history.length > 0,
        can_forward: next.cursor < next.history.length - 1,
        visited_count: visitedCountWith(record.nodeId),
      })
      return true
    },
    [clearDrawerTimer, stopFollow, showDetails, liveSnapshot, dispatchNav, addVisited, visitedCountWith],
  )

  const handleBackgroundClick = useCallback(() => {
    clearDrawerTimer()
    stopFollow()
    const snapshot = liveSnapshot()
    dispatchNav({ type: 'unfocus', snapshot: snapshot ?? undefined })
    setPanelNode(null)
    schedule(() => fit(FIT_MS, FIT_PADDING), RESET_FIT_DELAY_MS)
  }, [clearDrawerTimer, stopFollow, fit, schedule, liveSnapshot, dispatchNav])

  const selectById = useCallback(
    (nodeId: string) => {
      const node = latestRef.current.nodeById.get(nodeId)
      if (node) select(node)
    },
    [select],
  )

  // ── Keyboard: Escape unfocuses (inspector first), ArrowLeft/Right walk ────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' && e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      const target = e.target
      if (target instanceof Element && target.closest(KEY_GUARD_SELECTOR)) return
      if (e.key === 'Escape') {
        if (latestRef.current.onEscape?.()) return
        if (latestRef.current.panelNode !== null) {
          setPanelNode(null)
          return
        }
        resetView()
      } else if (e.key === 'ArrowLeft') {
        walkHistory('back')
      } else {
        walkHistory('forward')
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [resetView, walkHistory])

  // ── Camera follow: user input on the canvas ends it ──────────────────────
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('pointerdown', stopFollow, true)
    el.addEventListener('wheel', stopFollow, true)
    return () => {
      el.removeEventListener('pointerdown', stopFollow, true)
      el.removeEventListener('wheel', stopFollow, true)
    }
  }, [stopFollow])

  // ── Sizing ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let first = true
    const obs = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect
      setSize({ width: rect?.width || 800, height: rect?.height || 600 })
      if (first) {
        first = false
        return
      }
      // Re-fit after a resize settles, unless a node is focused.
      if (resizeFitTimerRef.current !== null) clearTimeout(resizeFitTimerRef.current)
      resizeFitTimerRef.current = setTimeout(() => {
        resizeFitTimerRef.current = null
        // While a traversal is active the width-change effect refits the results instead.
        if (renderRef.current.traversal.nodeStates.size > 0) return
        if (latestRef.current.state.focusedId === null) fitRef.current(FIT_MS, FIT_PADDING)
      }, RESIZE_FIT_DELAY_MS)
    })
    obs.observe(el)
    return () => {
      obs.disconnect()
      if (resizeFitTimerRef.current !== null) clearTimeout(resizeFitTimerRef.current)
    }
  }, [])

  // A width change during a query (e.g. the chat panel opening) refits the results.
  const prevWidthRef = useRef<number | null>(null)
  useEffect(() => {
    const prev = prevWidthRef.current
    prevWidthRef.current = size.width
    const states = renderRef.current.traversal.nodeStates
    if (prev === null || prev === size.width || states.size === 0) return
    if (latestRef.current.state.focusedId !== null) return
    const results = [...states].filter(([, v]) => v === 'result').map(([k]) => k)
    const ids = new Set(results.length > 0 ? results : [...states.keys()])
    fitTo(ids, RESULTS_FIT_MS, RESULTS_FIT_PADDING)
  }, [size.width, fitTo])

  // ── Graph changes: prune state, refit, perf start ────────────────────────
  const graphStartRef = useRef(0)
  const initialFitScheduledRef = useRef(false)
  useEffect(() => {
    const renderedIds = new Set(graph.nodes.map((n) => n.id))
    const renderedTypes = new Set(graph.nodes.map((n) => n.type))
    dispatchNav({ type: 'prune', renderedIds, renderedTypes })
    graphStartRef.current = performance.now()

    if (!initialFitScheduledRef.current && graph.nodes.length === 0) return
    const initial = !initialFitScheduledRef.current
    initialFitScheduledRef.current = true
    if (graphFitTimerRef.current !== null) clearTimeout(graphFitTimerRef.current)
    graphFitTimerRef.current = setTimeout(() => {
      graphFitTimerRef.current = null
      const s = latestRef.current.state
      if (s.focusedId === null && (!initial || s.history.length === 0)) {
        fit(FIT_MS, FIT_PADDING)
      }
    }, INITIAL_FIT_DELAY_MS)
  }, [graph, fit, dispatchNav])

  // ── Imperative handle ────────────────────────────────────────────────────
  useImperativeHandle(
    ref,
    () => ({
      flyToNode(nodeId: string): boolean {
        const { nodeById: byId, graph: g } = latestRef.current
        const node = byId.get(nodeId)
        if (!node) {
          console.debug('[graph] flyToNode: node not rendered', {
            nodeId,
            rendered_nodes: g.nodes.length,
          })
          return false
        }
        select(node)
        return true
      },
      fitNodes(nodeIds, durationMs, padding) {
        fitTo(nodeIds, durationMs, padding)
      },
      emitParticlesFor(nodeIds) {
        const fg = fgRef.current
        if (!fg) return
        for (const link of latestRef.current.graph.links) {
          const l = link as FgLink
          if (nodeIds.has(endpointId(l.source)) || nodeIds.has(endpointId(l.target))) {
            fg.emitParticle(l)
          }
        }
      },
      resetView,
      clearSelection() {
        stopFollow()
        dispatch({ type: 'unfocus' })
        clearDrawerTimer()
        setPanelNode(null)
      },
      resetAll() {
        stopFollow()
        // The live camera + view are read BEFORE the fit, so Back after a
        // query/chat close restores the camera that was on screen.
        const snapshot = liveSnapshot()
        dispatchNav({ type: 'clearAll', snapshot: snapshot ?? undefined })
        clearDrawerTimer()
        setPanelNode(null)
        fit(FIT_MS, FIT_PADDING)
      },
      back() {
        return walkHistory('back')
      },
      forward() {
        return walkHistory('forward')
      },
    }),
    [select, fitTo, resetView, clearDrawerTimer, stopFollow, fit, liveSnapshot, dispatchNav, walkHistory],
  )

  // ── ForceGraph accessors ─────────────────────────────────────────────────
  // nodeVisibility subtracts the excluded members (excludedMemberIds) from
  // the focusSet memo, so a narrowed member drops out of the visible focus
  // neighbourhood; linkVisibility additionally hides a link whose non-focused
  // endpoint is an excluded member (isLinkVisible alone only compares the
  // endpoints against focusedId).
  const nodeVisibility = useCallback(
    (n: FgNode) => isNodeVisible(n, focusSet) && !excludedIds.has(n.id),
    [focusSet, excludedIds],
  )
  const linkVisibility = useCallback(
    (l: FgLink) => {
      const s = endpointId(l.source)
      const t = endpointId(l.target)
      if (excludedIds.size > 0) {
        if (s !== focusedId && excludedIds.has(s)) return false
        if (t !== focusedId && excludedIds.has(t)) return false
      }
      return isLinkVisible(s, t, focusedId)
    },
    [focusedId, excludedIds],
  )
  const linkColor = useCallback((l: FgLink) => linkStyle(l, traversal).color, [traversal])
  const linkWidth = useCallback((l: FgLink) => linkStyle(l, traversal).width, [traversal])
  const particleWidth = useCallback(
    (l: FgLink) => linkStyle(l, traversal).particleWidth,
    [traversal],
  )
  const particleColor = useCallback(
    (l: FgLink) => linkStyle(l, traversal).particleColor,
    [traversal],
  )
  const particleSpeed = useCallback(
    (l: FgLink) => linkStyle(l, traversal).particleSpeed,
    [traversal],
  )
  const nodeCanvasObject = useCallback(
    (n: FgNode, ctx: CanvasRenderingContext2D, s: number) => drawNode(n, ctx, s, renderRef.current),
    [],
  )
  const onRenderFramePre = useCallback(() => {
    renderRef.current.now = performance.now()
  }, [])
  const onEngineTick = useCallback(() => {
    const f = followRef.current
    if (f === null || performance.now() < f.after) return
    const node = latestRef.current.nodeById.get(f.id)
    if (node?.x === undefined || node.y === undefined) return
    fgRef.current?.centerAt(node.x + f.dx, node.y + f.dy)
  }, [])
  const onNodeHover = useCallback((n: FgNode | null) => {
    renderRef.current.hoveredId = n?.id ?? null
  }, [])

  // ── Perf (opt-in) ────────────────────────────────────────────────────────
  const frameTimesRef = useRef<number[]>([])
  const settleMsRef = useRef<number | null>(null)
  const perfLoggedForRef = useRef<object | null>(null)
  const onRenderFramePost = useCallback(() => {
    const now = performance.now()
    const times = frameTimesRef.current
    times.push(now)
    while (times.length > 0 && times[0] <= now - PERF_WINDOW_MS) times.shift()
  }, [])
  const onEngineStop = useCallback(() => {
    const { graph: g, perf: p } = latestRef.current
    if (perfLoggedForRef.current === g) return
    perfLoggedForRef.current = g
    const now = performance.now()
    const settle = now - graphStartRef.current
    settleMsRef.current = settle
    console.info('[graph-perf]', {
      nodes: g.nodes.length,
      links: g.links.length,
      settle_ms: Math.round(settle),
      load_ms: p?.loadMs ?? null,
      fps: fpsFromFrameTimes(frameTimesRef.current, now),
    })
  }, [])
  const readPerf = useCallback(
    () => ({
      fps: fpsFromFrameTimes(frameTimesRef.current, performance.now()),
      settleMs: settleMsRef.current,
      loadMs: latestRef.current.perf?.loadMs ?? null,
    }),
    [],
  )


  // Filter edits write only filters[groupKey]; expanded and scrollTop writes
  // flow through here too, so the filter telemetry diffs before emitting.
  const onInspectorViewChange = (next: InspectorViewState) => {
    const prevFilters = inspectorView.filters
    setInspectorView(next)
    if (panelNode === null) return
    const changed = panelGroups.filter((g) => {
      const f = next.filters[g.key]
      if (f === undefined) return false
      const p = prevFilters[g.key]
      return p === undefined || p.text !== f.text || !sameHidden(p.hidden, f.hidden)
    })
    if (changed.length === 0) return
    const excluded = excludedMemberIds(
      panelNode.id,
      panelGroups,
      next.filters,
      (nodeId, groupKey) => memberPasses(nodeId, groupKey, next.filters),
    )
    for (const g of changed) {
      const filter = next.filters[g.key]
      const members = g.members.map((m) => panelMembersById.get(m.nodeId))
      const matching = members.reduce(
        (sum, m) => sum + (m !== undefined && memberPasses(m.nodeId, g.key, next.filters) ? 1 : 0),
        0,
      )
      console.info('[graph-inspector] filter', {
        node_id: panelNode.id,
        group_key: g.key,
        text: filter.text,
        hidden: filter.hidden,
        matching,
        total: g.members.length,
        canvas_excluded: excluded.size,
      })
    }
  }

  // Rows and pills write the same hover field onNodeHover writes, so drawNode
  // highlights an inspector-hovered member exactly like a canvas hover.
  const onInspectorHover = useCallback((id: string | null) => {
    renderRef.current.hoveredId = id
  }, [])

  return (
    <Box sx={{ position: 'relative', width: '100%', height: '100%' }}>
      <Box ref={containerRef} sx={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
        <ForceGraph2D<ExplorerNode, ExplorerLink>
          ref={fgRef}
          graphData={graph}
          width={size.width}
          height={size.height}
          nodeId="id"
          nodeVal="val"
          nodeColor={(n) => nodeColor(n.type)}
          nodeCanvasObject={nodeCanvasObject}
          nodeCanvasObjectMode={() => 'replace'}
          backgroundColor={BACKGROUND}
          maxZoom={MAX_ZOOM}
          autoPauseRedraw={false}
          onRenderFramePre={onRenderFramePre}
          linkColor={linkColor}
          linkWidth={linkWidth}
          linkDirectionalParticles={LINK_PARTICLES}
          linkDirectionalParticleWidth={particleWidth}
          linkDirectionalParticleColor={particleColor}
          linkDirectionalParticleSpeed={particleSpeed}
          linkPointerAreaPaint={NO_LINK_HIT_AREA}
          nodeVisibility={nodeVisibility}
          linkVisibility={linkVisibility}
          onNodeHover={onNodeHover}
          onEngineTick={onEngineTick}
          onNodeClick={select}
          onBackgroundClick={handleBackgroundClick}
          onRenderFramePost={perf ? onRenderFramePost : undefined}
          onEngineStop={perf ? onEngineStop : undefined}
        />
      </Box>

      {graph.nodes.length === 0 && (
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
          }}
        >
          <Typography variant="body2" sx={{ color: '#888' }}>
            No nodes match the current filters
          </Typography>
        </Box>
      )}

      <Box
        data-testid="graph-right-column"
        sx={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          bottom: '12px',
          zIndex: 2,
          maxWidth: 'calc(100% - 24px)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          gap: 1,
          pointerEvents: 'none',
          '& > *': { pointerEvents: 'auto' },
        }}
      >
      {panelNode && (
        <GraphNodeInspector
          node={panelNode}
          groups={panelGroups}
          nodeById={nodeById}
          adjacency={adjacency}
          view={inspectorView}
          visited={visited}
          onViewChange={onInspectorViewChange}
          scrollTopRef={inspectorScrollTopRef}
          onOpenNode={selectById}
          onHover={onInspectorHover}
          onClose={() => setPanelNode(null)}
        />
      )}
      </Box>

      {perf && <PerfReadout read={readPerf} />}
    </Box>
  )
}

type PerfSample = { fps: number; settleMs: number | null; loadMs: number | null }

function fmtSeconds(ms: number | null): string {
  return ms === null ? '…' : `${(ms / 1000).toFixed(1)} s`
}

function PerfReadout({ read }: { read: () => PerfSample }) {
  const [sample, setSample] = useState<PerfSample>({ fps: 0, settleMs: null, loadMs: null })
  useEffect(() => {
    const id = setInterval(() => setSample(read()), PERF_TICK_MS)
    return () => clearInterval(id)
  }, [read])
  return (
    <Box
      data-testid="graph-perf"
      sx={{
        position: 'absolute',
        bottom: '32px',
        left: '12px',
        zIndex: 1,
        fontSize: '12px',
        color: '#555',
      }}
    >
      {`${sample.fps} fps · settled ${fmtSeconds(sample.settleMs)} · loaded ${fmtSeconds(sample.loadMs)}`}
    </Box>
  )
}
