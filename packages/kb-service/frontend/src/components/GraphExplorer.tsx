// The tuned graph explorer (ported from personal_kb 47afec7 explorer.js):
// labels by zoom, click-to-focus, solo legend, ArrowLeft backtrack, Escape
// reset, non-modal node panel. Query-mode traversal is rendered from the
// `traversal` prop; it is populated elsewhere.
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
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import CloseIcon from '@mui/icons-material/Close'
import { alpha } from '@mui/material/styles'
import ForceGraph2D from 'react-force-graph-2d'
import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import { RESULTS_FIT_MS, RESULTS_FIT_PADDING } from '../graph/queryModel'
import { drawNode } from '../graph/drawNode'
import type { NodeRenderState } from '../graph/drawNode'
import {
  BACKGROUND,
  BACKTRACK_DRAWER_DELAY_MS,
  BACKTRACK_MS,
  CLAMP_AFTER_FIT_MS,
  CLAMP_MS,
  CONNECTIONS_LIMIT,
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
  NODE_PANEL_W,
  NODE_TYPES,
  PERF_TICK_MS,
  PERF_WINDOW_MS,
  RESET_FIT_DELAY_MS,
  activeNeighborSet,
  backtrackTarget,
  buildAdjacency,
  computeFocusSet,
  endpointId,
  explorerReducer,
  focusCenter,
  boundingBox,
  fitView,
  fpsFromFrameTimes,
  initialExplorerState,
  isLinkVisible,
  isNodeVisible,
  linkStyle,
  nodeColor,
  zoomClampTarget,
} from '../graph/explorerModel'
import type {
  Adjacency,
  ExplorerLink,
  ExplorerNode,
  ExplorerState,
  Neighbor,
  TraversalState,
} from '../graph/explorerModel'

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
}

export interface GraphExplorerProps {
  graph: { nodes: ExplorerNode[]; links: ExplorerLink[] }
  traversal?: TraversalState
  /** Called first on Escape; return true to consume it. */
  onEscape?: () => boolean
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
  openEntry: (id: string) => void
  onEscape: (() => boolean) | undefined
  perf: GraphExplorerProps['perf']
  rightInset: number
}

export function GraphExplorer({
  graph,
  traversal = EMPTY_TRAVERSAL,
  onEscape,
  perf,
  fitInsetRef,
  rightInset = 0,
  ref,
}: GraphExplorerProps) {
  const { openEntry } = useEntryDrawer()
  const fgRef = useRef<FgMethods | undefined>(undefined)
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })
  const [state, dispatch] = useReducer(explorerReducer, initialExplorerState)
  const [panelNode, setPanelNode] = useState<ExplorerNode | null>(null)
  const { focusedId, highlightedId, soloType } = state

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

  // ── Refs read by handlers and timers ─────────────────────────────────────
  const renderRef = useRef<NodeRenderState>({
    hoveredId: null,
    highlightedId: null,
    focusSet: null,
    soloType: null,
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
    openEntry,
    onEscape,
    perf,
    rightInset,
  })
  useLayoutEffect(() => {
    const rs = renderRef.current
    rs.highlightedId = highlightedId
    rs.focusSet = focusSet
    rs.soloType = soloType
    rs.traversal = traversal
    rs.activeNeighbors = activeNeighbors
    latestRef.current = {
      state,
      graph,
      nodeById,
      adjacency,
      width: size.width,
      height: size.height,
      openEntry,
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

  const showDetails = useCallback((node: ExplorerNode) => {
    if (node.type === 'entry') {
      setPanelNode(null)
      latestRef.current.openEntry(node.id)
    } else {
      setPanelNode(node)
    }
  }, [])

  const select = useCallback(
    (node: ExplorerNode) => {
      clearDrawerTimer()
      const fg = fgRef.current
      if (!fg) return
      const c = fg.centerAt()
      dispatch({ type: 'select', nodeId: node.id, view: { zoom: fg.zoom(), cx: c.x, cy: c.y } })
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
    [clearDrawerTimer, showDetails],
  )

  const resetView = useCallback(() => {
    clearDrawerTimer()
    stopFollow()
    dispatch({ type: 'reset' })
    setPanelNode(null)
    fit(FIT_MS, FIT_PADDING)
  }, [clearDrawerTimer, stopFollow, fit])

  const backtrack = useCallback(() => {
    const t = backtrackTarget(latestRef.current.state.history)
    if (t === null) return
    clearDrawerTimer()
    stopFollow()
    dispatch({ type: 'backtrack' })
    const fg = fgRef.current
    fg?.centerAt(t.view.cx, t.view.cy, BACKTRACK_MS)
    fg?.zoom(t.view.zoom, BACKTRACK_MS)
    drawerTimerRef.current = setTimeout(() => {
      drawerTimerRef.current = null
      const node = latestRef.current.nodeById.get(t.nodeId)
      if (node) showDetails(node)
    }, BACKTRACK_DRAWER_DELAY_MS)
  }, [clearDrawerTimer, stopFollow, showDetails])

  const handleBackgroundClick = useCallback(() => {
    clearDrawerTimer()
    stopFollow()
    dispatch({ type: 'unfocus' })
    setPanelNode(null)
    schedule(() => fit(FIT_MS, FIT_PADDING), RESET_FIT_DELAY_MS)
  }, [clearDrawerTimer, stopFollow, fit, schedule])

  const toggleSolo = useCallback(
    (nodeType: string) => {
      dispatch({ type: 'toggleSolo', nodeType })
      schedule(() => fit(FIT_MS, FIT_PADDING), RESET_FIT_DELAY_MS)
    },
    [fit, schedule],
  )

  const selectById = useCallback(
    (nodeId: string) => {
      const node = latestRef.current.nodeById.get(nodeId)
      if (node) select(node)
    },
    [select],
  )

  // ── Keyboard: Escape resets, ArrowLeft backtracks ────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' && e.key !== 'ArrowLeft') return
      const target = e.target
      if (target instanceof Element && target.closest(KEY_GUARD_SELECTOR)) return
      if (e.key === 'Escape') {
        if (latestRef.current.onEscape?.()) return
        resetView()
      } else {
        backtrack()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [resetView, backtrack])

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
    dispatch({ type: 'prune', renderedIds, renderedTypes })
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
  }, [graph, fit])

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
        dispatch({ type: 'clearSolo' })
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
        dispatch({ type: 'clearSolo' })
        clearDrawerTimer()
        setPanelNode(null)
      },
      resetAll() {
        stopFollow()
        dispatch({ type: 'clearAll' })
        clearDrawerTimer()
        setPanelNode(null)
        fit(FIT_MS, FIT_PADDING)
      },
    }),
    [select, fitTo, resetView, clearDrawerTimer, stopFollow, fit],
  )

  // ── ForceGraph accessors ─────────────────────────────────────────────────
  const nodeVisibility = useCallback(
    (n: FgNode) => isNodeVisible(n, focusSet, soloType),
    [focusSet, soloType],
  )
  const linkVisibility = useCallback(
    (l: FgLink) => isLinkVisible(endpointId(l.source), endpointId(l.target), focusedId, soloType),
    [focusedId, soloType],
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

  const panelConnections = panelNode ? (adjacency.get(panelNode.id) ?? []) : []

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
          bottom: '56px',
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
        <NodePanel
          node={panelNode}
          connections={panelConnections}
          onClose={() => setPanelNode(null)}
          onSelect={selectById}
        />
      )}

      <Box
        role="group"
        aria-label="Solo node type"
        sx={{
          marginTop: 'auto',
          flexShrink: 0,
          maxWidth: '100%',
          boxSizing: 'border-box',
          fontSize: { xs: '13px', sm: '16px' },
          color: 'text.primary',
          display: 'flex',
          gap: { xs: '8px 12px', sm: '16px' },
          flexWrap: 'wrap',
          bgcolor: (t) => alpha(t.palette.background.paper, 0.85),
          padding: '8px 12px',
          borderRadius: '6px',
        }}
      >
        {NODE_TYPES.map((type) => (
          <Box
            key={type}
            component="button"
            type="button"
            aria-label={`Solo ${type}`}
            aria-pressed={soloType === type}
            onClick={() => toggleSolo(type)}
            sx={{
              background: 'none',
              border: 0,
              padding: 0,
              color: 'inherit',
              font: 'inherit',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              transition: 'opacity 0.2s',
              opacity: soloType !== null && soloType !== type ? 0.3 : 1,
            }}
          >
            <Box
              component="span"
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: NODE_COLORS[type],
                border: 1,
                borderColor: 'divider',
                boxSizing: 'border-box',
              }}
            />
            {type}
          </Box>
        ))}
      </Box>
      </Box>

      {perf && <PerfReadout read={readPerf} />}
    </Box>
  )
}

function NodePanel({
  node,
  connections,
  onClose,
  onSelect,
}: {
  node: ExplorerNode
  connections: Neighbor[]
  onClose: () => void
  onSelect: (id: string) => void
}) {
  const props = node.properties ?? {}
  return (
    <Paper
      role="region"
      aria-label="Node details"
      elevation={8}
      sx={{
        width: NODE_PANEL_W,
        maxWidth: '100%',
        flexShrink: 1,
        minHeight: 0,
        overflowY: 'auto',
        p: 2,
      }}
    >
      <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
        <Chip
          label={node.type}
          size="small"
          sx={{ bgcolor: nodeColor(node.type), color: '#000', mb: 1 }}
        />
        <IconButton size="small" aria-label="Close node details" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Stack>
      <Typography variant="h6" sx={{ wordBreak: 'break-word' }}>
        {node.label || node.id}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {node.id}
      </Typography>
      <Typography variant="body2">
        <strong>Val:</strong> {node.val}
      </Typography>
      {Object.keys(props).length > 0 && (
        <>
          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle2" gutterBottom>
            Properties
          </Typography>
          <Stack spacing={0.5}>
            {Object.entries(props).map(([k, v]) => (
              <Typography key={k} variant="body2" sx={{ wordBreak: 'break-word' }}>
                <strong>{k}:</strong>{' '}
                {typeof v === 'object' ? JSON.stringify(v) : String(v ?? '')}
              </Typography>
            ))}
          </Stack>
        </>
      )}
      {connections.length > 0 && (
        <>
          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle2" gutterBottom>
            Connections ({connections.length})
          </Typography>
          <Stack>
            {connections.slice(0, CONNECTIONS_LIMIT).map((c, i) => (
              <Box
                key={`${c.dir}-${c.type}-${c.node}-${i}`}
                component="button"
                type="button"
                onClick={() => onSelect(c.node)}
                sx={{
                  background: 'none',
                  border: 0,
                  borderBottom: 1,
                  borderColor: 'divider',
                  color: 'text.secondary',
                  font: 'inherit',
                  fontSize: 12,
                  textAlign: 'left',
                  py: 0.5,
                  px: 0,
                  cursor: 'pointer',
                  wordBreak: 'break-word',
                  '&:hover': { color: 'text.primary' },
                }}
              >
                {c.dir === 'out' ? '→' : '←'} <strong>{c.type}</strong> {c.node}
              </Box>
            ))}
            {connections.length > CONNECTIONS_LIMIT && (
              <Typography variant="body2" color="text.secondary">
                ... and {connections.length - CONNECTIONS_LIMIT} more
              </Typography>
            )}
          </Stack>
        </>
      )}
    </Paper>
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
