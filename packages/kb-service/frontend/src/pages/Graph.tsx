import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Snackbar from '@mui/material/Snackbar'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Select from '@mui/material/Select'
import Stack from '@mui/material/Stack'
import { alpha, darken } from '@mui/material/styles'
import { getGraphFull } from '../api'
import { GraphExplorer } from '../components/GraphExplorer'
import type { GraphExplorerHandle } from '../components/GraphExplorer'
import { GraphAskBar } from '../components/GraphAskBar'
import { GraphQueryPanel } from '../components/GraphQueryPanel'
import { GraphChatPanel } from '../components/GraphChatPanel'
import { GraphSearch } from '../components/GraphSearch'
import { useAuth } from '../contexts/AuthContext'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import { BACKGROUND, NODE_COLORS, NODE_TYPES, filterGraph } from '../graph/explorerModel'
import { GRAPH_CHAT_W, chatSeedDecision } from '../graph/chatModel'
import { useGraphQuery } from '../graph/useGraphQuery'
import type { GraphFullResponse } from '../kbTypes'

const EMPTY_PAYLOAD: GraphFullResponse = {
  nodes: [],
  edges: [],
  stats: { node_count: 0, edge_count: 0 },
}

export function Graph() {
  const explorerRef = useRef<GraphExplorerHandle>(null)
  const controlsRef = useRef<HTMLDivElement>(null)
  const [payload, setPayload] = useState<GraphFullResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadMs, setLoadMs] = useState<number | null>(null)
  const [perfOn] = useState(() => new URLSearchParams(window.location.search).has('perf'))

  const [activeTypes, setActiveTypes] = useState<ReadonlySet<string>>(() => new Set(NODE_TYPES))
  const [scope, setScope] = useState('')

  useEffect(() => {
    const t0 = performance.now()
    getGraphFull()
      .then((data) => {
        setLoadMs(performance.now() - t0)
        setPayload(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
  }, [])

  const data = payload ?? EMPTY_PAYLOAD

  const scopeOptions = useMemo(() => {
    const seen = new Set<string>()
    for (const node of data.nodes) {
      const ref = node.properties?.project_ref
      if (node.type === 'entry' && typeof ref === 'string' && ref) seen.add(ref)
    }
    return Array.from(seen).sort()
  }, [data])

  const graph = useMemo(
    () => filterGraph(data.nodes, data.edges, activeTypes, scope),
    [data, activeTypes, scope],
  )

  const { openEntry, closeEntry, setNodeHandler } = useEntryDrawer()
  const [searchParams, setSearchParams] = useSearchParams()
  const renderedIds = useMemo(() => new Set(graph.nodes.map((n) => n.id)), [graph])
  const allIds = useMemo(() => new Set((payload?.nodes ?? []).map((n) => n.id)), [payload])
  const graphIds = useMemo(() => ({ rendered: renderedIds, all: allIds }), [renderedIds, allIds])
  const labelById = useMemo(
    () => new Map((payload?.nodes ?? []).map((n) => [n.id, n.label] as const)),
    [payload],
  )
  const labelFor = useCallback((id: string) => labelById.get(id) ?? null, [labelById])

  const { state: queryState, traversal, start, close, reveal } = useGraphQuery(
    explorerRef,
    graphIds,
  )

  const { authMode } = useAuth()
  const chatEnabled = authMode === 'jwt'
  const {
    question: queryQuestion,
    answer: queryAnswer,
    sourceIds: querySourceIds,
    entries: queryEntries,
  } = queryState
  const seedDecision = useMemo(
    () =>
      chatSeedDecision({
        question: queryQuestion,
        answer: queryAnswer,
        sourceIds: querySourceIds,
        entries: queryEntries,
      }),
    [queryQuestion, queryAnswer, querySourceIds, queryEntries],
  )
  const chatSeed = chatEnabled ? (seedDecision?.seed ?? null) : null

  const loggedSeedRef = useRef<typeof seedDecision>(null)
  useEffect(() => {
    if (seedDecision === null || seedDecision === loggedSeedRef.current) return
    loggedSeedRef.current = seedDecision
    console.info('[graph-chat] seed', {
      decision: chatEnabled && seedDecision.seed ? 'open' : 'none',
      reason: chatEnabled ? seedDecision.reason : 'auth_none',
      mode: seedDecision.mode,
      auth_mode: authMode,
      answer_len: queryAnswer?.length ?? 0,
      entry_ids: seedDecision.seed?.entryIds.length ?? 0,
    })
  }, [seedDecision, chatEnabled, authMode, queryAnswer])

  const [cite, setCite] = useState<{ id: string; open: boolean }>({ id: '', open: false })
  const citeOpenRef = useRef(cite.open)
  useLayoutEffect(() => {
    citeOpenRef.current = cite.open
  }, [cite.open])

  const phaseRef = useRef(queryState.phase)
  useLayoutEffect(() => {
    phaseRef.current = queryState.phase
  })
  const handleEscape = useCallback((): boolean => {
    if (citeOpenRef.current) {
      setCite((c) => ({ ...c, open: false }))
      return true
    }
    if (phaseRef.current === 'idle') return false
    close()
    return true
  }, [close])

  const handleCitation = useCallback(
    (id: string) => {
      if (explorerRef.current?.flyToNode(id) === true) return
      console.info('[graph-chat] citation not rendered', {
        id,
        in_payload: allIds.has(id),
        scope,
        active_types: activeTypes.size,
      })
      setCite({ id, open: true })
    },
    [allIds, scope, activeTypes],
  )

  const handleNewEntries = useCallback(
    (ids: string[]): string[] => {
      const shown = ids.filter((id) => renderedIds.has(id))
      const hidden = ids.filter((id) => !renderedIds.has(id))
      console.info('[graph-chat] new_entries', {
        count: ids.length,
        rendered: shown.length,
        not_rendered: hidden.slice(0, 10),
      })
      if (shown.length > 0) reveal(shown)
      return hidden
    },
    [renderedIds, reveal],
  )

  // Fly to a node; if it is filtered out, reset filters and retry once after
  // the unfiltered graph has rendered.
  const pendingFlyRef = useRef<string | null>(null)
  const [retryTick, setRetryTick] = useState(0)
  const flyOrReset = useCallback(
    (id: string, canRetry: boolean) => {
      if (explorerRef.current?.flyToNode(id) === true) return
      if (canRetry && allIds.has(id)) {
        setActiveTypes(new Set(NODE_TYPES))
        setScope('')
        pendingFlyRef.current = id
        setRetryTick((t) => t + 1)
        return
      }
      console.debug('[graph] openNode: node not rendered', { nodeId: id })
    },
    [allIds],
  )

  useEffect(() => {
    const id = pendingFlyRef.current
    if (id === null) return
    pendingFlyRef.current = null
    flyOrReset(id, false)
  }, [retryTick, graph, flyOrReset])

  const openNodeHandler = useCallback(
    (id: string) => {
      closeEntry()
      flyOrReset(id, true)
    },
    [closeEntry, flyOrReset],
  )
  useEffect(() => {
    setNodeHandler(openNodeHandler)
    return () => setNodeHandler(null)
  }, [setNodeHandler, openNodeHandler])

  // ?focus=<node id>: fly once after the payload has loaded, then drop the param.
  const focusParam = searchParams.get('focus')
  const focusDoneRef = useRef(false)
  useEffect(() => {
    if (focusParam === null || payload === null || loading || focusDoneRef.current) return
    focusDoneRef.current = true
    flyOrReset(focusParam, true)
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.delete('focus')
        return next
      },
      { replace: true },
    )
  }, [focusParam, payload, loading, flyOrReset, setSearchParams])

  const toggleType = (type: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>
  }

  return (
    <Box
      sx={{
        position: 'relative',
        m: -3,
        height: 'calc(100vh - 64px)',
        overflow: 'hidden',
        bgcolor: BACKGROUND,
        display: 'flex',
      }}
    >
      <Box
        data-testid="graph-canvas-area"
        sx={{ position: 'relative', flex: 1, minWidth: 0, height: '100%' }}
      >
      <GraphExplorer
        ref={explorerRef}
        graph={graph}
        traversal={traversal}
        onEscape={handleEscape}
        perf={perfOn ? { loadMs } : undefined}
        fitInsetRef={controlsRef}
        rightInset={chatSeed ? GRAPH_CHAT_W : 0}
      />

      <Box
        data-testid="graph-left-column"
        sx={{
          position: 'absolute',
          top: 12,
          left: 12,
          zIndex: 1,
          maxWidth: 'calc(100% - 24px)',
          maxHeight: 'calc(100% - 72px)',
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          pointerEvents: 'none',
          '& > *': { pointerEvents: 'auto' },
        }}
      >
        <Paper ref={controlsRef} sx={{ p: 1.5, flexShrink: 0 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <GraphSearch
                nodes={graph.nodes}
                placeholder="Search nodes or ask a question..."
                onFreeTextSubmit={start}
                onSelectNode={(id) => {
                  explorerRef.current?.flyToNode(id)
                }}
              />
              <GraphAskBar onAsk={start} busy={queryState.phase === 'streaming'} />
              {scopeOptions.length > 0 && (
                <FormControl size="small" sx={{ minWidth: 180 }}>
                  <InputLabel id="graph-scope-label">Scope</InputLabel>
                  <Select
                    labelId="graph-scope-label"
                    label="Scope"
                    value={scope}
                    onChange={(e) => setScope(e.target.value)}
                  >
                    <MenuItem value="">All</MenuItem>
                    {scopeOptions.map((p) => (
                      <MenuItem key={p} value={p}>
                        {p}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}
            </Stack>
            <Stack
              role="group"
              aria-label="Node type filters"
              direction="row"
              flexWrap="wrap"
              useFlexGap
              spacing={1}
            >
              {NODE_TYPES.map((type) => {
                const on = activeTypes.has(type)
                return (
                  <Chip
                    key={type}
                    label={type}
                    onClick={() => toggleType(type)}
                    variant={on ? 'filled' : 'outlined'}
                    size="small"
                    sx={{
                      bgcolor: on ? NODE_COLORS[type] : undefined,
                      color: on ? '#000' : undefined,
                      borderColor: on ? undefined : NODE_COLORS[type],
                      '&:hover, &.Mui-focusVisible': {
                        bgcolor: on
                          ? darken(NODE_COLORS[type], 0.15)
                          : alpha(NODE_COLORS[type], 0.18),
                      },
                    }}
                  />
                )
              })}
            </Stack>
          </Stack>
        </Paper>
        {chatSeed?.mode !== 'summarize' && (
        <GraphQueryPanel
          state={queryState}
          renderedIds={renderedIds}
          labelFor={labelFor}
          onSelectEntry={(id) => {
            const shown = explorerRef.current?.flyToNode(id) ?? false
            if (!shown) openEntry(id)
          }}
          onClose={close}
          onCitation={handleCitation}
        />
        )}
      </Box>

      <Box
        sx={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          zIndex: 1,
          fontSize: '12px',
          color: 'text.secondary',
        }}
      >
        {`${graph.nodes.length} of ${data.stats.node_count} nodes · ${graph.links.length} of ${data.stats.edge_count} edges`}
      </Box>
      </Box>
      {chatSeed && (
        <GraphChatPanel
          key={`${chatSeed.mode}|${chatSeed.question}|${chatSeed.answer}`}
          seed={chatSeed}
          onCitation={handleCitation}
          onNewEntries={handleNewEntries}
          onClose={close}
        />
      )}
      <Snackbar
        open={cite.open}
        autoHideDuration={6000}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        onClose={(_, reason) => {
          if (reason !== 'clickaway') setCite((c) => ({ ...c, open: false }))
        }}
        message={`${cite.id} is not shown on the graph`}
        action={
          <Button
            size="small"
            onClick={() => {
              openEntry(cite.id)
              setCite((c) => ({ ...c, open: false }))
            }}
          >
            Open entry
          </Button>
        }
      />
    </Box>
  )
}
