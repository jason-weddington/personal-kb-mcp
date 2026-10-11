import { useEffect, useMemo, useRef, useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Select from '@mui/material/Select'
import Stack from '@mui/material/Stack'
import { getGraphFull } from '../api'
import { GraphExplorer } from '../components/GraphExplorer'
import type { GraphExplorerHandle } from '../components/GraphExplorer'
import { GraphSearch } from '../components/GraphSearch'
import { BACKGROUND, NODE_COLORS, NODE_TYPES, filterGraph } from '../graph/explorerModel'
import type { GraphFullResponse } from '../kbTypes'

const EMPTY_PAYLOAD: GraphFullResponse = {
  nodes: [],
  edges: [],
  stats: { node_count: 0, edge_count: 0 },
}

export function Graph() {
  const explorerRef = useRef<GraphExplorerHandle>(null)
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
      }}
    >
      <GraphExplorer ref={explorerRef} graph={graph} perf={perfOn ? { loadMs } : undefined} />

      <Paper
        sx={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          maxWidth: 'calc(100% - 24px)',
          zIndex: 1,
          p: 1.5,
        }}
      >
        <Stack spacing={1.5}>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <GraphSearch
              nodes={graph.nodes}
              onSelectNode={(id) => {
                explorerRef.current?.flyToNode(id)
              }}
            />
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
                  }}
                />
              )
            })}
          </Stack>
        </Stack>
      </Paper>

      <Box
        sx={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          zIndex: 1,
          fontSize: '12px',
          color: '#555',
        }}
      >
        {`${graph.nodes.length} of ${data.stats.node_count} nodes · ${graph.links.length} of ${data.stats.edge_count} edges`}
      </Box>
    </Box>
  )
}
