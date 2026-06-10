import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import Divider from '@mui/material/Divider'
import Drawer from '@mui/material/Drawer'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import ForceGraph2D from 'react-force-graph-2d'
import { getGraphFull } from '../api'
import type { GraphFullResponse, GraphFullNode, GraphFullEdge } from '../kbTypes'

// Legacy 8-type colour palette from personal_kb explorer renderer.py
const NODE_COLORS: Record<string, string> = {
  entry: '#e0e0e0',
  tag: '#00bcd4',
  project: '#ff9800',
  person: '#ffc107',
  tool: '#4caf50',
  concept: '#9c27b0',
  technology: '#2196f3',
  note: '#78909c',
}
const KNOWN_TYPES = Object.keys(NODE_COLORS)
const FALLBACK_COLOR = '#666'

const DRAWER_WIDTH = 320

export function Graph() {
  const navigate = useNavigate()
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)

  const [payload, setPayload] = useState<GraphFullResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filter state
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set(KNOWN_TYPES))
  const [projectFilter, setProjectFilter] = useState('')

  // Drawer
  const [drawerNode, setDrawerNode] = useState<GraphFullNode | null>(null)

  useEffect(() => {
    getGraphFull()
      .then((data) => {
        setPayload(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
  }, [])

  // Track container width for responsive canvas
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(() => {
      setWidth(el.clientWidth || 800)
    })
    obs.observe(el)
    setWidth(el.clientWidth || 800)
    return () => obs.disconnect()
  }, [])

  // Available project options derived from payload (entry nodes only)
  const projectOptions = useMemo(() => {
    if (!payload) return []
    const seen = new Set<string>()
    for (const node of payload.nodes) {
      if (node.type === 'entry' && typeof node.properties.project_ref === 'string') {
        seen.add(node.properties.project_ref)
      }
    }
    return Array.from(seen).sort()
  }, [payload])

  // Filtered subgraph
  const graphData = useMemo(() => {
    if (!payload) return { nodes: [], links: [] }

    const keptNodeIds = new Set<string>()
    for (const node of payload.nodes) {
      const isKnown = KNOWN_TYPES.includes(node.type)
      if (!isKnown) {
        // Unknown types are ALWAYS kept
        keptNodeIds.add(node.id)
        continue
      }
      if (!activeTypes.has(node.type)) continue
      if (projectFilter && node.type === 'entry') {
        if (node.properties.project_ref !== projectFilter) continue
      }
      keptNodeIds.add(node.id)
    }

    // Also keep non-entry nodes that are still connected to a kept node
    // (re-run connectivity pass for project filter)
    if (projectFilter) {
      let changed = true
      while (changed) {
        changed = false
        for (const edge of payload.edges) {
          const srcKept = keptNodeIds.has(edge.source)
          const tgtKept = keptNodeIds.has(edge.target)
          // If one end is kept (entry type), also keep the other non-entry end
          if (srcKept !== tgtKept) {
            const otherNode = payload.nodes.find(
              (n) => n.id === (srcKept ? edge.target : edge.source),
            )
            if (otherNode && otherNode.type !== 'entry') {
              keptNodeIds.add(otherNode.id)
              changed = true
            }
          }
        }
      }
    }

    const nodes = payload.nodes.filter((n) => keptNodeIds.has(n.id))
    const links = (payload.edges as GraphFullEdge[]).filter(
      (e) => keptNodeIds.has(e.source) && keptNodeIds.has(e.target),
    )
    return { nodes, links }
  }, [payload, activeTypes, projectFilter])

  const handleNodeClick = useCallback((node: object) => {
    setDrawerNode(node as GraphFullNode)
  }, [])

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

  if (!payload) return null

  return (
    <Box>
      {/* Stats header */}
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {payload.stats.node_count} nodes · {payload.stats.edge_count} edges
      </Typography>

      {/* Controls */}
      <Stack spacing={1} sx={{ mb: 2 }}>
        {/* Node-type toggle chips */}
        <Stack direction="row" spacing={1} flexWrap="wrap">
          {KNOWN_TYPES.map((type) => (
            <Chip
              key={type}
              label={type}
              onClick={() => toggleType(type)}
              variant={activeTypes.has(type) ? 'filled' : 'outlined'}
              size="small"
              sx={{
                bgcolor: activeTypes.has(type) ? NODE_COLORS[type] : undefined,
                color: activeTypes.has(type) ? '#000' : undefined,
              }}
            />
          ))}
        </Stack>

        {/* Project filter */}
        {projectOptions.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 200, maxWidth: 320 }}>
            <InputLabel id="graph-project-label">Project</InputLabel>
            <Select
              labelId="graph-project-label"
              label="Project"
              value={projectFilter}
              onChange={(e) => setProjectFilter(e.target.value)}
            >
              <MenuItem value="">All projects</MenuItem>
              {projectOptions.map((p) => (
                <MenuItem key={p} value={p}>
                  {p}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
      </Stack>

      {/* Canvas */}
      <Box ref={containerRef} sx={{ width: '100%', border: 1, borderColor: 'divider', borderRadius: 1 }}>
        <ForceGraph2D
          graphData={graphData}
          width={width}
          height={Math.min(600, Math.max(400, window.innerHeight - 280))}
          nodeVal={(n) => (n as GraphFullNode).val}
          nodeColor={(n) => NODE_COLORS[(n as GraphFullNode).type] ?? FALLBACK_COLOR}
          nodeLabel={(n) => {
            const node = n as GraphFullNode
            return `${node.id}: ${node.label} (${node.type})`
          }}
          cooldownTicks={200}
          onNodeClick={handleNodeClick}
        />
      </Box>

      {/* Node detail drawer */}
      <Drawer
        anchor="right"
        open={drawerNode !== null}
        onClose={() => setDrawerNode(null)}
        PaperProps={{ sx: { width: DRAWER_WIDTH, p: 2 } }}
      >
        {drawerNode && (
          <NodeDrawerContent
            node={drawerNode}
            onClose={() => setDrawerNode(null)}
            onNavigate={navigate}
          />
        )}
      </Drawer>
    </Box>
  )
}

function NodeDrawerContent({
  node,
  onClose,
  onNavigate,
}: {
  node: GraphFullNode
  onClose: () => void
  onNavigate: (path: string) => void
}) {
  const isEntryLink =
    node.type === 'entry' && /^kb-\d{5}$/.test(node.id)

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h6" noWrap>
          Node detail
        </Typography>
        <Button size="small" onClick={onClose}>
          Close
        </Button>
      </Stack>

      <Stack spacing={0.5} sx={{ mb: 2 }}>
        <Typography variant="body2"><strong>ID:</strong> {node.id}</Typography>
        <Typography variant="body2"><strong>Label:</strong> {node.label}</Typography>
        <Typography variant="body2"><strong>Type:</strong> {node.type}</Typography>
        <Typography variant="body2"><strong>Val:</strong> {node.val}</Typography>
      </Stack>

      {Object.keys(node.properties).length > 0 && (
        <>
          <Divider sx={{ mb: 1 }} />
          <Typography variant="subtitle2" gutterBottom>
            Properties
          </Typography>
          <Stack spacing={0.5}>
            {Object.entries(node.properties).map(([k, v]) => (
              <Typography key={k} variant="body2">
                <strong>{k}:</strong>{' '}
                {typeof v === 'object' ? JSON.stringify(v) : String(v ?? '')}
              </Typography>
            ))}
          </Stack>
        </>
      )}

      {isEntryLink && (
        <Box sx={{ mt: 2 }}>
          <Button
            variant="contained"
            size="small"
            onClick={() => {
              onClose()
              onNavigate(`/entries/${node.id}`)
            }}
          >
            Open entry
          </Button>
        </Box>
      )}
    </Box>
  )
}
