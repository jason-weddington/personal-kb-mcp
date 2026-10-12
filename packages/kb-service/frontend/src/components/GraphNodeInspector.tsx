// The graph node inspector: one non-modal panel for every rendered node on
// the Graph page, and the shared grouped-connections body the entry drawer
// variant reuses. All member metadata comes from the caller — the Graph page
// feeds the loaded graph payload, the drawer variant the neighbors API — so
// the connections section itself never fetches.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { getEntries } from '../api'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import { MarkdownBody } from './ResponseCard'
import { ENTRY_TYPE_COLORS, entryTypeLabel } from '../entryTypes'
import type { ChipColor } from '../entryTypes'
import { BACKGROUND, INSPECTOR_W, nodeColor } from '../graph/explorerModel'
import type { ExplorerNode } from '../graph/explorerModel'
import {
  FACET_TOP_N,
  INSPECTOR_PILL_MAX,
  computeFacets,
  entrySummary,
  memberMatches,
} from '../graph/inspectorModel'
import type { ConnectionGroup, ConnInput, Facet, GroupFilter } from '../graph/inspectorModel'
import type { GetEntryResult, KnowledgeEntry, PointerRotTarget } from '../kbTypes'

/** ── View state (lifted to GraphExplorer so history steps can snapshot it) ── */

export type InspectorViewState = {
  /** Group expand state by group key; overrides the size-based default. */
  expanded: Record<string, boolean>
  /** Per-group filters (item C populates these; this item renders none). */
  filters: Record<string, { text: string; facets: string[] }>
  /** The connections scroll container's scrollTop. */
  scrollTop: number
}

export const DEFAULT_INSPECTOR_VIEW: InspectorViewState = {
  expanded: {},
  filters: {},
  scrollTop: 0,
}

/** ── Members ─────────────────────────────────────────────────────────────── */

export type InspectorMember = {
  nodeId: string
  label: string
  nodeType: string
  edgeType: string
  dir: 'in' | 'out'
  tooltip?: string
  entryType: string | null
  projectRef: string | null
  /** The graph-variant member builder always sets it; the drawer variant does not. */
  longTitle?: string | null
}

// ENTRY_TYPE_COLORS is total over EntryType, so the 'info' fallback is dead
// code for payload-valid entry types — but a bad value must not crash (the
// theme has no 'default' palette key to index).
const dotPaletteColor = (entryType: string): Exclude<ChipColor, 'default'> =>
  (ENTRY_TYPE_COLORS as Partial<Record<string, Exclude<ChipColor, 'default'>>>)[entryType] ??
  'info'

/**
 * The Graph-page member builder: metadata comes straight from the loaded node
 * payload, so the connections section never fetches. `longTitle` is always
 * set (null when the payload carries none) because the group title filter
 * matches over it.
 */
export function memberFromNode(node: ExplorerNode, conn: ConnInput): InspectorMember {
  const p = (node.properties ?? {}) as Record<string, unknown>
  const longTitle = typeof p.long_title === 'string' && p.long_title.length > 0 ? p.long_title : undefined
  return {
    nodeId: node.id,
    label: node.label || node.id,
    nodeType: node.type,
    edgeType: conn.edgeType,
    dir: conn.dir,
    tooltip: longTitle,
    entryType: node.type === 'entry' && typeof p.entry_type === 'string' ? p.entry_type : null,
    projectRef: typeof p.project_ref === 'string' ? p.project_ref : null,
    longTitle: typeof p.long_title === 'string' ? p.long_title : null,
  }
}

function hoverable(onHover: ((id: string | null) => void) | undefined, nodeId: string) {
  return {
    onMouseOver: () => onHover?.(nodeId),
    onMouseOut: () => onHover?.(null),
  }
}

/**
 * Wrap a member in an MUI Tooltip only when its tooltip is a non-empty
 * string. The Tooltip's child is a span so the member keeps its own
 * accessible name (Tooltip would otherwise stamp its title as aria-label
 * onto the member itself).
 */
function maybeTooltip(member: InspectorMember, element: ReactNode) {
  return member.tooltip !== undefined && member.tooltip.length > 0 ? (
    <Tooltip title={member.tooltip}>
      <Box component="span" sx={{ display: 'block', maxWidth: '100%' }}>
        {element}
      </Box>
    </Tooltip>
  ) : (
    element
  )
}

function MemberRow({
  member,
  group,
  onOpenNode,
  onHover,
}: {
  member: InspectorMember
  group: ConnectionGroup
  onOpenNode: (id: string) => void
  onHover: ((id: string | null) => void) | undefined
}) {
  const m: InspectorMember = { ...member, edgeType: group.edgeType, dir: group.dir }
  const isEntry = m.nodeType === 'entry'
  const body = isEntry ? (
    <Box sx={{ minWidth: 0, flex: 1, textAlign: 'left' }}>
      <Typography
        component="div"
        sx={{ fontWeight: 'bold', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
      >
        {m.label}
      </Typography>
      {m.tooltip ? (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {m.tooltip}
        </Typography>
      ) : null}
      {typeof m.projectRef === 'string' && m.projectRef.length > 0 ? (
        <Typography variant="caption" color="text.secondary">
          {m.projectRef}
        </Typography>
      ) : null}
    </Box>
  ) : (
    <Box sx={{ minWidth: 0, flex: 1, textAlign: 'left', wordBreak: 'break-word' }}>
      {m.label}
    </Box>
  )
  const dot =
    isEntry && m.entryType !== null ? (
      <Box
        aria-hidden
        sx={{
          width: 10,
          height: 10,
          borderRadius: '50%',
          flexShrink: 0,
          mt: '4px',
          bgcolor: (t) => t.palette[dotPaletteColor(m.entryType ?? '')].main,
        }}
      />
    ) : null
  const typeChip = !isEntry ? (
    <Chip
      size="small"
      label={m.nodeType}
      sx={{ bgcolor: nodeColor(m.nodeType), color: '#000', flexShrink: 0 }}
    />
  ) : null
  const row = (
    <Box
      component="button"
      type="button"
      onClick={() => onOpenNode(m.nodeId)}
      {...hoverable(onHover, m.nodeId)}
      sx={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 1,
        width: '100%',
        background: 'none',
        border: 0,
        borderBottom: 1,
        borderColor: 'divider',
        px: 0,
        py: 0.75,
        color: 'inherit',
        font: 'inherit',
        cursor: 'pointer',
        '&:hover': { bgcolor: 'action.hover' },
      }}
    >
      {dot}
      {body}
      {typeChip}
    </Box>
  )
  return maybeTooltip(m, row)
}

function MemberPill({
  member,
  group,
  onOpenNode,
  onHover,
}: {
  member: InspectorMember
  group: ConnectionGroup
  onOpenNode: (id: string) => void
  onHover: ((id: string | null) => void) | undefined
}) {
  const m: InspectorMember = { ...member, edgeType: group.edgeType, dir: group.dir }
  const chip = (
    <Chip
      label={m.label}
      size="small"
      clickable
      onClick={() => onOpenNode(m.nodeId)}
      {...hoverable(onHover, m.nodeId)}
    />
  )
  return maybeTooltip(m, chip)
}

/** ── Facet chips (item C: narrowing large connection groups) ─────────────── */

/**
 * One facet chip: exactly `${label} ${count}` text, the legend colour of its
 * node type (the entry-type palette colour for entry types), and an
 * aria-pressed selected state. Clicking toggles the facet; it never walks to
 * the facet's node.
 */
function FacetChip({
  facet,
  selected,
  onToggle,
}: {
  facet: Facet
  selected: boolean
  onToggle: () => void
}) {
  const label = `${facet.label} ${facet.count}`
  if (facet.kind === 'entry_type') {
    return (
      <Chip
        size="small"
        clickable
        label={label}
        aria-pressed={selected}
        // ENTRY_TYPE_COLORS is total over EntryType; Facet.value is string, so
        // the cast (and the 'info' fallback) are the same shape Search.tsx
        // uses for a bad value that must not crash.
        color={(ENTRY_TYPE_COLORS as Record<string, ChipColor>)[facet.value] ?? 'info'}
        onClick={onToggle}
      />
    )
  }
  const bg = facet.kind === 'project' ? nodeColor('project') : nodeColor(facet.nodeType ?? '')
  return (
    <Chip
      size="small"
      clickable
      label={label}
      aria-pressed={selected}
      onClick={onToggle}
      sx={{ bgcolor: bg, color: BACKGROUND }}
    />
  )
}

const EMPTY_GROUP_FILTER: GroupFilter = { text: '', facets: [] }

/**
 * The filter wiring the Graph page passes in: the lifted view state, the
 * loaded graph's nodes and adjacency (from which computeFacets' inputs are
 * derived), and the inspected node id (excluded from the co-occurring-node
 * facets). The entry drawer variant passes none of it and gets no filter UI.
 */
export type InspectorFiltering = {
  viewState: InspectorViewState
  onViewChange: (next: InspectorViewState) => void
  nodeById: ReadonlyMap<string, { id: string; label: string; type: string }>
  adjacency: ReadonlyMap<string, ReadonlyArray<{ node: string; type: string }>>
  inspectedNodeId: string
}

/** Write one group's filter and nothing else into the lifted view state. */
function writeGroupFilter(filtering: InspectorFiltering, groupKey: string, next: GroupFilter) {
  const { viewState, onViewChange } = filtering
  onViewChange({ ...viewState, filters: { ...viewState.filters, [groupKey]: next } })
}

/** ── The shared grouped connections body ─────────────────────────────────── */

export function InspectorConnections({
  groups,
  membersById,
  onOpenNode,
  onHover,
  view,
  onViewChange,
  filtering,
  revealedFacetGroups,
  onRevealFacetGroup,
}: {
  groups: ReadonlyArray<ConnectionGroup>
  membersById: ReadonlyMap<string, InspectorMember>
  onOpenNode: (id: string) => void
  /** Writes the member node id / null into the caller's hover state. */
  onHover?: (id: string | null) => void
  /** Controlled view state (Graph page); uncontrolled when omitted (drawer). */
  view?: InspectorViewState
  onViewChange?: (view: InspectorViewState) => void
  /** Present only on the Graph page: per-group filter wiring (item C). */
  filtering?: InspectorFiltering
  /** Group keys whose '+N' facet reveal is open (local to GraphNodeInspector). */
  revealedFacetGroups?: ReadonlySet<string>
  onRevealFacetGroup?: (groupKey: string) => void
}) {
  const [localExpanded, setLocalExpanded] = useState<Record<string, boolean>>({})
  const warnedRef = useRef(new Set<string>())
  const missing = useMemo(() => {
    const out: Array<{ groupKey: string; nodeId: string }> = []
    for (const g of groups) {
      for (const m of g.members) {
        if (!membersById.has(m.nodeId)) out.push({ groupKey: g.key, nodeId: m.nodeId })
      }
    }
    return out
  }, [groups, membersById])
  useEffect(() => {
    const fresh = missing.filter((m) => !warnedRef.current.has(`${m.groupKey}\u0000${m.nodeId}`))
    if (fresh.length === 0) return
    for (const m of fresh) {
      warnedRef.current.add(`${m.groupKey}\u0000${m.nodeId}`)
      console.warn('[graph-inspector] member not in payload', {
        group_key: m.groupKey,
        node_id: m.nodeId,
      })
    }
  }, [missing])
  // computeFacets' inputs, derived once per mounted target from the loaded
  // graph (the pinned Neighbor {node, type} → {id, type} mapping is explicit
  // because adjacency values carry the edge type and direction).
  const facetInputs = useMemo(() => {
    if (filtering === undefined) return null
    const members = groups.flatMap((g) => g.members)
    const neighborByMember = new Map(
      members.map((m) => [
        m.nodeId,
        (filtering.adjacency.get(m.nodeId) ?? []).map(({ node, type }) => ({ id: node, type })),
      ]),
    )
    const nodeTypeById = new Map<string, string>()
    const nodeLabelById = new Map<string, string>()
    for (const n of filtering.nodeById.values()) {
      nodeTypeById.set(n.id, n.type)
      nodeLabelById.set(n.id, n.label || n.id)
    }
    return { neighborByMember, nodeTypeById, nodeLabelById }
  }, [filtering, groups])
  const neighborIdsOf = useCallback(
    (nodeId: string) =>
      new Set((facetInputs?.neighborByMember.get(nodeId) ?? []).map((n) => n.id)),
    [facetInputs],
  )
  const expandedFor = (g: ConnectionGroup): boolean => {
    const override = view !== undefined ? view.expanded[g.key] : localExpanded[g.key]
    if (override !== undefined) return override
    return g.members.length <= INSPECTOR_PILL_MAX
  }
  const toggleGroup = (g: ConnectionGroup, expanded: boolean) => {
    if (view !== undefined && onViewChange !== undefined) {
      onViewChange({ ...view, expanded: { ...view.expanded, [g.key]: expanded } })
      return
    }
    setLocalExpanded((prev) => ({ ...prev, [g.key]: expanded }))
  }

  const resolve = (g: ConnectionGroup, nodeId: string): InspectorMember => {
    const member = membersById.get(nodeId)
    if (member !== undefined) return member
    // the missing-member warn fires from the effect above
    return {
      nodeId,
      label: nodeId,
      nodeType: 'unknown',
      edgeType: g.edgeType,
      dir: g.dir,
      entryType: null,
      projectRef: null,
    }
  }

  const arrow = (dir: 'in' | 'out') => (dir === 'out' ? '→' : '←')

  return (
    <Box>
      {groups.map((g) => {
        const count = g.members.length
        if (count <= INSPECTOR_PILL_MAX) {
          return (
            <Box key={g.key} sx={{ mb: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                {`${g.edgeType} ${arrow(g.dir)} ${count}`}
              </Typography>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                {g.members.map((m) => (
                  <MemberPill
                    key={m.nodeId}
                    member={resolve(g, m.nodeId)}
                    group={g}
                    onOpenNode={onOpenNode}
                    onHover={onHover}
                  />
                ))}
              </Stack>
            </Box>
          )
        }
        // Large group with the filter wiring: a title filter box and a facet
        // chip row above the rows (item C). Without `filtering` (the drawer
        // variant) this branch renders exactly the item A output.
        const resolved = g.members.map((m) => resolve(g, m.nodeId))
        const filter =
          filtering !== undefined ? (filtering.viewState.filters[g.key] ?? EMPTY_GROUP_FILTER) : EMPTY_GROUP_FILTER
        const active = filter.text.trim().length > 0 || filter.facets.length > 0
        const facets =
          filtering !== undefined && facetInputs !== null
            ? computeFacets({
                inspectedNodeId: filtering.inspectedNodeId,
                members: resolved,
                neighborByMember: facetInputs.neighborByMember,
                nodeTypeById: facetInputs.nodeTypeById,
                nodeLabelById: facetInputs.nodeLabelById,
              })
            : []
        const shown = active
          ? resolved.filter((m) => memberMatches(m, filter, neighborIdsOf(m.nodeId)))
          : resolved
        const revealed = revealedFacetGroups?.has(g.key) ?? false
        const visibleFacets = revealed ? facets : facets.slice(0, FACET_TOP_N)
        return (
          <Accordion
            key={g.key}
            expanded={expandedFor(g)}
            onChange={(_, expanded) => toggleGroup(g, expanded)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" component="span">
                {`${g.key} ${arrow(g.dir)} ${count}`}
              </Typography>
              {filtering !== undefined && active && (
                <>
                  <Typography variant="subtitle2" component="span" sx={{ ml: 1 }}>
                    {`${shown.length} of ${count}`}
                  </Typography>
                  {/* A Chip, not a Button: AccordionSummary renders a native
                      <button> in MUI v7, and a nested native button is invalid
                      DOM (a hydration error). The clickable Chip is a div
                      with role="button" and the same keyboard handling, and
                      stopPropagation keeps the accordion closed on clear. */}
                  <Chip
                    size="small"
                    clickable
                    label="Clear"
                    data-testid={`group-clear-${g.key}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (filtering !== undefined) writeGroupFilter(filtering, g.key, { text: '', facets: [] })
                    }}
                    sx={{ ml: 'auto', flexShrink: 0 }}
                  />
                </>
              )}
            </AccordionSummary>
            <AccordionDetails sx={{ py: 0, maxHeight: 320, overflowY: 'auto' }}>
              {filtering !== undefined && (
                <Box sx={{ mb: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <input
                    data-testid={`group-filter-${g.key}`}
                    type="text"
                    value={filter.text}
                    aria-label={`Filter ${g.edgeType} by title`}
                    placeholder="Filter by title"
                    onChange={(e) =>
                      writeGroupFilter(filtering, g.key, { text: e.target.value, facets: filter.facets })
                    }
                    style={{
                      width: '100%',
                      padding: '4px 8px',
                      borderRadius: 4,
                      border: '1px solid rgba(128,128,128,0.5)',
                      background: 'transparent',
                      color: 'inherit',
                      font: 'inherit',
                    }}
                  />
                  <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                    {visibleFacets.map((facet) => (
                      <FacetChip
                        key={facet.key}
                        facet={facet}
                        selected={filter.facets.includes(facet.key)}
                        onToggle={() =>
                          writeGroupFilter(filtering, g.key, {
                            text: filter.text,
                            facets: filter.facets.includes(facet.key)
                              ? filter.facets.filter((k) => k !== facet.key)
                              : [...filter.facets, facet.key],
                          })
                        }
                      />
                    ))}
                    {!revealed && facets.length > FACET_TOP_N && (
                      <Chip
                        size="small"
                        clickable
                        label={`+${facets.length - FACET_TOP_N}`}
                        onClick={() => onRevealFacetGroup?.(g.key)}
                      />
                    )}
                  </Stack>
                </Box>
              )}
              {shown.map((m) => (
                <MemberRow
                  key={m.nodeId}
                  member={m}
                  group={g}
                  onOpenNode={onOpenNode}
                  onHover={onHover}
                />
              ))}
              {filtering !== undefined && active && shown.length === 0 && (
                <Typography data-testid={`inspector-empty-${g.key}`} variant="body2" sx={{ py: 1 }}>
                  No matching members.
                </Typography>
              )}
            </AccordionDetails>
          </Accordion>
        )
      })}
    </Box>
  )
}

/** ── Details fetch (Graph page entry variant) ────────────────────────────── */

type DetailsState = {
  id: string
  status: 'loading' | 'ok' | 'failed'
  entry: KnowledgeEntry | null
  pointerRot: PointerRotTarget[]
  error: string | null
}

function formatUpdated(iso: string | null): string | null {
  if (!iso) return null
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

/** ── The inspector panel ─────────────────────────────────────────────────── */

export type GraphNodeInspectorProps = {
  node: ExplorerNode
  /** Grouped connections of the target, from the loaded graph payload. */
  groups: ReadonlyArray<ConnectionGroup>
  nodeById: ReadonlyMap<string, ExplorerNode>
  /**
   * The loaded graph's adjacency, keyed by node id (Neighbor values carry the
   * edge type). Present only on the Graph page; it enables the per-group
   * filter UI by feeding computeFacets.
   */
  adjacency?: ReadonlyMap<string, ReadonlyArray<{ node: string; type: string }>>
  view: InspectorViewState
  onViewChange: (view: InspectorViewState) => void
  /** The select path a canvas click on that node takes. */
  onOpenNode: (id: string) => void
  /** The shared hover setter (writes the canvas hover render state). */
  onHover: (id: string | null) => void
  onClose: () => void
}

export function GraphNodeInspector({
  node,
  groups,
  nodeById,
  adjacency,
  view,
  onViewChange,
  onOpenNode,
  onHover,
  onClose,
}: GraphNodeInspectorProps) {
  const { openNode } = useEntryDrawer()
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const isEntry = node.type === 'entry'
  const props = (node.properties ?? {}) as Record<string, unknown>
  const longTitle = typeof props.long_title === 'string' ? props.long_title : ''
  const entryType = isEntry && typeof props.entry_type === 'string' ? props.entry_type : null

  // Target-scoped state resets in the render that changes the target (the
  // parent resets the lifted view alongside setting the new target), never
  // in an effect, so a restored snapshot survives.
  const [targetId, setTargetId] = useState(node.id)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [details, setDetails] = useState<DetailsState | null>(null)
  // The '+N' facet reveal is deliberately NOT in InspectorViewState (nothing
  // persisted): local state that resets with the target like the rest.
  const [revealedFacetGroups, setRevealedFacetGroups] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  )
  if (targetId !== node.id) {
    setTargetId(node.id)
    setDetailsOpen(false)
    setDetails(null)
    setRevealedFacetGroups(new Set())
  }
  const shown = details !== null && details.id === node.id ? details : null

  const loadDetails = (id: string) => {
    // one fetch per shown id, even across Details collapse/re-expand
    if (details !== null && details.id === id) return
    setDetails({ id, status: 'loading', entry: null, pointerRot: [], error: null })
    void getEntries([id], { touch: false })
      .then((resp) => {
        const r: GetEntryResult | undefined = resp.results[0]
        if (r === undefined || !r.found || r.entry === null) {
          console.warn('[graph-inspector] details fetch failed', { node_id: id, error: 'not found' })
          setDetails({ id, status: 'failed', entry: null, pointerRot: [], error: 'not found' })
          return
        }
        setDetails({
          id,
          status: 'ok',
          entry: r.entry,
          pointerRot: r.pointer_rot ?? [],
          error: null,
        })
      })
      .catch((err: unknown) => {
        const error = err instanceof Error ? err.message : String(err)
        console.warn('[graph-inspector] details fetch failed', { node_id: id, error })
        setDetails({ id, status: 'failed', entry: null, pointerRot: [], error })
      })
  }

  // Restore a scrollTop that changed from outside (a history snapshot or the
  // reset on re-target); reporting it back through onScroll is idempotent.
  useEffect(() => {
    const el = scrollRef.current
    if (el !== null) el.scrollTop = view.scrollTop
  }, [node.id, view.scrollTop])

  const membersById = useMemo(() => {
    const byNode = new Map<string, InspectorMember>()
    for (const g of groups) {
      for (const m of g.members) {
        if (byNode.has(m.nodeId)) continue
        const n = nodeById.get(m.nodeId)
        if (n === undefined) continue // InspectorConnections warns and falls back
        byNode.set(m.nodeId, memberFromNode(n, { nodeId: m.nodeId, edgeType: g.edgeType, dir: g.dir }))
      }
    }
    return byNode
  }, [groups, nodeById])

  const tagGroup = groups.find((g) => g.key === 'has_tag:out') ?? null
  const projectGroup = groups.find((g) => g.key === 'in_project:out') ?? null
  const connGroups = isEntry
    ? groups.filter((g) => g.key !== 'has_tag:out' && g.key !== 'in_project:out')
    : groups
  const summary = isEntry
    ? ''
    : entrySummary(
        connGroups.flatMap((g) =>
          g.members.flatMap((m) => {
            const member = membersById.get(m.nodeId)
            return member !== undefined &&
              member.nodeType === 'entry' &&
              member.entryType !== null
              ? [member.entryType]
              : []
          }),
        ),
      )

  const metaParts: string[] = []
  if (isEntry) {
    const projectRef = typeof props.project_ref === 'string' ? props.project_ref : ''
    if (projectRef.length > 0) metaParts.push(projectRef)
    if (typeof props.confidence_level === 'number') {
      metaParts.push(props.confidence_level.toFixed(2))
    }
    const updated = formatUpdated(shown?.entry?.updated_at ?? null)
    if (updated !== null) metaParts.push(`updated ${updated}`)
  }

  return (
    <Paper
      ref={scrollRef}
      role="region"
      aria-label="Node inspector"
      elevation={8}
      onScroll={(e) =>
        onViewChange({ ...view, scrollTop: (e.currentTarget as HTMLDivElement).scrollTop })
      }
      sx={{
        width: { xs: '100%', sm: INSPECTOR_W },
        maxWidth: '100%',
        flexShrink: 1,
        minHeight: 0,
        overflowY: 'auto',
        p: 2,
      }}
    >
      <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
        {isEntry && entryType !== null ? (
          <Chip
            size="small"
            color={dotPaletteColor(entryType)}
            label={entryTypeLabel(entryType)}
            sx={{ mb: 1 }}
          />
        ) : (
          <Chip
            size="small"
            label={node.type}
            sx={{ bgcolor: nodeColor(node.type), color: '#000', mb: 1 }}
          />
        )}
        <IconButton size="small" aria-label="Close inspector" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Stack>
      <Typography variant="h6" sx={{ wordBreak: 'break-word' }}>
        {isEntry ? node.label : node.label || node.id}
      </Typography>
      {isEntry ? (
        <>
          {longTitle.length > 0 && (
            <Typography variant="body2" color="text.secondary">
              {longTitle}
            </Typography>
          )}
          {metaParts.length > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {metaParts.join(' · ')}
            </Typography>
          )}
          {tagGroup !== null && tagGroup.members.length > 0 && (
            <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
              {tagGroup.members.map((m) => (
                <MemberPill
                  key={m.nodeId}
                  member={resolveMember(membersById, tagGroup, m.nodeId)}
                  group={tagGroup}
                  onOpenNode={onOpenNode}
                  onHover={onHover}
                />
              ))}
            </Stack>
          )}
          {projectGroup !== null && projectGroup.members.length > 0 && (
            <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
              {projectGroup.members.map((m) => (
                <MemberPill
                  key={m.nodeId}
                  member={resolveMember(membersById, projectGroup, m.nodeId)}
                  group={projectGroup}
                  onOpenNode={onOpenNode}
                  onHover={onHover}
                />
              ))}
            </Stack>
          )}
        </>
      ) : (
        <>
          <Typography variant="body2" color="text.secondary">
            {node.id}
          </Typography>
          {summary.length > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {summary}
            </Typography>
          )}
        </>
      )}

      {connGroups.length > 0 && (
        <>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 1 }} />
          <InspectorConnections
            key={node.id}
            groups={connGroups}
            membersById={membersById}
            onOpenNode={onOpenNode}
            onHover={onHover}
            view={view}
            onViewChange={onViewChange}
            filtering={
              adjacency === undefined
                ? undefined
                : {
                    viewState: view,
                    onViewChange,
                    nodeById,
                    adjacency,
                    inspectedNodeId: node.id,
                  }
            }
            revealedFacetGroups={revealedFacetGroups}
            onRevealFacetGroup={(groupKey) =>
              setRevealedFacetGroups(new Set([...revealedFacetGroups, groupKey]))
            }
          />
        </>
      )}

      {isEntry && (
        <Accordion expanded={detailsOpen} onChange={(_, open) => {
          setDetailsOpen(open)
          if (open) loadDetails(node.id)
        }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" component="span">
              Details
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            {shown === null || shown.status === 'loading' ? (
              <CircularProgress size={20} />
            ) : shown.status === 'failed' ? (
              <Alert severity="warning">Details unavailable ({shown.error})</Alert>
            ) : shown.entry !== null ? (
              <>
                <MarkdownBody>{shown.entry.knowledge_details}</MarkdownBody>
                {shown.entry.superseded_by !== null && (
                  <Button
                    size="small"
                    onClick={() => openNode(shown.entry?.superseded_by ?? '')}
                    sx={{ mt: 1 }}
                  >
                    {`Superseded by ${shown.entry.superseded_by}`}
                  </Button>
                )}
                {shown.pointerRot.map((pr) => (
                  <Alert key={pr.target_id} severity="warning" sx={{ mt: 1 }}>
                    {pr.superseded_by !== null
                      ? `${pr.target_id} superseded by ${pr.superseded_by}`
                      : `${pr.target_id} deactivated`}
                  </Alert>
                ))}
              </>
            ) : null}
          </AccordionDetails>
        </Accordion>
      )}
    </Paper>
  )
}

function resolveMember(
  membersById: ReadonlyMap<string, InspectorMember>,
  group: ConnectionGroup,
  nodeId: string,
): InspectorMember {
  const member = membersById.get(nodeId)
  if (member !== undefined) return member
  return {
    nodeId,
    label: nodeId,
    nodeType: 'unknown',
    edgeType: group.edgeType,
    dir: group.dir,
    entryType: null,
    projectRef: null,
  }
}
