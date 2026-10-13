// The graph node inspector: one non-modal panel for every rendered node on
// the Graph page, and the shared grouped-connections body the entry drawer
// variant reuses. All member metadata comes from the caller — the Graph page
// feeds the loaded graph payload, the drawer variant the neighbors API — so
// the connections section itself never fetches.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
import Typography from '@mui/material/Typography'
import CheckIcon from '@mui/icons-material/Check'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { getEntries } from '../api'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import { MarkdownBody } from './ResponseCard'
import { toggleChipSx } from './chipState'
import { ENTRY_TYPE_COLORS, entryTypeLabel } from '../entryTypes'
import type { ChipColor } from '../entryTypes'
import { BACKGROUND, INSPECTOR_W, NODE_TYPES, nodeColor } from '../graph/explorerModel'
import type { ExplorerNode } from '../graph/explorerModel'
import {
  FACET_ROW_TOP_N,
  INSPECTOR_GROUP_OPEN_MAX,
  computeFacets,
  entrySummary,
  facetRows,
  memberMatches,
} from '../graph/inspectorModel'
import type { ConnectionGroup, ConnInput, Facet, GroupFilter } from '../graph/inspectorModel'
import type { GetEntryResult, KnowledgeEntry, PointerRotTarget } from '../kbTypes'

/** ── View state (lifted to GraphExplorer so history steps can snapshot it) ── */

export type InspectorViewState = {
  /** Group expand state by group key; overrides the size-based default. */
  expanded: Record<string, boolean>
  /** Per-group filters (item C populates these; this item renders none). */
  filters: Record<string, { text: string; hidden: string[] }>
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
 * One summary row: entry members render the bold one-line short title, the
 * long title clamped to two lines, the entry-type dot and the project ref;
 * non-entry members their label plus a node-type chip. Hovering keeps
 * highlighting the node on the canvas; no Tooltip — the hover tooltip left
 * the panel and drifted while scrolling.
 */
function MemberRow({
  member,
  group,
  onOpenNode,
  onHover,
  visited,
}: {
  member: InspectorMember
  group: ConnectionGroup
  onOpenNode: (id: string) => void
  onHover: ((id: string | null) => void) | undefined
  /** The member's node was focused earlier this page session. */
  visited?: boolean
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
        // The divider separates rows; the group's card edge closes the last one.
        '&:last-of-type': { borderBottom: 0 },
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
      {visited ? (
        <CheckIcon data-testid="visited-check" fontSize="small" sx={{ flexShrink: 0, mt: '2px' }} />
      ) : null}
    </Box>
  )
  return row
}

/**
 * The entry-header tag and project chips (the only remaining pill variant):
 * small, pinned to one height, never a check and never a Tooltip, still
 * walking to the member node on click.
 */
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
  return (
    <Chip
      label={m.label}
      size="small"
      clickable
      onClick={() => onOpenNode(m.nodeId)}
      sx={{ height: 24 }}
      {...hoverable(onHover, m.nodeId)}
    />
  )
}

/** ── Facet chips (item C: narrowing large connection groups) ─────────────── */

/**
 * One facet chip: exactly `${label} ${count}` text, the legend colour of its
 * node type (the entry-type palette colour for entry types), and an
 * aria-pressed state that means the facet is OFF/hidden. All facets start
 * ON (filled, legend colour); clicking one switches it OFF — grey and
 * outlined through the shared toggleChipSx — which hides every member it
 * matches. It never walks to the facet's node.
 */
function FacetChip({
  facet,
  hidden,
  onToggle,
}: {
  facet: Facet
  hidden: boolean
  onToggle: () => void
}) {
  const label = `${facet.label} ${facet.count}`
  if (facet.kind === 'entry_type') {
    return (
      <Chip
        size="small"
        clickable
        label={label}
        aria-pressed={hidden}
        variant={hidden ? 'outlined' : 'filled'}
        // ENTRY_TYPE_COLORS is total over EntryType; Facet.value is string, so
        // the cast (and the 'info' fallback) are the same shape Search.tsx
        // uses for a bad value that must not crash.
        color={
          hidden ? undefined : ((ENTRY_TYPE_COLORS as Record<string, ChipColor>)[facet.value] ?? 'info')
        }
        onClick={onToggle}
        sx={hidden ? toggleChipSx(false) : undefined}
      />
    )
  }
  const bg = facet.kind === 'project' ? nodeColor('project') : nodeColor(facet.nodeType ?? '')
  return (
    <Chip
      size="small"
      clickable
      label={label}
      aria-pressed={hidden}
      variant={hidden ? 'outlined' : 'filled'}
      onClick={onToggle}
      sx={hidden ? toggleChipSx(false) : { bgcolor: bg, color: BACKGROUND }}
    />
  )
}

const EMPTY_GROUP_FILTER: GroupFilter = { text: '', hidden: [] }

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
  visited,
}: {
  groups: ReadonlyArray<ConnectionGroup>
  membersById: ReadonlyMap<string, InspectorMember>
  onOpenNode: (id: string) => void
  /** Node ids focused earlier this page session (visited check marks). */
  visited?: ReadonlySet<string>
  /** Writes the member node id / null into the caller's hover state. */
  onHover?: (id: string | null) => void
  /** Controlled view state (Graph page); uncontrolled when omitted (drawer). */
  view?: InspectorViewState
  onViewChange?: (view: InspectorViewState) => void
  /** Present only on the Graph page: per-group filter wiring (item C). */
  filtering?: InspectorFiltering
  /**
   * `${groupKey}:${rowLabel}` keys whose per-row '+N more' facet reveal is
   * open (local to GraphNodeInspector; nothing persisted).
   */
  revealedFacetGroups?: ReadonlySet<string>
  onRevealFacetGroup?: (rowKey: string) => void
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
    return g.members.length <= INSPECTOR_GROUP_OPEN_MAX
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
  const headerText = (g: ConnectionGroup, shownCount: number, active: boolean) =>
    `${g.edgeType.replace(/_/g, ' ')} ${arrow(g.dir)} ${
      active ? `${shownCount} of ${g.members.length}` : g.members.length
    }`

  return (
    <Box>
      {groups.map((g) => {
        const count = g.members.length
        // Every group is one accordion of summary rows with the same header
        // style; groups over the open threshold (with the filter wiring)
        // gain the filter area. Without `filtering` (the drawer variant)
        // this renders rows only, no filter UI.
        const resolved = g.members.map((m) => resolve(g, m.nodeId))
        const filter =
          filtering !== undefined
            ? (filtering.viewState.filters[g.key] ?? EMPTY_GROUP_FILTER)
            : EMPTY_GROUP_FILTER
        const active = filter.text.trim().length > 0 || filter.hidden.length > 0
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
        return (
          <Accordion
            key={g.key}
            expanded={expandedFor(g)}
            onChange={(_, expanded) => toggleGroup(g, expanded)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" component="span">
                {headerText(g, shown.length, active)}
              </Typography>
              {filtering !== undefined && active && (
                <>
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
                      if (filtering !== undefined)
                        writeGroupFilter(filtering, g.key, { text: '', hidden: [] })
                    }}
                    sx={{ ml: 'auto', flexShrink: 0 }}
                  />
                </>
              )}
            </AccordionSummary>
            <AccordionDetails sx={{ py: 0 }}>
              {filtering !== undefined && count > INSPECTOR_GROUP_OPEN_MAX && (
                <Box sx={{ mb: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <input
                    data-testid={`group-filter-${g.key}`}
                    type="text"
                    value={filter.text}
                    aria-label={`Filter ${g.edgeType} by title`}
                    placeholder="Filter by title"
                    onChange={(e) =>
                      writeGroupFilter(filtering, g.key, { text: e.target.value, hidden: filter.hidden })
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
                  {facetRows(facets, NODE_TYPES).map((row) => {
                    const rowKey = `${g.key}:${row.label}`
                    const revealed = revealedFacetGroups?.has(rowKey) ?? false
                    const visibleRowFacets = revealed ? row.facets : row.facets.slice(0, FACET_ROW_TOP_N)
                    return (
                      <Box key={row.label}>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {row.label}
                        </Typography>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                          {visibleRowFacets.map((facet) => (
                            <FacetChip
                              key={facet.key}
                              facet={facet}
                              hidden={filter.hidden.includes(facet.key)}
                              onToggle={() =>
                                writeGroupFilter(filtering, g.key, {
                                  text: filter.text,
                                  hidden: filter.hidden.includes(facet.key)
                                    ? filter.hidden.filter((k) => k !== facet.key)
                                    : [...filter.hidden, facet.key],
                                })
                              }
                            />
                          ))}
                          {!revealed && row.facets.length > FACET_ROW_TOP_N && (
                            <Button
                              size="small"
                              data-testid={`facets-more-${rowKey}`}
                              onClick={() => onRevealFacetGroup?.(rowKey)}
                              sx={{
                                minWidth: 0,
                                minHeight: 24,
                                px: 1,
                                py: 0,
                                fontSize: '0.75rem',
                                textTransform: 'none',
                                color: 'text.secondary',
                              }}
                            >
                              {`+${row.facets.length - FACET_ROW_TOP_N} more`}
                            </Button>
                          )}
                        </Stack>
                      </Box>
                    )
                  })}
                </Box>
              )}
              {shown.map((m) => (
                <MemberRow
                  key={m.nodeId}
                  member={m}
                  group={g}
                  onOpenNode={onOpenNode}
                  onHover={onHover}
                  visited={visited?.has(m.nodeId)}
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
  /** Node ids focused earlier this page session (visited check marks). */
  visited?: ReadonlySet<string>
  onViewChange: (view: InspectorViewState) => void
  /**
   * Live scroll position sink: when present, scrolling writes it and fires
   * no view-state update (the scrollTop feedback loop — a lagging
   * view.scrollTop written back through the restore effect — is why). The
   * history snapshot reads the ref at press time instead. Absent in the
   * drawer-less standalone renderers that never scroll from outside.
   */
  scrollTopRef?: { current: number }
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
  visited,
  onViewChange,
  scrollTopRef,
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
  const [detailsOpen, setDetailsOpen] = useState(true)
  const [details, setDetails] = useState<DetailsState | null>(null)
  // The per-row '+N more' facet reveal is deliberately NOT in
  // InspectorViewState (nothing persisted): local state that resets with
  // the target like the rest.
  const [revealedFacetGroups, setRevealedFacetGroups] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  )
  if (targetId !== node.id) {
    setTargetId(node.id)
    setDetailsOpen(true)
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

  // Details starts expanded and fetches the entry when the inspector opens:
  // once per shown id (the ref guard plus the details.id guard make
  // collapse/re-expand and repeat renders refetch nothing) and only for
  // entry targets — a non-entry inspector fetches nothing.
  const detailsFetchedForRef = useRef<string | null>(null)
  useEffect(() => {
    if (node.type !== 'entry') return
    if (detailsFetchedForRef.current === node.id) return
    detailsFetchedForRef.current = node.id
    console.info('[graph-inspector] details fetch', { node_id: node.id })
    loadDetails(node.id)
    // loadDetails re-creates each render; the ref guard keeps this once
    // per shown id.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node.id, loadDetails])

  // Restore a scrollTop that changed from outside (a history snapshot or the
  // reset on re-target). Deps keep [node.id, view.scrollTop]: live scrolling
  // no longer touches view state, so this fires only on re-target or when a
  // restored view carries a different scrollTop, and never mid-scroll.
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
      role="region"
      aria-label="Node inspector"
      elevation={8}
      sx={{
        width: { xs: '100%', sm: INSPECTOR_W },
        maxWidth: '100%',
        flexShrink: 1,
        flexGrow: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
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

      {connGroups.length > 0 && <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 1 }} />}
      <Box
        data-testid="inspector-scroll"
        ref={scrollRef}
        onScroll={(e) => {
          if (scrollTopRef === undefined) return
          // The live position is a ref, not React state: writing it into the
          // view state re-rendered the whole explorer mid-scroll and the
          // restore effect wrote a stale position back — a scroll bounce
          // feedback loop. History snapshots read the ref at press time.
          scrollTopRef.current = (e.currentTarget as HTMLDivElement).scrollTop
        }}
        sx={{ flex: 1, minHeight: 0, overflowY: 'auto' }}
      >
        {connGroups.length > 0 && (
          <InspectorConnections
            key={node.id}
            groups={connGroups}
            membersById={membersById}
            onOpenNode={onOpenNode}
            onHover={onHover}
            visited={visited}
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
            onRevealFacetGroup={(rowKey) =>
              setRevealedFacetGroups(new Set([...revealedFacetGroups, rowKey]))
            }
          />
        )}
        {isEntry && (
          <Accordion
            expanded={detailsOpen}
            onChange={(_, open) => {
              setDetailsOpen(open)
              if (open) loadDetails(node.id)
            }}
          >
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
      </Box>
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
