// Pure model for the graph node inspector: connection grouping keyed by
// `${edge_type}:${dir}` and the non-entry summary line. Shared by the Graph
// page inspector (payload-fed) and the entry drawer variant (neighbors API).
import { ENTRY_TYPES, entryTypeLabel } from '../entryTypes'
import type { EntryType } from '../kbTypes'

/** A single connection: the neighbour node plus the edge type and direction. */
export type ConnInput = { nodeId: string; edgeType: string; dir: 'in' | 'out' }

/** Connections grouped by `${edgeType}:${dir}`, in first-appearance order. */
export type ConnectionGroup = {
  key: string
  edgeType: string
  dir: 'in' | 'out'
  members: Array<{ nodeId: string }>
}

/**
 * A group with at most this many members defaults expanded; bigger groups
 * default collapsed and gain the filter area.
 */
export const INSPECTOR_GROUP_OPEN_MAX = 12

/** How many facet chips render per labelled row before its `+N more` button. */
export const FACET_ROW_TOP_N = 6

/**
 * The member shape the group filter model consumes. `InspectorMember`
 * (GraphNodeInspector.tsx) satisfies it; it is structural so this pure model
 * never imports the component module.
 */
export type FilterMember = {
  nodeId: string
  label: string
  longTitle?: string | null
  nodeType: string
  entryType: string | null
  projectRef: string | null
}

/** A per-group filter: a title substring plus the facet keys switched OFF (hidden). */
export type GroupFilter = { text: string; hidden: string[] }

/** One facet chip of a large connection group. */
export type Facet = {
  key: string
  kind: 'entry_type' | 'project' | 'node'
  value: string
  label: string
  count: number
  /** The facet node's type for kind 'node' — the key into nodeColor — else null. */
  nodeType: string | null
}

/** One labelled row of facet chips: the row label plus its full facet list. */
export type FacetRow = { label: string; facets: Facet[] }

/**
 * Does `member` match the single facet `key`? Facet keys: `entry_type:<t>`
 * (member is an entry of type <t>), `project:<ref>` (member's project ref),
 * `node:<id>` (the member's own id or one of its neighbour ids); any other
 * key shape matches nothing.
 */
function matchesFacet(
  member: FilterMember,
  key: string,
  memberNeighborIds: ReadonlySet<string>,
): boolean {
  const sep = key.indexOf(':')
  const kind = sep === -1 ? '' : key.slice(0, sep)
  const value = sep === -1 ? '' : key.slice(sep + 1)
  if (kind === 'entry_type') {
    return member.nodeType === 'entry' && member.entryType === value
  }
  if (kind === 'project') {
    return member.projectRef === value
  }
  if (kind === 'node') {
    return member.nodeId === value || memberNeighborIds.has(value)
  }
  return false
}

/**
 * Does `member` pass the group's filter? The text query (trimmed,
 * lowercased) must be a substring of the label plus long title, and the
 * member must match NO key in `filter.hidden` — a member matching any hidden
 * facet is excluded.
 */
export function memberMatches(
  member: FilterMember,
  filter: GroupFilter,
  memberNeighborIds: ReadonlySet<string>,
): boolean {
  const text = filter.text.trim().toLowerCase()
  if (text.length > 0) {
    const haystack = [member.label, member.longTitle ?? ''].join(' ').toLowerCase()
    if (!haystack.includes(text)) return false
  }
  for (const key of filter.hidden) {
    if (matchesFacet(member, key, memberNeighborIds)) return false
  }
  return true
}

/**
 * Facet chips for ONE group's members (never pooled across groups): the
 * members' entry types, project refs and co-occurring non-entry neighbours
 * (excluding the inspected node). Each facet's count is the number of the
 * group's members it matches. Sorted count descending, ties by label ascending.
 */
export function computeFacets(input: {
  inspectedNodeId: string
  members: ReadonlyArray<FilterMember>
  neighborByMember: ReadonlyMap<string, ReadonlyArray<{ id: string; type: string }>>
  nodeTypeById: ReadonlyMap<string, string>
  nodeLabelById: ReadonlyMap<string, string>
}): Facet[] {
  const { inspectedNodeId, members, neighborByMember, nodeTypeById, nodeLabelById } = input
  const seen = new Set<string>()
  const facets: Facet[] = []
  const push = (facet: Facet) => {
    if (seen.has(facet.key)) return
    seen.add(facet.key)
    facets.push(facet)
  }
  for (const m of members) {
    if (m.nodeType === 'entry' && m.entryType !== null) {
      push({
        key: `entry_type:${m.entryType}`,
        kind: 'entry_type',
        value: m.entryType,
        label: entryTypeLabel(m.entryType),
        count: 0,
        nodeType: null,
      })
    }
    if (m.projectRef !== null && m.projectRef.length > 0) {
      push({
        key: `project:${m.projectRef}`,
        kind: 'project',
        value: m.projectRef,
        label: m.projectRef,
        count: 0,
        nodeType: null,
      })
    }
    for (const n of neighborByMember.get(m.nodeId) ?? []) {
      if (n.id === inspectedNodeId) continue
      if (nodeTypeById.get(n.id) === 'entry') continue
      // A project-typed neighbour is covered by the member's own project
      // facet; offering both made two chips for the same project.
      if (nodeTypeById.get(n.id) === 'project') continue
      push({
        key: `node:${n.id}`,
        kind: 'node',
        value: n.id,
        label: nodeLabelById.get(n.id) ?? n.id,
        count: 0,
        nodeType: nodeTypeById.get(n.id) ?? null,
      })
    }
  }
  for (const facet of facets) {
    facet.count = members.reduce((sum, m) => {
      const neighborIds = new Set((neighborByMember.get(m.nodeId) ?? []).map((n) => n.id))
      return matchesFacet(m, facet.key, neighborIds) ? sum + 1 : sum
    }, 0)
  }
  // A facet matching every member would empty the list when switched off.
  const surviving = facets.filter((f) => f.count !== members.length)
  surviving.sort((a, b) => b.count - a.count || (a.label < b.label ? -1 : a.label > b.label ? 1 : 0))
  return surviving
}

/**
 * Group a group's facet chips into labelled rows so colour never carries
 * meaning on its own: a 'Type' row (entry types), a 'Project' row
 * (project_ref values), then one row per other neighbour node type present,
 * labelled with that type (first letter capitalised), in the caller-supplied
 * `nodeTypeOrder` (the Graph page passes the NODE_COLORS key order); a node
 * facet whose nodeType is null or not in the order lands in a trailing
 * 'Other' row. Each row carries its FULL facet list (already sorted count
 * descending, ties by label ascending by computeFacets) — the render side
 * cuts to FACET_ROW_TOP_N with a `+N more` reveal. A row with no facets is
 * omitted.
 */
export function facetRows(
  facets: ReadonlyArray<Facet>,
  nodeTypeOrder: ReadonlyArray<string>,
): FacetRow[] {
  const rows: FacetRow[] = []
  const push = (label: string, rowFacets: ReadonlyArray<Facet>) => {
    if (rowFacets.length === 0) return
    rows.push({ label, facets: [...rowFacets] })
  }
  push('Type', facets.filter((f) => f.kind === 'entry_type'))
  push('Project', facets.filter((f) => f.kind === 'project'))
  const nodeFacets = facets.filter((f) => f.kind === 'node')
  for (const t of nodeTypeOrder) {
    if (t === 'entry' || t === 'project') continue
    push(t.charAt(0).toUpperCase() + t.slice(1), nodeFacets.filter((f) => f.nodeType === t))
  }
  push(
    'Other',
    nodeFacets.filter((f) => f.nodeType === null || !nodeTypeOrder.includes(f.nodeType ?? '')),
  )
  return rows
}

/**
 * The nodeIds that are members of a group with an active filter (a non-empty
 * trimmed text or a non-empty hidden facet list) and that fail `matchMember`
 * for that group — the set the canvas subtracts from the visible focus
 * neighbourhood. The focused node itself is NEVER excluded (guarded with a
 * runtime warn, so a filter bug can never hide the inspected node), and
 * unfiltered groups contribute nothing.
 */
export function excludedMemberIds(
  focusedId: string,
  groups: ReadonlyArray<ConnectionGroup>,
  filters: Record<string, GroupFilter>,
  matchMember: (nodeId: string, groupKey: string) => boolean,
): Set<string> {
  const out = new Set<string>()
  for (const g of groups) {
    const filter = filters[g.key]
    if (filter === undefined) continue
    if (filter.text.trim().length === 0 && filter.hidden.length === 0) continue
    for (const m of g.members) {
      if (m.nodeId === focusedId) {
        if (!matchMember(m.nodeId, g.key)) {
          console.warn('[graph-inspector] focused member excluded', {
            node_id: focusedId,
            group_key: g.key,
          })
        }
        continue
      }
      if (!matchMember(m.nodeId, g.key)) out.add(m.nodeId)
    }
  }
  return out
}

/**
 * Group connections by `${edgeType}:${dir}`. Groups appear in
 * first-appearance order of the input, and each (nodeId, edgeType, dir)
 * tuple appears at most once per group.
 */
export function groupNeighbors(items: ReadonlyArray<ConnInput>): ConnectionGroup[] {
  const byKey = new Map<string, ConnectionGroup>()
  const seen = new Set<string>()
  for (const item of items) {
    const key = `${item.edgeType}:${item.dir}`
    let group = byKey.get(key)
    if (group === undefined) {
      group = { key, edgeType: item.edgeType, dir: item.dir, members: [] }
      byKey.set(key, group)
    }
    const tuple = `${key}\u0000${item.nodeId}`
    if (seen.has(tuple)) continue
    seen.add(tuple)
    group.members.push({ nodeId: item.nodeId })
  }
  return [...byKey.values()]
}

/** Map the neighbors API direction values onto the inspector's in/out. */
export function normalizeDirection(direction: string): 'in' | 'out' {
  return direction === 'incoming' ? 'in' : 'out'
}

/**
 * Summary line over a node's entry members, e.g.
 * `3 entries · decision 2, factual reference 1` with ` · 1 other` appended
 * past two distinct types. Types are labelled with underscores as spaces,
 * sorted by count desc with ties broken by ENTRY_TYPES declaration order.
 * Always plural; '' when there are no entries.
 */
export function entrySummary(entryTypes: ReadonlyArray<string>): string {
  if (entryTypes.length === 0) return ''
  const counts = new Map<string, number>()
  for (const t of entryTypes) counts.set(t, (counts.get(t) ?? 0) + 1)
  const declOrder = (t: string) => {
    const i = ENTRY_TYPES.indexOf(t as EntryType)
    return i === -1 ? ENTRY_TYPES.length : i
  }
  const sorted = [...counts.entries()].sort(
    (a, b) => b[1] - a[1] || declOrder(a[0]) - declOrder(b[0]),
  )
  const top = sorted.slice(0, 2)
  let out = `${entryTypes.length} entries · ${top
    .map(([t, c]) => `${entryTypeLabel(t)} ${c}`)
    .join(', ')}`
  if (sorted.length > 2) {
    const rest = entryTypes.length - top.reduce((sum, [, c]) => sum + c, 0)
    out += ` · ${rest} other`
  }
  return out
}
