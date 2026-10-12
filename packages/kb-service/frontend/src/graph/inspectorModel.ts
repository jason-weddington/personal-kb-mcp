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

/** A group with at most this many members renders as pills; more, as rows. */
export const INSPECTOR_PILL_MAX = 12

/** How many facet chips render before the `+N` reveal chip. */
export const FACET_TOP_N = 8

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

/** A per-group filter: a title substring plus selected facet keys (ANDed). */
export type GroupFilter = { text: string; facets: string[] }

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

/**
 * Does `member` pass the group's filter? The text query (trimmed, lowercased)
 * must be a substring of the label plus long title; every facet key must hold.
 * Facet keys: `entry_type:<t>` (member is an entry of type <t>),
 * `project:<ref>` (member's project ref), `node:<id>` (the member's own id or
 * one of its neighbour ids).
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
  for (const facet of filter.facets) {
    const sep = facet.indexOf(':')
    const kind = sep === -1 ? '' : facet.slice(0, sep)
    const value = sep === -1 ? '' : facet.slice(sep + 1)
    if (kind === 'entry_type') {
      if (!(member.nodeType === 'entry' && member.entryType === value)) return false
    } else if (kind === 'project') {
      if (member.projectRef !== value) return false
    } else if (kind === 'node') {
      if (!(member.nodeId === value || memberNeighborIds.has(value))) return false
    } else {
      return false
    }
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
      return memberMatches(m, { text: '', facets: [facet.key] }, neighborIds) ? sum + 1 : sum
    }, 0)
  }
  facets.sort((a, b) => b.count - a.count || (a.label < b.label ? -1 : a.label > b.label ? 1 : 0))
  return facets
}

/**
 * The nodeIds that are members of a group with an active filter (a non-empty
 * trimmed text or a non-empty facet list) and that fail `matchMember` for that
 * group — the set the canvas subtracts from the visible focus neighbourhood.
 * The focused node itself is NEVER excluded (guarded with a runtime warn, so a
 * filter bug can never hide the inspected node), and unfiltered groups
 * contribute nothing.
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
    if (filter.text.trim().length === 0 && filter.facets.length === 0) continue
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
