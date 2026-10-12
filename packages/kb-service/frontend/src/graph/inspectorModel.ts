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
