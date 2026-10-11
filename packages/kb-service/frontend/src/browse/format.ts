// Display helpers for the Home browse panel.

// Mirrors STALE_FILTER_THRESHOLD in kb_core/confidence/decay.py: below this
// effective confidence, kb_search hides an entry unless include_stale is set.
export const STALE_FILTER_THRESHOLD = 0.3

// Mirrors STALENESS_THRESHOLD in kb_core/confidence/decay.py: below this,
// search results carry a staleness warning.
export const STALENESS_THRESHOLD = 0.5

const TZ_SUFFIX_RE = /(Z|[+-]\d{2}:\d{2})$/i

/** Epoch ms for an ISO-8601 instant; a string without a zone suffix is UTC. */
export function parseInstant(iso: string): number {
  return Date.parse(TZ_SUFFIX_RE.test(iso) ? iso : `${iso}Z`)
}

/** Compact age such as 'just now', '5m ago', '3d ago' or '2y ago'. */
export function formatRelative(iso: string, nowMs: number): string {
  const d = (nowMs - parseInstant(iso)) / 1000
  if (d < 60) return 'just now'
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  if (d < 2592000) return `${Math.floor(d / 86400)}d ago`
  if (d < 31536000) return `${Math.floor(d / 2592000)}mo ago`
  return `${Math.floor(d / 31536000)}y ago`
}

export function formatConfidence(x: number): string {
  return `${Math.round(x * 100)}%`
}

export function confidenceColor(
  x: number,
): 'error.main' | 'warning.main' | 'text.secondary' {
  if (x < STALE_FILTER_THRESHOLD) return 'error.main'
  if (x < STALENESS_THRESHOLD) return 'warning.main'
  return 'text.secondary'
}

export function connectionsLabel(n: number): string {
  return `${n} ${n === 1 ? 'connection' : 'connections'}`
}

export function entriesLabel(n: number): string {
  return `${n.toLocaleString()} ${n === 1 ? 'entry' : 'entries'}`
}
