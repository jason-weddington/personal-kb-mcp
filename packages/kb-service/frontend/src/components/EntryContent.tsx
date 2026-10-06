import { useState, useEffect } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import MuiLink from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { ResponseCard, MarkdownBody } from './ResponseCard'
import { getEntries, getGraphNeighbors } from '../api'
import type { KnowledgeEntry, PointerRotTarget, EntryType } from '../kbTypes'

const ENTRY_TYPE_COLORS: Record<
  EntryType,
  'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning'
> = {
  factual_reference: 'info',
  decision: 'primary',
  pattern_convention: 'secondary',
  lesson_learned: 'warning',
  mental_map: 'success',
}

// Matches kb-core entry ids: kb-NNNNN (one or more digits)
const ENTRY_ID_RE = /^kb-\d+$/

interface NeighborItem {
  id: string
  title: string | undefined
}

interface ConnectionGroup {
  edgeType: string
  entryItems: NeighborItem[]
  nonEntryItems: string[]
}

interface ConnectionsState {
  groups: ConnectionGroup[]
}

/**
 * Fetches and groups one-hop graph neighbors for a KB entry.
 *
 * Grouping strategy: neighbors are grouped by edge_type, with a small caption
 * per group so the section stays scannable without clutter.
 * Deduplication: neighbor_id is deduped (first occurrence wins for edge_type
 * assignment) before grouping, covering the case where the same node appears
 * in both directions.
 *
 * Returns null while loading or on error (section is simply omitted).
 */
function useEntryNeighbors(entryId: string): ConnectionsState | null {
  const [state, setState] = useState<ConnectionsState | null>(null)

  useEffect(() => {
    let cancelled = false

    void (async () => {
      setState(null)
      try {
        const resp = await getGraphNeighbors(entryId, 'both', 50)
        if (cancelled) return

        // Dedupe by neighbor_id — first occurrence wins edge_type assignment
        const seen = new Set<string>()
        const deduped = resp.neighbors.filter((n) => {
          if (seen.has(n.neighbor_id)) return false
          seen.add(n.neighbor_id)
          return true
        })

        if (deduped.length === 0) {
          return // section absent when no neighbors
        }

        // Resolve display titles for entry-id neighbors (chunked at 20 per request)
        const entryNeighborIds = deduped
          .filter((n) => ENTRY_ID_RE.test(n.neighbor_id))
          .map((n) => n.neighbor_id)

        const titleMap: Record<string, string> = {}
        for (let i = 0; i < entryNeighborIds.length; i += 20) {
          if (cancelled) return
          const chunk = entryNeighborIds.slice(i, i + 20)
          const batch = await getEntries(chunk)
          for (const r of batch.results) {
            if (r.found && r.entry) {
              titleMap[r.id] = r.entry.short_title
            }
          }
        }

        if (cancelled) return

        // Group by edge_type (insertion order preserved by Map)
        const groupMap = new Map<
          string,
          { entryItems: NeighborItem[]; nonEntryItems: string[] }
        >()
        for (const n of deduped) {
          if (!groupMap.has(n.edge_type)) {
            groupMap.set(n.edge_type, { entryItems: [], nonEntryItems: [] })
          }
          const g = groupMap.get(n.edge_type)!
          if (ENTRY_ID_RE.test(n.neighbor_id)) {
            g.entryItems.push({ id: n.neighbor_id, title: titleMap[n.neighbor_id] })
          } else {
            g.nonEntryItems.push(n.neighbor_id)
          }
        }

        const groups: ConnectionGroup[] = Array.from(groupMap.entries()).map(
          ([edgeType, items]) => ({ edgeType, ...items }),
        )

        setState({ groups })
      } catch {
        // Silent failure — section is simply absent on error
        if (!cancelled) setState(null)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [entryId])

  return state
}

/**
 * Renders the Connections section for an entry.
 *
 * Shows one-hop graph neighbors grouped by edge_type:
 *   - Entry-id neighbors (kb-NNNNN) → clickable chips "[id] short_title"
 *     that open the target in the entry drawer via onOpenEntry.
 *   - Non-entry nodes (tag:foo, tech:postgres, etc.) → muted non-clickable
 *     chips showing the prefixed id (informative only).
 *
 * Omitted entirely when loading, on fetch error, or when there are no neighbors.
 */
function EntryConnections({
  entryId,
  onOpenEntry,
}: {
  entryId: string
  onOpenEntry: (id: string) => void
}) {
  const connections = useEntryNeighbors(entryId)

  if (!connections || connections.groups.length === 0) return null

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Connections
      </Typography>
      {connections.groups.map(({ edgeType, entryItems, nonEntryItems }) => (
        <Box key={edgeType} sx={{ mb: 1 }}>
          {/* Small caption per group so edge_type is visible without taking up much space */}
          <Typography
            variant="caption"
            color="text.secondary"
            component="div"
            sx={{ mb: 0.5 }}
          >
            {edgeType}
          </Typography>
          <Stack direction="row" spacing={0.5} flexWrap="wrap">
            {entryItems.map((item) => (
              <Chip
                key={item.id}
                label={`[${item.id}] ${item.title ?? item.id}`}
                size="small"
                clickable
                onClick={() => onOpenEntry(item.id)}
              />
            ))}
            {nonEntryItems.map((id) => (
              // Non-entry node: informative only, not navigable
              <Chip
                key={id}
                label={id}
                size="small"
                variant="outlined"
                sx={{ opacity: 0.6 }}
              />
            ))}
          </Stack>
        </Box>
      ))}
    </Box>
  )
}

interface Props {
  entry: KnowledgeEntry
  pointerRot: PointerRotTarget[]
  onOpenEntry: (id: string) => void
}

export function EntryContent({ entry, pointerRot, onOpenEntry }: Props) {
  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        {entry.short_title}
      </Typography>

      {entry.long_title && entry.long_title !== entry.short_title && (
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          {entry.long_title}
        </Typography>
      )}

      <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 2 }}>
        <Chip
          label={entry.entry_type}
          color={ENTRY_TYPE_COLORS[entry.entry_type] ?? 'default'}
          size="small"
        />
        {entry.tags.map((tag) => (
          <Chip key={tag} label={tag} size="small" variant="outlined" />
        ))}
      </Stack>

      <Stack spacing={0.5} sx={{ mb: 2 }}>
        <Typography variant="body2">
          <strong>ID:</strong> {entry.id}
        </Typography>
        {entry.project_ref && (
          <Typography variant="body2">
            <strong>Project:</strong> {entry.project_ref}
          </Typography>
        )}
        {entry.contributor && (
          <Typography variant="body2">
            <strong>Contributor:</strong> {entry.contributor}
          </Typography>
        )}
        {entry.team && (
          <Typography variant="body2">
            <strong>Team:</strong> {entry.team}
          </Typography>
        )}
        <Typography variant="body2">
          <strong>Confidence:</strong> {entry.confidence_level.toFixed(2)}
        </Typography>
        {entry.created_at && (
          <Typography variant="body2">
            <strong>Created:</strong>{' '}
            {new Date(entry.created_at).toLocaleDateString()}
          </Typography>
        )}
        {entry.updated_at && (
          <Typography variant="body2">
            <strong>Updated:</strong>{' '}
            {new Date(entry.updated_at).toLocaleDateString()}
          </Typography>
        )}
        {entry.superseded_by && (
          <Typography variant="body2">
            <strong>Superseded by:</strong>{' '}
            <MuiLink
              component="button"
              onClick={() => onOpenEntry(entry.superseded_by!)}
              sx={{ verticalAlign: 'baseline' }}
            >
              {entry.superseded_by}
            </MuiLink>
          </Typography>
        )}
      </Stack>

      {/* Connections section: one-hop graph neighbors, between metadata and body */}
      <EntryConnections entryId={entry.id} onOpenEntry={onOpenEntry} />

      <Divider sx={{ mb: 2 }} />

      <ResponseCard>
        <MarkdownBody>{entry.knowledge_details}</MarkdownBody>
      </ResponseCard>

      {pointerRot.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Alert severity="warning" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Pointer rot detected in linked entries:
            </Typography>
            <Stack spacing={0.5}>
              {pointerRot.map((pr) => (
                <Typography key={pr.target_id} variant="body2">
                  <MuiLink
                    component="button"
                    onClick={() => onOpenEntry(pr.target_id)}
                    sx={{ verticalAlign: 'baseline' }}
                  >
                    {pr.target_id}
                  </MuiLink>
                  {pr.superseded_by ? (
                    <>
                      {' '}superseded by{' '}
                      <MuiLink
                        component="button"
                        onClick={() => onOpenEntry(pr.superseded_by!)}
                        sx={{ verticalAlign: 'baseline' }}
                      >
                        {pr.superseded_by}
                      </MuiLink>
                    </>
                  ) : (
                    <> deactivated</>
                  )}
                </Typography>
              ))}
            </Stack>
          </Alert>
        </Box>
      )}
    </Box>
  )
}
