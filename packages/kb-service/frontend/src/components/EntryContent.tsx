import { useState, useEffect } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import MuiLink from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { ResponseCard, MarkdownBody } from './ResponseCard'
import { InspectorConnections } from './GraphNodeInspector'
import type { InspectorMember } from './GraphNodeInspector'
import { getEntries, getGraphNeighbors } from '../api'
import { groupNeighbors, normalizeDirection } from '../graph/inspectorModel'
import type { ConnectionGroup, ConnInput } from '../graph/inspectorModel'
import type { KnowledgeEntry, PointerRotTarget } from '../kbTypes'
import { ENTRY_TYPE_COLORS } from '../entryTypes'

// Matches kb-core entry ids: kb-NNNNN (one or more digits)
const ENTRY_ID_RE = /^kb-\d+$/

/** Entry-neighbour metadata lookups are chunked at 20 ids per request. */
const NEIGHBOR_CHUNK = 20

interface ConnectionsState {
  groups: ConnectionGroup[]
  membersById: Map<string, InspectorMember>
}

/**
 * Fetches and groups one-hop graph neighbors for a KB entry for the shared
 * inspector connections body. Neighbors come from getGraphNeighbors(id,
 * 'both', 50); the API's 'outgoing'/'incoming' directions map onto the
 * inspector's 'out'/'in' and the shared groupNeighbors helper groups them by
 * `${edge_type}:${dir}`. Entry-id neighbours resolve their short_title,
 * long_title, entry_type and project_ref through chunked getEntries({ touch:
 * false }) lookups (they are not reads); non-entry members keep the raw
 * neighbor_id as their label.
 *
 * Returns state=null while loading; on failure the hook reports the error so
 * the section can show a warning instead of silently disappearing.
 */
function useEntryConnections(entryId: string): { state: ConnectionsState | null; error: string | null } {
  const [state, setState] = useState<ConnectionsState | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      setState(null)
      setError(null)
      try {
        const resp = await getGraphNeighbors(entryId, 'both', 50)
        if (cancelled) return
        const conns: ConnInput[] = resp.neighbors.map((n) => ({
          nodeId: n.neighbor_id,
          edgeType: n.edge_type,
          dir: normalizeDirection(n.direction),
        }))
        // Tags and the project render as the entry's own chips; the has_tag
        // and in_project groups are not repeated as connections.
        const groups = groupNeighbors(conns).filter(
          (g) => g.edgeType !== 'has_tag' && g.edgeType !== 'in_project',
        )
        if (groups.length === 0) {
          if (!cancelled) setState(null)
          return
        }

        // Resolve display metadata for entry-id neighbours (chunked per request)
        const entryNeighborIds = [...new Set(conns.map((c) => c.nodeId))].filter((id) =>
          ENTRY_ID_RE.test(id),
        )
        const meta = new Map<string, KnowledgeEntry>()
        for (let i = 0; i < entryNeighborIds.length; i += NEIGHBOR_CHUNK) {
          if (cancelled) return
          const chunk = entryNeighborIds.slice(i, i + NEIGHBOR_CHUNK)
          const batch = await getEntries(chunk, { touch: false })
          if (cancelled) return
          for (const r of batch.results) {
            if (r.found && r.entry) meta.set(r.id, r.entry)
          }
        }
        if (cancelled) return

        const membersById = new Map<string, InspectorMember>()
        for (const c of conns) {
          if (membersById.has(c.nodeId)) continue
          if (ENTRY_ID_RE.test(c.nodeId)) {
            const e = meta.get(c.nodeId)
            membersById.set(c.nodeId, {
              nodeId: c.nodeId,
              label: e?.short_title ?? c.nodeId,
              nodeType: 'entry',
              edgeType: c.edgeType,
              dir: c.dir,
              tooltip: e?.long_title && e.long_title.length > 0 ? e.long_title : undefined,
              entryType: e?.entry_type ?? null,
              projectRef: e?.project_ref ?? null,
            })
          } else {
            membersById.set(c.nodeId, {
              nodeId: c.nodeId,
              label: c.nodeId,
              nodeType: c.nodeId.includes(':') ? c.nodeId.split(':', 1)[0] : 'unknown',
              edgeType: c.edgeType,
              dir: c.dir,
              entryType: null,
              projectRef: null,
            })
          }
        }

        setState({ groups, membersById })
      } catch (err) {
        console.warn('[graph-inspector] connections fetch failed', {
          node_id: entryId,
          error: err instanceof Error ? err.message : String(err),
        })
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setState(null)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [entryId])

  return { state, error }
}

/**
 * Renders the Connections section for an entry: one-hop graph neighbours
 * grouped by edge type and direction through the shared inspector body
 * (pills for small groups, a scrolling list of summary rows for big ones).
 * Every pill and row calls onOpenNode with the member node id.
 */
function EntryConnections({
  entryId,
  onOpenNode,
}: {
  entryId: string
  onOpenNode: (id: string) => void
}) {
  const { state, error } = useEntryConnections(entryId)

  if (error !== null) {
    return (
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Connections
        </Typography>
        <Alert severity="warning">Connections unavailable: {error}</Alert>
      </Box>
    )
  }
  if (state === null || state.groups.length === 0) return null

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Connections
      </Typography>
      <InspectorConnections
        groups={state.groups}
        membersById={state.membersById}
        onOpenNode={onOpenNode}
      />
    </Box>
  )
}

interface Props {
  entry: KnowledgeEntry
  pointerRot: PointerRotTarget[]
  onOpenEntry: (id: string) => void
  onOpenNode: (id: string) => void
}

export function EntryContent({ entry, pointerRot, onOpenEntry, onOpenNode }: Props) {
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
          label={entry.entry_type.replace(/_/g, ' ')}
          color={ENTRY_TYPE_COLORS[entry.entry_type] ?? 'default'}
          size="small"
        />
        {entry.tags.map((tag) => (
          <Chip
            key={tag}
            label={tag}
            size="small"
            variant="outlined"
            clickable
            onClick={() => onOpenNode(`tag:${tag}`)}
          />
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
      <EntryConnections entryId={entry.id} onOpenNode={onOpenNode} />

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
