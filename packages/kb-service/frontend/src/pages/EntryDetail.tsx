import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import Divider from '@mui/material/Divider'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getEntries } from '../api'
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

export function EntryDetail() {
  const { id } = useParams<{ id: string }>()
  const [entry, setEntry] = useState<KnowledgeEntry | null>(null)
  const [pointerRot, setPointerRot] = useState<PointerRotTarget[]>([])
  const [notFound, setNotFound] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    let cancelled = false
    void (async () => {
      setLoading(true)
      setNotFound(false)
      setEntry(null)
      try {
        const resp = await getEntries([id])
        if (!cancelled) {
          const result = resp.results[0]
          if (!result || !result.found) {
            setNotFound(true)
          } else {
            setEntry(result.entry)
            setPointerRot(result.pointer_rot ?? [])
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [id])

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

  if (notFound) {
    return (
      <Alert severity="warning">{id} not found</Alert>
    )
  }

  if (!entry) return null

  return (
    <Box sx={{ maxWidth: 900 }}>
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
            <Link to={`/entries/${entry.superseded_by}`}>
              {entry.superseded_by}
            </Link>
          </Typography>
        )}
      </Stack>

      <Divider sx={{ mb: 2 }} />

      <Box
        sx={{
          '& pre': {
            overflowX: 'auto',
            bgcolor: 'action.hover',
            p: 1.5,
            borderRadius: 1,
          },
          '& code': { fontFamily: 'monospace', fontSize: '0.875em' },
          '& table': { borderCollapse: 'collapse', width: '100%' },
          '& th, & td': { border: '1px solid', borderColor: 'divider', p: 0.75 },
        }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {entry.knowledge_details}
        </ReactMarkdown>
      </Box>

      {pointerRot.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Alert severity="warning" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Pointer rot detected in linked entries:
            </Typography>
            <Stack spacing={0.5}>
              {pointerRot.map((pr) => (
                <Typography key={pr.target_id} variant="body2">
                  <Link to={`/entries/${pr.target_id}`}>{pr.target_id}</Link>
                  {pr.superseded_by ? (
                    <> superseded by{' '}
                      <Link to={`/entries/${pr.superseded_by}`}>{pr.superseded_by}</Link>
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
