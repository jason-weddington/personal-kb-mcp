import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import MuiLink from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { ResponseCard, MarkdownBody } from './ResponseCard'
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
