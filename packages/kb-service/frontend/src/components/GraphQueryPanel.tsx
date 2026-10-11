import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import IconButton from '@mui/material/IconButton'
import LinearProgress from '@mui/material/LinearProgress'
import List from '@mui/material/List'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemText from '@mui/material/ListItemText'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import CloseIcon from '@mui/icons-material/Close'
import type { QueryState } from '../graph/queryModel'
import { MarkdownBody, ResponseCard } from './ResponseCard'
import { CitedMarkdown } from './CitedMarkdown'

export interface GraphQueryPanelProps {
  state: QueryState
  renderedIds: ReadonlySet<string>
  labelFor: (id: string) => string | null
  onSelectEntry: (id: string) => void
  onClose: () => void
  footer?: React.ReactNode
  onCitation?: (id: string) => void
}

export function GraphQueryPanel({
  state,
  renderedIds,
  labelFor,
  onSelectEntry,
  onClose,
  footer,
  onCitation,
}: GraphQueryPanelProps) {
  if (state.phase === 'idle') return null

  const summarize = state.answer !== null
  const entries = state.entries ?? []
  const entryById = new Map(entries.map((e) => [e.id, e]))
  const ids = summarize ? state.sourceIds : entries.map((e) => e.id)
  const hidden = ids.filter((id) => !renderedIds.has(id)).length
  const hasAnswer = typeof state.answer === 'string' && state.answer !== ''

  return (
    <Paper
      role="region"
      aria-label="Query results"
      elevation={8}
      sx={{
        p: 2,
        width: { xs: '100%', sm: 420 },
        minHeight: 0,
        overflowY: 'auto',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Chip
          size="small"
          label={state.queryMode === 'pending' ? 'classifying' : state.queryMode}
          color={state.queryMode === 'summarize' ? 'primary' : 'default'}
        />
        <Typography variant="subtitle2" sx={{ flex: 1, wordBreak: 'break-word' }}>
          {state.question}
        </Typography>
        <IconButton size="small" aria-label="Close query" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {state.phase === 'streaming' && <LinearProgress data-testid="query-progress" />}
      {state.statusMessage !== '' && (
        <Typography variant="body2" color="text.secondary" data-testid="query-status">
          {state.statusMessage}
        </Typography>
      )}

      {state.error !== null && (
        <Alert severity="error" sx={{ mt: 1 }}>
          {state.error}
        </Alert>
      )}

      {hasAnswer && (
        <ResponseCard sx={{ mt: 1 }}>
          {onCitation ? (
            <CitedMarkdown onCitation={onCitation}>{state.answer as string}</CitedMarkdown>
          ) : (
            <MarkdownBody>{state.answer as string}</MarkdownBody>
          )}
        </ResponseCard>
      )}

      {ids.length > 0 && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="subtitle2">
            {summarize
              ? 'Sources'
              : ids.length === 1
                ? 'Found 1 entry'
                : `Found ${ids.length} entries`}
          </Typography>
          {hidden > 0 && (
            <Typography variant="caption" color="text.secondary" component="div">
              {`${hidden} of ${ids.length} results not shown on the graph`}
            </Typography>
          )}
          <List dense disablePadding>
            {ids.map((id) => {
              const entry = entryById.get(id)
              const label = labelFor(id)
              const primary = summarize ? (label ?? id) : entry?.short_title || id
              const context = !summarize && entry?.context ? entry.context : null
              const note =
                label === null ? 'not in graph' : !renderedIds.has(id) ? 'hidden by filters' : null
              return (
                <ListItemButton key={id} onClick={() => onSelectEntry(id)}>
                  <ListItemText
                    primary={primary}
                    secondary={
                      <>
                        {context !== null && (
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            component="span"
                            display="block"
                          >
                            {context}
                          </Typography>
                        )}
                        <Typography variant="caption" color="text.secondary" component="span">
                          {id}
                        </Typography>
                        {note !== null && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            component="span"
                            sx={{ ml: 1 }}
                          >
                            {note}
                          </Typography>
                        )}
                      </>
                    }
                  />
                </ListItemButton>
              )
            })}
          </List>
        </Box>
      )}

      {state.phase === 'done' && ids.length === 0 && !hasAnswer && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          No matching entries found.
        </Typography>
      )}

      {footer}
    </Paper>
  )
}
