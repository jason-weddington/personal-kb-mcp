import { useState } from 'react'
import { Link } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { streamSSE } from '../streaming'
import { getToken } from '../api'
import type {
  ClassifiedEvent,
  StatusEvent,
  SynthesisResultEvent,
  EntriesEvent,
  AskEntry,
} from '../kbTypes'

export function Ask() {
  const [question, setQuestion] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [mode, setMode] = useState<string | null>(null)
  const [statusMsg, setStatusMsg] = useState('')
  const [answer, setAnswer] = useState<string | null>(null)
  const [entryIds, setEntryIds] = useState<string[]>([])
  const [exploreEntries, setExploreEntries] = useState<AskEntry[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleAsk = async () => {
    setSubmitting(true)
    setMode(null)
    setStatusMsg('')
    setAnswer(null)
    setEntryIds([])
    setExploreEntries([])
    setError(null)

    const token = getToken()
    const url = `/api/kb/query/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`

    try {
      await streamSSE(url, { question }, (event, data) => {
        switch (event) {
          case 'classified':
            setMode((data as unknown as ClassifiedEvent).mode)
            break
          case 'status':
            setStatusMsg((data as unknown as StatusEvent).message)
            break
          case 'synthesis_result': {
            const d = data as unknown as SynthesisResultEvent
            setAnswer(d.answer)
            setEntryIds(d.entry_ids)
            break
          }
          case 'entries': {
            const d = data as unknown as EntriesEvent
            setExploreEntries(d.entries)
            break
          }
          case 'error':
            setError(String((data as { message?: unknown }).message ?? data))
            break
          case 'stream_end':
            // no-op — cleanup handled in finally
            break
          default:
            break
        }
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
      setStatusMsg('')
    }
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Ask
      </Typography>

      <Stack spacing={2} sx={{ mb: 3 }}>
        <TextField
          label="Question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          multiline
          minRows={2}
          fullWidth
          disabled={submitting}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && e.ctrlKey && !submitting && question.trim()) {
              void handleAsk()
            }
          }}
        />
        <Button
          variant="contained"
          disabled={submitting || !question.trim()}
          onClick={() => void handleAsk()}
          startIcon={submitting ? <CircularProgress size={18} /> : null}
        >
          {submitting ? 'Thinking…' : 'Ask'}
        </Button>
      </Stack>

      {/* Live status line */}
      {statusMsg && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mb: 1 }}
          data-testid="status-line"
        >
          {statusMsg}
        </Typography>
      )}

      {/* Mode chip (shown once classified event arrives) */}
      {mode && (
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
          <Typography variant="body2">Mode:</Typography>
          <Chip label={mode} size="small" color="primary" />
        </Stack>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Summarize mode result */}
      {answer !== null && (
        <Box sx={{ mb: 2 }}>
          <Box
            sx={{
              '& pre': { overflowX: 'auto', bgcolor: 'action.hover', p: 1.5, borderRadius: 1 },
              '& code': { fontFamily: 'monospace', fontSize: '0.875em' },
            }}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
          </Box>
          {entryIds.length > 0 && (
            <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 1 }}>
              <Typography variant="body2">Sources:</Typography>
              {entryIds.map((id) => (
                <Chip
                  key={id}
                  label={id}
                  size="small"
                  component={Link}
                  to={`/entries/${id}`}
                  clickable
                />
              ))}
            </Stack>
          )}
        </Box>
      )}

      {/* Explore mode results */}
      {exploreEntries.length > 0 && (
        <Stack spacing={2}>
          {exploreEntries.map((e) => (
            <Box
              key={e.id}
              sx={{
                p: 2,
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
              }}
            >
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                <Typography
                  variant="subtitle2"
                  component={Link}
                  to={`/entries/${e.id}`}
                  sx={{ textDecoration: 'none', color: 'primary.main' }}
                >
                  {e.short_title}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {e.id}
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                {e.context}
              </Typography>
            </Box>
          ))}
        </Stack>
      )}
    </Box>
  )
}
