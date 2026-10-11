import { useEffect, useReducer, useRef, useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import CloseIcon from '@mui/icons-material/Close'
import { ResponseCard } from './ResponseCard'
import { CitedMarkdown } from './CitedMarkdown'
import { getToken } from '../api'
import { streamSSE } from '../streaming'
import type { ChatDoneEvent, ChatResponseEvent, ChatSessionEvent } from '../kbTypes'
import {
  GRAPH_CHAT_W,
  STREAM_NO_RESPONSE_MESSAGE,
  SUMMARIZE_THESE_MESSAGE,
  buildChatStreamBody,
  chatReducer,
  citationCoverage,
  initialChatState,
} from '../graph/chatModel'
import type { ChatSeed } from '../graph/chatModel'

export function GraphChatPanel({
  seed,
  onCitation,
  onNewEntries,
  onClose,
}: {
  seed: ChatSeed
  onCitation: (entryId: string) => void
  onNewEntries: (entryIds: string[]) => string[]
  onClose: () => void
}) {
  const [state, dispatch] = useReducer(chatReducer, seed, initialChatState)
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const inFlightRef = useRef(false)
  const aliveRef = useRef(true)
  const loggedIndexRef = useRef(-1)

  useEffect(() => {
    aliveRef.current = true
    return () => {
      aliveRef.current = false
    }
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [state.messages])

  useEffect(() => {
    const index = state.messages.length - 1
    const last = state.messages[index]
    if (!last || last.role !== 'assistant' || last.error) return
    const c = citationCoverage(last.content)
    if (c.mentioned > c.linked && index !== loggedIndexRef.current) {
      loggedIndexRef.current = index
      console.info('[graph-chat] unlinked citations', {
        message_index: index,
        linked: c.linked,
        mentioned: c.mentioned,
      })
    }
  }, [state.messages])

  async function send(text: string) {
    const trimmed = text.trim()
    if (trimmed === '' || inFlightRef.current) return
    if (state.followUps >= 1 && state.sessionId === null) {
      console.warn('[graph-chat] re-seeding: no session id after first exchange', {
        follow_ups: state.followUps,
      })
    }
    inFlightRef.current = true
    setInput('')
    dispatch({ type: 'send', message: trimmed })
    const token = getToken()
    const url = `/api/chat/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`
    let currentSession = state.sessionId
    let terminal: 'response' | 'error' | null = null
    const eventsSeen: string[] = []

    const onEvent = (event: string, data: Record<string, unknown>) => {
      if (!aliveRef.current) {
        console.debug('[graph-chat] dropped event after unmount', { event })
        return
      }
      eventsSeen.push(event)
      switch (event) {
        case 'chat_session':
          currentSession = String((data as unknown as ChatSessionEvent).session_id)
          dispatch({ type: 'session', sessionId: currentSession })
          break
        case 'chat_thinking':
          dispatch({ type: 'thinking' })
          break
        case 'chat_done': {
          const raw = (data as unknown as ChatDoneEvent).new_entries
          const ids = Array.isArray(raw) ? raw.filter((x): x is string => typeof x === 'string') : []
          if (ids.length > 0) {
            const hidden = onNewEntries(ids)
            if (hidden.length > 0) {
              dispatch({
                type: 'notice',
                content: `${hidden.length} new ${hidden.length === 1 ? 'entry is' : 'entries are'} not on the graph`,
              })
            }
          }
          break
        }
        case 'chat_response': {
          const d = data as unknown as ChatResponseEvent
          terminal = 'response'
          if (typeof d.session_id === 'string' && d.session_id !== currentSession) {
            console.warn('[graph-chat] session id mismatch', {
              state_session_id: currentSession,
              response_session_id: d.session_id,
            })
            currentSession = d.session_id
            dispatch({ type: 'session', sessionId: d.session_id })
          }
          dispatch({ type: 'response', answer: String(d.answer ?? '') })
          break
        }
        case 'error':
          terminal = 'error'
          dispatch({
            type: 'error',
            message: String((data as { message?: unknown }).message ?? 'unknown error'),
          })
          break
        default:
          break
      }
    }

    try {
      await streamSSE(url, buildChatStreamBody(seed, state.sessionId, trimmed), onEvent)
    } catch (err) {
      terminal = 'error'
      if (aliveRef.current) {
        dispatch({ type: 'error', message: err instanceof Error ? err.message : String(err) })
      }
    } finally {
      inFlightRef.current = false
      if (aliveRef.current) {
        if (terminal === null) {
          dispatch({ type: 'error', message: STREAM_NO_RESPONSE_MESSAGE })
          console.warn('[graph-chat] stream ended without response', {
            session_id: currentSession,
            events_seen: eventsSeen,
          })
        }
        dispatch({ type: 'end' })
      }
    }
  }

  return (
    <Paper
      component="aside"
      role="complementary"
      aria-label="Graph chat"
      square
      elevation={0}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        flexShrink: 0,
        width: { xs: '100%', sm: GRAPH_CHAT_W },
        position: { xs: 'absolute', sm: 'relative' },
        inset: { xs: 0, sm: 'auto' },
        zIndex: { xs: 3, sm: 'auto' },
        borderLeft: 1,
        borderColor: 'divider',
      }}
    >
      <Stack
        direction="row"
        sx={{ alignItems: 'center', gap: 1, px: 1.5, py: 1, borderBottom: 1, borderColor: 'divider' }}
      >
        <Chip
          label={seed.mode}
          size="small"
          sx={{
            textTransform: 'uppercase',
            fontSize: 11,
            fontWeight: 600,
            ...(seed.mode === 'summarize'
              ? { bgcolor: '#1a5276', color: '#a8d8f0' }
              : { bgcolor: '#4a3520', color: '#f0c878' }),
          }}
        />
        <Typography
          variant="body2"
          noWrap
          title={seed.question}
          sx={{ color: 'text.secondary', flex: 1, minWidth: 0 }}
        >
          {seed.question}
        </Typography>
        <IconButton size="small" aria-label="Close chat" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Stack>

      <Box
        role="log"
        aria-label="Chat messages"
        sx={{ flex: 1, overflowY: 'auto', p: 2, display: 'flex', flexDirection: 'column', gap: 1.25 }}
      >
        {state.messages.map((m, i) => {
          if (m.role === 'user') {
            return (
              <Paper
                key={i}
                elevation={0}
                sx={{
                  alignSelf: 'flex-end',
                  maxWidth: '85%',
                  px: 1.5,
                  py: 1,
                  bgcolor: 'primary.main',
                  color: 'primary.contrastText',
                  borderRadius: 2,
                }}
              >
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {m.content}
                </Typography>
              </Paper>
            )
          }
          if (m.role === 'notice') {
            return (
              <Typography key={i} variant="caption" sx={{ color: 'text.secondary', alignSelf: 'center' }}>
                {m.content}
              </Typography>
            )
          }
          if (m.error) {
            return (
              <Alert key={i} severity="error">
                {m.content}
              </Alert>
            )
          }
          return (
            <ResponseCard key={i} sx={{ alignSelf: 'flex-start', maxWidth: '85%' }}>
              <CitedMarkdown onCitation={onCitation}>{m.content}</CitedMarkdown>
            </ResponseCard>
          )
        })}
        {state.thinking && (
          <Stack direction="row" sx={{ alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} />
            <Typography variant="caption">Thinking…</Typography>
          </Stack>
        )}
        <div ref={endRef} />
      </Box>

      {seed.mode === 'explore' && state.followUps === 0 && !state.busy && (
        <Box sx={{ px: 1.5, pb: 1 }}>
          <Button size="small" variant="outlined" onClick={() => void send(SUMMARIZE_THESE_MESSAGE)}>
            Summarize these
          </Button>
        </Box>
      )}

      <Stack direction="row" sx={{ gap: 1, p: 1.5, borderTop: 1, borderColor: 'divider' }}>
        <TextField
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a follow-up…"
          slotProps={{ htmlInput: { 'aria-label': 'Follow-up message' } }}
          inputRef={inputRef}
          fullWidth
          size="small"
          multiline
          maxRows={4}
          autoFocus
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              void send(input)
            } else if (e.key === 'Escape') {
              inputRef.current?.blur()
            }
          }}
        />
        <Button
          variant="contained"
          disabled={state.busy || !input.trim()}
          onClick={() => void send(input)}
        >
          Send
        </Button>
      </Stack>
    </Paper>
  )
}
