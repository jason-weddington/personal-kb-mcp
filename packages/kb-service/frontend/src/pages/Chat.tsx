import { useState, useEffect, useRef } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import CircularProgress from '@mui/material/CircularProgress'
import Divider from '@mui/material/Divider'
import IconButton from '@mui/material/IconButton'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemText from '@mui/material/ListItemText'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import DeleteIcon from '@mui/icons-material/Delete'
import { streamSSE } from '../streaming'
import { ResponseCard, MarkdownBody } from '../components/ResponseCard'
import { getToken, listChats, getChatMessages, deleteChat, ApiError } from '../api'
import type {
  ChatListItem as ChatListItemType,
  ChatMessageItem,
  ChatSessionEvent,
  ChatResponseEvent,
} from '../kbTypes'

const SIDEBAR_WIDTH = 260

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export function Chat() {
  const [sessions, setSessions] = useState<ChatListItemType[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadHistory = async () => {
    try {
      const items = await listChats()
      setSessions(items)
    } catch {
      // non-fatal
    }
  }

  useEffect(() => {
    void (async () => {
      await loadHistory()
      setLoadingHistory(false)
    })()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const loadSession = async (id: string) => {
    setError(null)
    try {
      const msgs = await getChatMessages(id)
      setSessionId(id)
      setMessages(
        msgs.map((m: ChatMessageItem) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      )
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setError('Chat not found')
        void loadHistory()
      } else {
        setError(err instanceof Error ? err.message : String(err))
      }
    }
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this chat session?')) return
    try {
      await deleteChat(id)
      if (sessionId === id) {
        setSessionId(null)
        setMessages([])
      }
      void loadHistory()
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setError('Chat not found')
        void loadHistory()
      } else {
        setError(err instanceof Error ? err.message : String(err))
      }
    }
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text || sending) return

    setInput('')
    setSending(true)
    setThinking(false)
    setError(null)

    // Optimistic user bubble
    setMessages((prev) => [...prev, { role: 'user', content: text }])

    const token = getToken()
    const url = `/api/chat/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`

    let currentSession = sessionId

    try {
      await streamSSE(
        url,
        { session_id: currentSession, message: text, mode: 'chat' },
        (event, data) => {
          switch (event) {
            case 'chat_session':
              currentSession = (data as unknown as ChatSessionEvent).session_id
              setSessionId(currentSession)
              break
            case 'chat_thinking':
              setThinking(true)
              break
            case 'chat_response': {
              const d = data as unknown as ChatResponseEvent
              setThinking(false)
              setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: d.answer },
              ])
              break
            }
            case 'chat_done':
              setThinking(false)
              break
            case 'error':
              setError(String((data as { message?: unknown }).message ?? data))
              break
            case 'stream_end':
              // no-op — cleanup in finally
              break
            default:
              break
          }
        },
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSending(false)
      setThinking(false)
      void loadHistory()
    }
  }

  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 120px)', gap: 0 }}>
      {/* Sidebar */}
      <Box
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          borderRight: 1,
          borderColor: 'divider',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="space-between"
          sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}
        >
          <Typography variant="subtitle2">Conversations</Typography>
          <Button
            size="small"
            onClick={() => {
              setSessionId(null)
              setMessages([])
              setError(null)
            }}
          >
            New chat
          </Button>
        </Stack>

        {loadingHistory ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <List dense sx={{ overflow: 'auto', flexGrow: 1 }}>
            {sessions.map((s) => (
              <ListItem
                key={s.id}
                disablePadding
                secondaryAction={
                  <IconButton
                    edge="end"
                    size="small"
                    aria-label={`delete ${s.title}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      void handleDelete(s.id)
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemButton
                  selected={sessionId === s.id}
                  onClick={() => void loadSession(s.id)}
                  sx={{ pr: 5 }}
                >
                  <ListItemText
                    primary={s.title}
                    secondary={new Date(s.updated_at).toLocaleString()}
                    primaryTypographyProps={{ noWrap: true, variant: 'body2' }}
                    secondaryTypographyProps={{ noWrap: true, variant: 'caption' }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
            {sessions.length === 0 && (
              <ListItem>
                <ListItemText
                  primary="No conversations yet"
                  primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
                />
              </ListItem>
            )}
          </List>
        )}
      </Box>

      {/* Main chat area */}
      <Box
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Messages */}
        <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
          {messages.map((msg, idx) => (
            <MessageBubble key={idx} message={msg} />
          ))}
          {thinking && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="caption" color="text.secondary">
                Thinking…
              </Typography>
            </Box>
          )}
          {error && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {error}
            </Alert>
          )}
          <div ref={messagesEndRef} />
        </Box>

        <Divider />

        {/* Input area */}
        <Box sx={{ p: 2 }}>
          <Stack direction="row" spacing={1}>
            <TextField
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Send a message…"
              fullWidth
              size="small"
              disabled={sending}
              multiline
              maxRows={4}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !sending && input.trim()) {
                  e.preventDefault()
                  void handleSend()
                }
              }}
            />
            <Button
              variant="contained"
              disabled={sending || !input.trim()}
              onClick={() => void handleSend()}
              sx={{ alignSelf: 'flex-end' }}
            >
              {sending ? <CircularProgress size={20} /> : 'Send'}
            </Button>
          </Stack>
        </Box>
      </Box>
    </Box>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 1.5,
      }}
    >
      {isUser ? (
        <Paper
          elevation={0}
          sx={{
            maxWidth: '80%',
            px: 2,
            py: 1,
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            borderRadius: 2,
          }}
        >
          <Typography variant="body2">{message.content}</Typography>
        </Paper>
      ) : (
        <ResponseCard sx={{ maxWidth: '80%' }}>
          <MarkdownBody>{message.content}</MarkdownBody>
        </ResponseCard>
      )}
    </Box>
  )
}
