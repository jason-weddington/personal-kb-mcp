import { useState, useEffect } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import CircularProgress from '@mui/material/CircularProgress'
import Drawer from '@mui/material/Drawer'
import IconButton from '@mui/material/IconButton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import CloseIcon from '@mui/icons-material/Close'
import { getEntries } from '../api'
import type { KnowledgeEntry, PointerRotTarget } from '../kbTypes'
import { EntryContent } from './EntryContent'

interface Props {
  stack: string[]
  onBack: () => void
  onClose: () => void
  onOpenEntry: (id: string) => void
}

export function EntryDrawer({ stack, onBack, onClose, onOpenEntry }: Props) {
  const id = stack[stack.length - 1]

  const [entry, setEntry] = useState<KnowledgeEntry | null>(null)
  const [pointerRot, setPointerRot] = useState<PointerRotTarget[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notFound, setNotFound] = useState(false)

  useEffect(() => {
    if (!id) return // kb-01449 gate: no fetch while closed
    let cancelled = false
    void (async () => {
      setLoading(true)
      setError(null)
      setNotFound(false)
      setEntry(null)
      try {
        const resp = await getEntries([id])
        if (!cancelled) {
          const result = resp.results[0]
          if (!result || !result.found || result.entry == null) {
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

  return (
    <Drawer
      anchor="right"
      variant="temporary"
      open={stack.length > 0}
      onClose={onClose}
      disableScrollLock
      sx={{
        '& .MuiDrawer-paper': {
          width: { xs: '100vw', sm: 640 },
          boxSizing: 'border-box',
          top: '64px',
          height: 'calc(100% - 64px)',
        },
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header */}
        <Stack
          direction="row"
          alignItems="center"
          sx={{ p: 2 }}
        >
          {stack.length > 1 && (
            <IconButton
              aria-label="Back"
              onClick={onBack}
              edge="start"
              sx={{ mr: 1 }}
            >
              <ArrowBackIcon />
            </IconButton>
          )}
          <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
            {entry?.short_title ?? id}
          </Typography>
          <IconButton aria-label="Close" onClick={onClose} edge="end">
            <CloseIcon />
          </IconButton>
        </Stack>

        {/* Scrollable content */}
        <Box sx={{ flex: 1, overflow: 'auto', px: 2, py: 1 }}>
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <CircularProgress />
            </Box>
          )}
          {!loading && error && (
            <Alert severity="error">{error}</Alert>
          )}
          {!loading && !error && notFound && (
            <Alert severity="warning">{id} not found</Alert>
          )}
          {!loading && !error && !notFound && entry && (
            <EntryContent
              entry={entry}
              pointerRot={pointerRot}
              onOpenEntry={onOpenEntry}
            />
          )}
        </Box>
      </Box>
    </Drawer>
  )
}
