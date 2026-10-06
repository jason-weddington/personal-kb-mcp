import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import CircularProgress from '@mui/material/CircularProgress'
import { getEntries } from '../api'
import type { KnowledgeEntry, PointerRotTarget } from '../kbTypes'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import { EntryContent } from '../components/EntryContent'

export function EntryDetail() {
  const { id } = useParams<{ id: string }>()
  const { openEntry } = useEntryDrawer()
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
      <EntryContent entry={entry} pointerRot={pointerRot} onOpenEntry={openEntry} />
    </Box>
  )
}
