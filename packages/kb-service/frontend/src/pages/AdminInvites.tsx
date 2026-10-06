import { useState, useEffect, useCallback } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import { api, ApiError } from '../api'
import type { Invite, CreatedInvite } from '../types'

export function AdminInvites() {
  const [invites, setInvites] = useState<Invite[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  // Increment to trigger a data reload
  const [reloadTick, setReloadTick] = useState(0)

  // Create dialog
  const [createOpen, setCreateOpen] = useState(false)
  const [note, setNote] = useState('')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)

  // Created-invite result dialog
  const [createdInvite, setCreatedInvite] = useState<CreatedInvite | null>(null)

  // Revoke state (per-row error shown in list alert)
  const [revokeError, setRevokeError] = useState<string | null>(null)

  // Load invites — inline async IIFE avoids react-hooks/set-state-in-effect
  useEffect(() => {
    void (async () => {
      setLoadError(null)
      try {
        const data = await api.admin.invites.list()
        setInvites(data)
      } catch (e) {
        if (e instanceof ApiError) {
          setLoadError(e.detail)
        } else {
          setLoadError('Failed to load invites')
        }
      }
    })()
  }, [reloadTick])

  const handleCreate = useCallback(async () => {
    setCreating(true)
    setCreateError(null)
    try {
      const result = await api.admin.invites.create(note)
      setCreatedInvite(result)
      setCreateOpen(false)
      setNote('')
      setReloadTick((t) => t + 1)
    } catch (e) {
      if (e instanceof ApiError) {
        setCreateError(e.detail)
      } else {
        setCreateError('Failed to create invite')
      }
    } finally {
      setCreating(false)
    }
  }, [note])

  const handleRevoke = useCallback(async (token: string) => {
    setRevokeError(null)
    try {
      await api.admin.invites.revoke(token)
    } catch (e) {
      if (e instanceof ApiError) {
        setRevokeError(e.detail)
      } else {
        setRevokeError('Failed to revoke invite')
      }
    } finally {
      setReloadTick((t) => t + 1)
    }
  }, [])

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h5">Invites</Typography>
        <Button
          variant="contained"
          onClick={() => {
            setNote('')
            setCreateError(null)
            setCreateOpen(true)
          }}
        >
          New invite
        </Button>
      </Box>

      {loadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {loadError}
        </Alert>
      )}
      {revokeError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {revokeError}
        </Alert>
      )}

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Note</TableCell>
            <TableCell>Created</TableCell>
            <TableCell>Status</TableCell>
            <TableCell />
          </TableRow>
        </TableHead>
        <TableBody>
          {invites.map((inv) => (
            <TableRow key={inv.token}>
              <TableCell>{inv.note ?? '—'}</TableCell>
              <TableCell>
                {new Date(inv.createdAt).toLocaleDateString()}
              </TableCell>
              <TableCell>
                {inv.usedAt !== null ? (
                  <Chip
                    label={`Used${inv.usedBy ? ` by ${inv.usedBy}` : ''}`}
                    size="small"
                    color="default"
                  />
                ) : (
                  <Chip label="Active" size="small" color="success" />
                )}
              </TableCell>
              <TableCell>
                {inv.usedAt === null && (
                  <Button
                    size="small"
                    color="error"
                    onClick={() => void handleRevoke(inv.token)}
                  >
                    Revoke
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))}
          {invites.length === 0 && (
            <TableRow>
              <TableCell colSpan={4} align="center">
                <Typography variant="body2" color="text.secondary">
                  No invites yet.
                </Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      {/* Create invite dialog */}
      <Dialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        fullWidth
        maxWidth="xs"
      >
        <DialogTitle>New Invite</DialogTitle>
        <DialogContent>
          {createError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {createError}
            </Alert>
          )}
          <TextField
            label="Note (optional)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            fullWidth
            autoFocus
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => void handleCreate()}
            disabled={creating}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Created invite URL dialog (shown once) */}
      <Dialog
        open={Boolean(createdInvite)}
        onClose={() => setCreatedInvite(null)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>Invite Created</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            Share this URL with the invitee. It will not be shown again.
          </Alert>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TextField
              value={createdInvite?.url ?? ''}
              inputProps={{ readOnly: true }}
              fullWidth
              size="small"
            />
            <IconButton
              size="small"
              aria-label="copy invite url"
              onClick={() => {
                void navigator.clipboard.writeText(createdInvite?.url ?? '')
              }}
            >
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreatedInvite(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
