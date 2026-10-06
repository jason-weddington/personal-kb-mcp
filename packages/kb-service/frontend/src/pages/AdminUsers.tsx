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
import { useAuth } from '../contexts/AuthContext'
import { api, ApiError } from '../api'
import type { AdminUser, PasswordResetIssued } from '../types'

export function AdminUsers() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState<AdminUser[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  // Increment to trigger a data reload
  const [reloadTick, setReloadTick] = useState(0)

  // Per-row action errors rendered at top (triggered by stale 404s etc.)
  const [actionError, setActionError] = useState<string | null>(null)

  // Password-reset dialog
  const [resetInfo, setResetInfo] = useState<PasswordResetIssued | null>(null)

  // Delete confirm dialog
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null)
  const [deleting, setDeleting] = useState(false)

  // Load users — inline async IIFE avoids react-hooks/set-state-in-effect
  useEffect(() => {
    void (async () => {
      setLoadError(null)
      try {
        const data = await api.admin.users.list()
        setUsers(data)
      } catch (e) {
        if (e instanceof ApiError) {
          setLoadError(e.detail)
        } else {
          setLoadError('Failed to load users')
        }
      }
    })()
  }, [reloadTick])

  const handlePromote = useCallback(
    async (id: string) => {
      setActionError(null)
      try {
        const updated = await api.admin.users.promote(id)
        setUsers((prev) => prev.map((u) => (u.id === id ? updated : u)))
      } catch (e) {
        if (e instanceof ApiError) {
          setActionError(e.detail)
          setReloadTick((t) => t + 1)
        } else {
          setActionError('Failed to promote user')
        }
      }
    },
    [],
  )

  const handleIssueReset = useCallback(
    async (id: string) => {
      setActionError(null)
      try {
        const info = await api.admin.users.issuePasswordReset(id)
        setResetInfo(info)
      } catch (e) {
        if (e instanceof ApiError) {
          setActionError(e.detail)
          setReloadTick((t) => t + 1)
        } else {
          setActionError('Failed to issue password reset')
        }
      }
    },
    [],
  )

  const handleDelete = useCallback(async () => {
    if (!deleteTarget) return
    setDeleting(true)
    setActionError(null)
    try {
      await api.admin.users.deleteUser(deleteTarget.id)
      setUsers((prev) => prev.filter((u) => u.id !== deleteTarget.id))
      setDeleteTarget(null)
    } catch (e) {
      if (e instanceof ApiError) {
        setActionError(e.detail)
        setReloadTick((t) => t + 1)
      } else {
        setActionError('Failed to delete user')
      }
      setDeleteTarget(null)
    } finally {
      setDeleting(false)
    }
  }, [deleteTarget])

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Users
      </Typography>

      {loadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {loadError}
        </Alert>
      )}
      {actionError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {actionError}
        </Alert>
      )}

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Email</TableCell>
            <TableCell>Admin</TableCell>
            <TableCell>Created</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {users.map((u) => {
            const isSelf = currentUser?.id === u.id
            return (
              <TableRow key={u.id}>
                <TableCell>{u.email}</TableCell>
                <TableCell>
                  {u.isAdmin ? (
                    <Chip label="Admin" size="small" color="primary" />
                  ) : (
                    <Chip label="User" size="small" color="default" />
                  )}
                </TableCell>
                <TableCell>
                  {new Date(u.createdAt).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {!u.isAdmin && (
                      <Button
                        size="small"
                        onClick={() => void handlePromote(u.id)}
                      >
                        Promote
                      </Button>
                    )}
                    <Button
                      size="small"
                      onClick={() => void handleIssueReset(u.id)}
                    >
                      Reset password
                    </Button>
                    {!isSelf && (
                      <Button
                        size="small"
                        color="error"
                        onClick={() => setDeleteTarget(u)}
                      >
                        Delete
                      </Button>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            )
          })}
          {users.length === 0 && (
            <TableRow>
              <TableCell colSpan={4} align="center">
                <Typography variant="body2" color="text.secondary">
                  No users.
                </Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      {/* Password-reset URL dialog (shown once) */}
      <Dialog
        open={Boolean(resetInfo)}
        onClose={() => setResetInfo(null)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>Password Reset Link</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            Share this link with the user. Expires:{' '}
            {resetInfo ? new Date(resetInfo.expiresAt).toLocaleString() : ''}
          </Alert>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TextField
              value={resetInfo?.url ?? ''}
              inputProps={{ readOnly: true }}
              fullWidth
              size="small"
            />
            <IconButton
              size="small"
              aria-label="copy reset url"
              onClick={() => {
                void navigator.clipboard.writeText(resetInfo?.url ?? '')
              }}
            >
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetInfo(null)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Delete confirm dialog */}
      <Dialog
        open={Boolean(deleteTarget)}
        onClose={() => setDeleteTarget(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Delete User?</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete <strong>{deleteTarget?.email}</strong>? This
            cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => void handleDelete()}
            disabled={deleting}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
