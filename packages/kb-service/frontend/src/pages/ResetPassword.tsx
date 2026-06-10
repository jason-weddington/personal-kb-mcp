import { useState, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { api, ApiError } from '../api'

export function ResetPassword() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') ?? ''
  const hasToken = token.length > 0

  const [newPassword, setNewPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [mismatch, setMismatch] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = useCallback(async () => {
    if (!hasToken) return
    if (newPassword !== confirm) {
      setMismatch(true)
      return
    }
    setMismatch(false)
    setError(null)
    setSubmitting(true)
    try {
      await api.auth.resetPassword(token, newPassword)
      setSuccess(true)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail)
      } else {
        setError('Unexpected error')
      }
    } finally {
      setSubmitting(false)
    }
  }, [hasToken, token, newPassword, confirm])

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      p={2}
    >
      <Card sx={{ maxWidth: 420, width: '100%' }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Reset Password
          </Typography>

          {!hasToken && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Missing reset token — request a new link from an admin.
            </Alert>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {success ? (
            <Alert severity="success">
              Password reset successfully.{' '}
              <Link to="/login">Sign in now</Link>
            </Alert>
          ) : (
            <Box
              component="form"
              onSubmit={(e) => {
                e.preventDefault()
                void handleSubmit()
              }}
              sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
            >
              <TextField
                label="New password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                disabled={!hasToken}
                fullWidth
              />
              <TextField
                label="Confirm new password"
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                disabled={!hasToken}
                error={mismatch}
                helperText={mismatch ? 'Passwords do not match' : undefined}
                fullWidth
              />
              <Button
                type="submit"
                variant="contained"
                disabled={!hasToken || submitting}
                fullWidth
              >
                Reset password
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}
