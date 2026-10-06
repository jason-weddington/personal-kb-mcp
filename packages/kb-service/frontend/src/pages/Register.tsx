import { useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import Box from '@mui/material/Box'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import TextField from '@mui/material/TextField'
import Button from '@mui/material/Button'
import Typography from '@mui/material/Typography'
import Alert from '@mui/material/Alert'
import { useAuth } from '../contexts/AuthContext'
import { ApiError } from '../api'

export function Register() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')
  const { register } = useAuth()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (!token) {
      return
    }

    setLoading(true)
    try {
      await register(email, password, token)
      void navigate('/')
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('An unexpected error occurred')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="center"
      minHeight="100vh"
    >
      <Card sx={{ minWidth: 360, maxWidth: 420, width: '100%', mx: 2 }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h5" fontWeight={600} gutterBottom>
            Create your account
          </Typography>

          {!token && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              An invite link from an administrator is required to register. Please
              use the invite link sent to you.
            </Alert>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSubmit} noValidate>
            <TextField
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              fullWidth
              required
              margin="normal"
              autoComplete="email"
            />
            <TextField
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              fullWidth
              required
              margin="normal"
              autoComplete="new-password"
            />
            <TextField
              label="Confirm Password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              fullWidth
              required
              margin="normal"
              autoComplete="new-password"
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 2 }}
              disabled={loading || !token}
            >
              {loading ? 'Creating account…' : 'Create account'}
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}
