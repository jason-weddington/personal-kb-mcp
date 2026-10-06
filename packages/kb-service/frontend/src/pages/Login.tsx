import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Box from '@mui/material/Box'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import TextField from '@mui/material/TextField'
import Button from '@mui/material/Button'
import Typography from '@mui/material/Typography'
import Alert from '@mui/material/Alert'
import IconButton from '@mui/material/IconButton'
import InputAdornment from '@mui/material/InputAdornment'
import Visibility from '@mui/icons-material/Visibility'
import VisibilityOff from '@mui/icons-material/VisibilityOff'
import { useAuth } from '../contexts/AuthContext'
import { ApiError } from '../api'

export function Login() {
  const navigate = useNavigate()
  const { login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await login(email, password)
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
            Sign in to Personal KB
          </Typography>

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
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              fullWidth
              required
              margin="normal"
              autoComplete="current-password"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword((v) => !v)}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 2 }}
              disabled={loading}
            >
              {loading ? 'Signing in…' : 'Sign in'}
            </Button>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            Registration is invite-only. An invite link from an administrator is
            required to create an account.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  )
}
