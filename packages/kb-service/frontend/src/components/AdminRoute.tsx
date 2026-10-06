import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import Box from '@mui/material/Box'
import CircularProgress from '@mui/material/CircularProgress'
import { useAuth } from '../contexts/AuthContext'

interface AdminRouteProps {
  children: ReactNode
}

export function AdminRoute({ children }: AdminRouteProps) {
  const { isAuthenticated, user, loading, authMode } = useAuth()

  if (loading) {
    return (
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
      >
        <CircularProgress />
      </Box>
    )
  }

  // No-auth (local) mode: admin is a hosted-only concept — no admin surface.
  if (authMode === 'none') {
    return <Navigate to="/" replace />
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  if (!user?.isAdmin) {
    return <Navigate to="/" replace />
  }

  return <>{children}</>
}
