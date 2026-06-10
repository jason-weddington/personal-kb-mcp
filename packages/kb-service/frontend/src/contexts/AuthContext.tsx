import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from 'react'
import type { ReactNode } from 'react'
import { api } from '../api'
import type { UserResponse } from '../types'

interface AuthContextValue {
  isAuthenticated: boolean
  user: UserResponse | null
  loading: boolean
  login(email: string, password: string): Promise<void>
  register(email: string, password: string, inviteToken: string): Promise<void>
  logout(): void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(() => {
    try {
      const stored = localStorage.getItem('kb-user')
      return stored ? (JSON.parse(stored) as UserResponse) : null
    } catch {
      return null
    }
  })

  const [loading, setLoading] = useState<boolean>(
    () => Boolean(localStorage.getItem('kb-token')),
  )

  useEffect(() => {
    let cancelled = false
    const token = localStorage.getItem('kb-token')
    if (!token) {
      // no token — no request, loading stays false
      return
    }
    api.auth
      .me()
      .then((u) => {
        if (!cancelled) {
          setUser(u)
          localStorage.setItem('kb-user', JSON.stringify(u))
        }
      })
      .catch(() => {
        if (!cancelled) {
          // kb-01449 guard #2: no navigation on bootstrap failure
          localStorage.removeItem('kb-token')
          localStorage.removeItem('kb-user')
          setUser(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.auth.login(email, password)
    localStorage.setItem('kb-token', res.token)
    localStorage.setItem('kb-user', JSON.stringify(res.user))
    setUser(res.user)
  }, [])

  const register = useCallback(
    async (email: string, password: string, inviteToken: string) => {
      const res = await api.auth.register(email, password, inviteToken)
      localStorage.setItem('kb-token', res.token)
      localStorage.setItem('kb-user', JSON.stringify(res.user))
      setUser(res.user)
    },
    [],
  )

  const logout = useCallback(() => {
    api.auth.logout().catch(() => {})
    localStorage.removeItem('kb-token')
    localStorage.removeItem('kb-user')
    setUser(null)
  }, [])

  const value: AuthContextValue = {
    isAuthenticated: user !== null,
    user,
    loading,
    login,
    register,
    logout,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}
