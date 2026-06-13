import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from 'react'
import type { ReactNode } from 'react'
import { api, getRuntime } from '../api'
import type { AuthMode, UserResponse } from '../types'

interface AuthContextValue {
  isAuthenticated: boolean
  authMode: AuthMode
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

  // Default to 'jwt' so the hosted behaviour holds while the runtime fetch is
  // pending or (in tests) unmocked — switched to 'none' only once the backend
  // confirms local mode.
  const [authMode, setAuthMode] = useState<AuthMode>('jwt')

  // Loading is driven by BOTH the runtime fetch and (when a token exists) the
  // me() bootstrap. CRUX: runtimeLoading starts true UNCONDITIONALLY — not from
  // token presence — so in no-auth mode ProtectedRoute/AdminRoute show their
  // spinner instead of flashing /login before authMode is known.
  const [runtimeLoading, setRuntimeLoading] = useState<boolean>(true)
  const [authLoading, setAuthLoading] = useState<boolean>(
    () => Boolean(localStorage.getItem('kb-token')),
  )
  const loading = runtimeLoading || authLoading

  // Fetch the runtime auth mode once at startup (no auth required).
  useEffect(() => {
    let cancelled = false
    getRuntime()
      .then((rt) => {
        if (!cancelled) setAuthMode(rt.auth)
      })
      .catch(() => {
        // Network/unknown failure: keep the hosted default.
        if (!cancelled) setAuthMode('jwt')
      })
      .finally(() => {
        if (!cancelled) setRuntimeLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const token = localStorage.getItem('kb-token')
    if (!token) {
      // no token — no request, authLoading stays false
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
          setAuthLoading(false)
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
    // In no-auth mode the session is synthesized: authenticated with no token
    // and no synthetic user object. In jwt mode the user-derived check holds.
    isAuthenticated: authMode === 'none' ? true : user !== null,
    authMode,
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
