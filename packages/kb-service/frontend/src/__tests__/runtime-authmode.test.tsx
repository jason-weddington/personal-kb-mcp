/**
 * Auth-mode awareness: the SPA hides hosted-only auth UI in local (no-auth) mode.
 *
 * Unlike the other suites that mock `useAuth` directly, this suite mounts the
 * REAL AuthProvider and mocks the `../api` module so that `getRuntime` actually
 * drives `authMode` end-to-end (the AC#13 wiring crux). Each surface is asserted
 * in BOTH modes:
 *   - getRuntime => {auth:'none'}: Settings hides AccountCard / ApiAccessCard
 *     (+ MCP snippet) / TeamCard; ProtectedRoute renders children without
 *     redirecting to /login; Sidebar hides the Admin section.
 *   - getRuntime => {auth:'jwt'}: ApiAccessCard, TeamCard, AccountCard and the
 *     Admin sidebar section DO render (guards against hiding them in both modes).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from '../contexts/AuthContext'
import { Settings } from '../pages/Settings'
import Sidebar from '../components/Sidebar'
import { ProtectedRoute } from '../components/ProtectedRoute'

// Mock the api module — getRuntime is the lever that selects auth mode.
vi.mock('../api', () => ({
  getRuntime: vi.fn(),
  api: {
    auth: {
      me: vi.fn(),
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    },
    apiKeys: { list: vi.fn(), create: vi.fn(), revoke: vi.fn() },
    settings: { get: vi.fn(), put: vi.fn() },
  },
  ApiError: class ApiError extends Error {
    status: number
    detail: string
    constructor(status: number, detail: string) {
      super(detail)
      this.status = status
      this.detail = detail
      this.name = 'ApiError'
    }
  },
}))

import { getRuntime, api } from '../api'
import type { UserResponse } from '../types'

const ADMIN: UserResponse = {
  id: 'u1',
  email: 'admin@example.com',
  isAdmin: true,
  createdAt: '',
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  vi.mocked(api.apiKeys.list).mockResolvedValue({ keys: [] })
  vi.mocked(api.settings.get).mockResolvedValue({ team: null })
  vi.mocked(api.auth.me).mockResolvedValue(ADMIN)
})

function renderWithAuth(ui: React.ReactNode) {
  return render(
    <AuthProvider>
      <MemoryRouter>{ui}</MemoryRouter>
    </AuthProvider>,
  )
}

describe('Settings — auth-mode gating', () => {
  it('no-auth mode: hides AccountCard, ApiAccessCard (+ MCP snippet) and TeamCard', async () => {
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'none' })
    renderWithAuth(<Settings />)

    // authMode defaults to 'jwt' while pending, so wait for the 'none' resolution
    // to remove the API Access card.
    await waitFor(() =>
      expect(screen.queryByText('API Access')).not.toBeInTheDocument(),
    )
    expect(screen.queryByText('Account')).not.toBeInTheDocument()
    expect(screen.queryByText('Team')).not.toBeInTheDocument()
    expect(screen.queryByText(/Signed in as/i)).not.toBeInTheDocument()
    // The MCP snippet lives inside ApiAccessCard — its env keys must not appear.
    expect(screen.queryByText(/PERSONAL_KB_API_KEY/)).not.toBeInTheDocument()
    expect(screen.queryByText(/MCP_TOOL_PREFIX/)).not.toBeInTheDocument()
    // The page itself still renders.
    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  it('jwt mode: renders AccountCard, ApiAccessCard and TeamCard', async () => {
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'jwt' })
    renderWithAuth(<Settings />)

    await waitFor(() =>
      expect(screen.getByText('API Access')).toBeInTheDocument(),
    )
    expect(screen.getByText('Account')).toBeInTheDocument()
    expect(screen.getByText('Team')).toBeInTheDocument()
  })
})

describe('Sidebar — auth-mode gating', () => {
  it('no-auth mode: does not render the Admin section', async () => {
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'none' })
    renderWithAuth(<Sidebar open />)

    await waitFor(() => expect(vi.mocked(getRuntime)).toHaveBeenCalled())
    expect(screen.queryByText('Admin')).not.toBeInTheDocument()
    expect(screen.queryByText('Invites')).not.toBeInTheDocument()
    expect(screen.queryByText('Users')).not.toBeInTheDocument()
  })

  it('jwt mode: renders the Admin section for an admin user', async () => {
    localStorage.setItem('kb-token', 'tok')
    localStorage.setItem('kb-user', JSON.stringify(ADMIN))
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'jwt' })
    renderWithAuth(<Sidebar open />)

    await waitFor(() => expect(screen.getByText('Admin')).toBeInTheDocument())
    expect(screen.getByText('Invites')).toBeInTheDocument()
    expect(screen.getByText('Users')).toBeInTheDocument()
  })
})

describe('ProtectedRoute — auth-mode gating', () => {
  it('no-auth mode: renders children without redirecting to /login', async () => {
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'none' })
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/protected']}>
          <Routes>
            <Route
              path="/protected"
              element={
                <ProtectedRoute>
                  <div>Protected Content</div>
                </ProtectedRoute>
              }
            />
            <Route path="/login" element={<div>Login Page</div>} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    )

    await waitFor(() =>
      expect(screen.getByText('Protected Content')).toBeInTheDocument(),
    )
    expect(screen.queryByText('Login Page')).not.toBeInTheDocument()
  })
})
