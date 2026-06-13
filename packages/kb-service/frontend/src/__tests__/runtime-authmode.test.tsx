/**
 * Auth-mode awareness: the SPA hides hosted-only auth UI in local (no-auth) mode.
 *
 * Unlike the other suites that mock `useAuth` directly, this suite mounts the
 * REAL AuthProvider and mocks the `../api` module so that `getRuntime` actually
 * drives `authMode` end-to-end (the AC#13 wiring crux). Each surface is asserted
 * in BOTH modes:
 *   - getRuntime => {auth:'none'}: Settings hides AccountCard / ApiAccessCard
 *     (+ MCP snippet) / TeamCard; ProtectedRoute renders children without
 *     redirecting to /login; Sidebar hides the Admin section AND the Chat nav
 *     entry; deep-linking to /chat redirects to Home.
 *   - getRuntime => {auth:'jwt'}: ApiAccessCard, TeamCard, AccountCard, the
 *     Admin sidebar section, the Chat nav entry, and the /chat route all
 *     render (guards against hiding them in both modes).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from '../contexts/AuthContext'
import { ThemeProvider } from '../contexts/ThemeContext'
import { Settings } from '../pages/Settings'
import Sidebar from '../components/Sidebar'
import { ProtectedRoute } from '../components/ProtectedRoute'
import { App } from '../App'

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

  it('no-auth mode: hides the Chat nav entry (hosted-only)', async () => {
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'none' })
    renderWithAuth(<Sidebar open />)

    await waitFor(() => expect(vi.mocked(getRuntime)).toHaveBeenCalled())
    // Chat is hidden: its backend endpoints hit the auth DB and 500 without one.
    expect(screen.queryByText('Chat')).not.toBeInTheDocument()
    // The other always-on entries still render.
    expect(screen.getByText('Home')).toBeInTheDocument()
    expect(screen.getByText('Search')).toBeInTheDocument()
    expect(screen.getByText('Graph')).toBeInTheDocument()
    expect(screen.getByText('Settings')).toBeInTheDocument()
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

  it('jwt mode: renders the Chat nav entry', async () => {
    localStorage.setItem('kb-token', 'tok')
    localStorage.setItem('kb-user', JSON.stringify(ADMIN))
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'jwt' })
    renderWithAuth(<Sidebar open />)

    await waitFor(() => expect(screen.getByText('Chat')).toBeInTheDocument())
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

// Stub the heavy page components so the App's routing — not their bodies —
// is what's under test for the /chat redirect.
vi.mock('../pages/Home', () => ({
  Home: () => <div>HOME-PAGE</div>,
}))
vi.mock('../pages/Chat', () => ({
  Chat: () => <div>CHAT-PAGE</div>,
}))
vi.mock('../pages/Search', () => ({
  Search: () => <div>SEARCH-PAGE</div>,
}))
vi.mock('../pages/Graph', () => ({
  Graph: () => <div>GRAPH-PAGE</div>,
}))
vi.mock('../pages/EntryDetail', () => ({
  EntryDetail: () => <div>ENTRY-PAGE</div>,
}))
vi.mock('../pages/AdminInvites', () => ({
  AdminInvites: () => <div>ADMIN-INVITES</div>,
}))
vi.mock('../pages/AdminUsers', () => ({
  AdminUsers: () => <div>ADMIN-USERS</div>,
}))

function renderApp(initialPath: string) {
  return render(
    <AuthProvider>
      <ThemeProvider>
        <MemoryRouter initialEntries={[initialPath]}>
          <App />
        </MemoryRouter>
      </ThemeProvider>
    </AuthProvider>,
  )
}

describe('App routing — /chat gating', () => {
  it('no-auth mode: visiting /chat redirects to Home', async () => {
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'none' })
    renderApp('/chat')

    // Once authMode resolves to 'none', the /chat route renders Navigate→"/",
    // landing on the Home stub. The Chat body must never appear.
    await waitFor(() =>
      expect(screen.getByText('HOME-PAGE')).toBeInTheDocument(),
    )
    expect(screen.queryByText('CHAT-PAGE')).not.toBeInTheDocument()
  })

  it('jwt mode: visiting /chat renders the Chat page', async () => {
    localStorage.setItem('kb-token', 'tok')
    localStorage.setItem('kb-user', JSON.stringify(ADMIN))
    vi.mocked(getRuntime).mockResolvedValue({ auth: 'jwt' })
    renderApp('/chat')

    await waitFor(() =>
      expect(screen.getByText('CHAT-PAGE')).toBeInTheDocument(),
    )
    expect(screen.queryByText('HOME-PAGE')).not.toBeInTheDocument()
  })
})
