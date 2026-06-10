import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { AdminRoute } from '../components/AdminRoute'
import { AdminInvites } from '../pages/AdminInvites'

// Mock AuthContext
vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

// Mock api module
vi.mock('../api', () => ({
  api: {
    admin: {
      invites: {
        list: vi.fn(),
        create: vi.fn(),
        revoke: vi.fn(),
      },
      users: {
        list: vi.fn(),
        promote: vi.fn(),
        issuePasswordReset: vi.fn(),
        deleteUser: vi.fn(),
      },
    },
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

import { useAuth } from '../contexts/AuthContext'
import { api } from '../api'

describe('AdminRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('redirects non-admin user away from /admin/users to /', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: true,
      user: { id: 'u1', email: 'user@example.com', isAdmin: false, createdAt: '' },
      loading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    render(
      <MemoryRouter initialEntries={['/admin/users']}>
        <Routes>
          <Route
            path="/admin/users"
            element={
              <AdminRoute>
                <div>Admin Users Page</div>
              </AdminRoute>
            }
          />
          <Route path="/" element={<div>Home Page</div>} />
          <Route path="/login" element={<div>Login Page</div>} />
        </Routes>
      </MemoryRouter>,
    )

    expect(screen.getByText('Home Page')).toBeInTheDocument()
    expect(screen.queryByText('Admin Users Page')).not.toBeInTheDocument()
  })

  it('shows CircularProgress when loading', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: false,
      user: null,
      loading: true,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    render(
      <MemoryRouter>
        <AdminRoute>
          <div>Admin Content</div>
        </AdminRoute>
      </MemoryRouter>,
    )

    expect(screen.getByRole('progressbar')).toBeInTheDocument()
    expect(screen.queryByText('Admin Content')).not.toBeInTheDocument()
  })

  it('redirects unauthenticated user to /login', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: false,
      user: null,
      loading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    render(
      <MemoryRouter initialEntries={['/admin/users']}>
        <Routes>
          <Route
            path="/admin/users"
            element={
              <AdminRoute>
                <div>Admin Users Page</div>
              </AdminRoute>
            }
          />
          <Route path="/login" element={<div>Login Page</div>} />
        </Routes>
      </MemoryRouter>,
    )

    expect(screen.getByText('Login Page')).toBeInTheDocument()
    expect(screen.queryByText('Admin Users Page')).not.toBeInTheDocument()
  })

  it('renders children for admin user', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: true,
      user: { id: 'u1', email: 'admin@example.com', isAdmin: true, createdAt: '' },
      loading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    render(
      <MemoryRouter>
        <AdminRoute>
          <div>Admin Content</div>
        </AdminRoute>
      </MemoryRouter>,
    )

    expect(screen.getByText('Admin Content')).toBeInTheDocument()
  })
})

describe('AdminInvites — create invite dialog shows URL once', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: true,
      user: { id: 'u1', email: 'admin@example.com', isAdmin: true, createdAt: '' },
      loading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })
    vi.mocked(api.admin.invites.list).mockResolvedValue([])
  })

  it('create dialog shows the invite URL after creation', async () => {
    const user = userEvent.setup()
    const createdInvite = {
      token: 'tok-abc123',
      url: 'https://example.com/register?token=tok-abc123',
      note: 'For Alice',
      createdAt: new Date().toISOString(),
    }
    vi.mocked(api.admin.invites.create).mockResolvedValue(createdInvite)

    render(
      <MemoryRouter>
        <AdminInvites />
      </MemoryRouter>,
    )

    await waitFor(() =>
      expect(vi.mocked(api.admin.invites.list)).toHaveBeenCalled(),
    )

    // Open create dialog
    await user.click(screen.getByRole('button', { name: /new invite/i }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    // Submit with a note
    const noteField = screen.getByLabelText(/note/i)
    await user.type(noteField, 'For Alice')
    await user.click(screen.getByRole('button', { name: /^create$/i }))

    // Result dialog shows the URL
    await waitFor(() =>
      expect(
        screen.getByText(/invite created/i),
      ).toBeInTheDocument(),
    )

    const resultDialog = screen.getByRole('dialog')
    const urlField = within(resultDialog).getByRole('textbox')
    expect((urlField as HTMLInputElement).value).toBe(createdInvite.url)
  })
})
