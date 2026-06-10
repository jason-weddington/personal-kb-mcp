import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Register } from '../pages/Register'

// Mock AuthContext
vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

// Mock api
vi.mock('../api', () => ({
  api: {
    auth: {
      register: vi.fn(),
    },
  },
  ApiError: class ApiError extends Error {
    status: number
    detail: string
    constructor(status: number, detail: string) {
      super(detail)
      this.status = status
      this.detail = detail
    }
  },
}))

// Mock react-router-dom navigate
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

import { useAuth } from '../contexts/AuthContext'
import { api } from '../api'

function renderRegister(initialEntry: string) {
  const mockRegister = vi.fn()
  vi.mocked(useAuth).mockReturnValue({
    isAuthenticated: false,
    user: null,
    loading: false,
    login: vi.fn(),
    register: mockRegister,
    logout: vi.fn(),
  })
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Register />
    </MemoryRouter>,
  )
  return { mockRegister }
}

describe('Register page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows warning Alert and disabled submit when no token', () => {
    renderRegister('/register')

    expect(
      screen.getByText(/invite link from an administrator is required/i),
    ).toBeInTheDocument()
    const submitBtn = screen.getByRole('button', { name: /create account/i })
    expect(submitBtn).toBeDisabled()
  })

  it('calls api.auth.register with correct args when token present and passwords match', async () => {
    const user = userEvent.setup()
    vi.mocked(api.auth.register).mockResolvedValue({
      token: 'tok',
      user: { id: '1', email: 'test@example.com', isAdmin: false, createdAt: '' },
    })

    const { mockRegister } = renderRegister('/register?token=abc123')

    await user.type(screen.getByLabelText(/^email/i), 'test@example.com')
    // MUI appends " *" to required field labels; use non-anchored regex to match "Password *"
    // but not "Confirm Password *" (which doesn't start with "password")
    await user.type(screen.getByLabelText(/^password/i), 'password123')
    await user.type(screen.getByLabelText(/confirm password/i), 'password123')

    const submitBtn = screen.getByRole('button', { name: /create account/i })
    expect(submitBtn).not.toBeDisabled()
    await user.click(submitBtn)

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        'test@example.com',
        'password123',
        'abc123',
      )
    })
  })

  it('shows password mismatch error and does NOT call register', async () => {
    const user = userEvent.setup()
    const { mockRegister } = renderRegister('/register?token=abc123')

    await user.type(screen.getByLabelText(/^email/i), 'test@example.com')
    await user.type(screen.getByLabelText(/^password/i), 'password123')
    await user.type(screen.getByLabelText(/confirm password/i), 'different456')

    await user.click(screen.getByRole('button', { name: /create account/i }))

    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument()
    })
    expect(mockRegister).not.toHaveBeenCalled()
  })
})
