/**
 * Home page tests:
 * - Stats card: renders fetched numbers from three endpoints
 * - Stats card: partial failure renders em dash, not an error
 * - No fetch storm: exactly one call per endpoint on mount
 * - Welcome heading is shown
 * - AskPanel is rendered (presence of Question textarea)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Home } from '../pages/Home'

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(() => ({
    isAuthenticated: true,
    user: { id: 'u1', email: 'user@example.com', isAdmin: false, createdAt: '' },
    loading: false,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  })),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: vi.fn(() => ({ openEntry: vi.fn(), closeEntry: vi.fn() })),
  EntryDrawerProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock('../streaming', () => ({
  streamSSE: vi.fn(),
}))

// api mock — must be declared before import below
vi.mock('../api', () => ({
  getToken: vi.fn(() => 'test-token'),
  listKbProjects: vi.fn(),
  listKbContributors: vi.fn(),
  getMapsIndex: vi.fn(),
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

import { listKbProjects, listKbContributors, getMapsIndex } from '../api'

// ── Helpers ───────────────────────────────────────────────────────────────────

function renderHome() {
  return render(
    <MemoryRouter>
      <Home />
    </MemoryRouter>,
  )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('Home page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows the welcome heading', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })

    renderHome()
    expect(screen.getByText('Welcome to Personal KB')).toBeInTheDocument()
    // Flush pending stat fetches to avoid act() warnings
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
  })

  it('renders the Ask question textarea (AskPanel is present)', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })

    renderHome()
    expect(screen.getByLabelText(/question/i)).toBeInTheDocument()
    // Flush pending stat fetches to avoid act() warnings
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
  })

  it('renders fetched stats numbers', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({
      items: [
        { name: 'proj-a', entry_count: 42 },
        { name: 'proj-b', entry_count: 8 },
      ],
    })
    vi.mocked(listKbContributors).mockResolvedValue({
      items: [
        { name: 'alice', entry_count: 30 },
        { name: 'bob', entry_count: 20 },
      ],
    })
    vi.mocked(getMapsIndex).mockResolvedValue({
      projects: [
        {
          project_ref: 'proj-a',
          maps: [
            { id: 'kb-00001', short_title: 'Map 1', long_title: 'Long Map 1' },
            { id: 'kb-00002', short_title: 'Map 2', long_title: 'Long Map 2' },
          ],
        },
      ],
    })

    renderHome()

    // Entries = 42 + 8 = 50
    await waitFor(() =>
      expect(screen.getByTestId('stat-value-entries').textContent).toBe('50'),
    )
    // Projects = 2 items
    expect(screen.getByTestId('stat-value-projects').textContent).toBe('2')
    // Contributors = 2 items
    expect(screen.getByTestId('stat-value-contributors').textContent).toBe('2')
    // Maps = 2 maps across 1 project
    expect(screen.getByTestId('stat-value-maps').textContent).toBe('2')
  })

  it('partial failure: failed stat renders em dash, page still works', async () => {
    vi.mocked(listKbProjects).mockRejectedValue(new Error('Network error'))
    vi.mocked(listKbContributors).mockResolvedValue({
      items: [{ name: 'alice', entry_count: 5 }],
    })
    vi.mocked(getMapsIndex).mockRejectedValue(new Error('Network error'))

    renderHome()

    // Contributors resolved successfully
    await waitFor(() =>
      expect(screen.getByTestId('stat-value-contributors').textContent).toBe('1'),
    )

    // Failed stats show em dash
    expect(screen.getByTestId('stat-value-entries').textContent).toBe('—')
    expect(screen.getByTestId('stat-value-projects').textContent).toBe('—')
    expect(screen.getByTestId('stat-value-maps').textContent).toBe('—')
  })

  it('no fetch storm: exactly one call per endpoint on mount', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })

    renderHome()

    expect(vi.mocked(listKbProjects)).toHaveBeenCalledTimes(1)
    expect(vi.mocked(listKbContributors)).toHaveBeenCalledTimes(1)
    expect(vi.mocked(getMapsIndex)).toHaveBeenCalledTimes(1)
    // Flush pending stat fetches to avoid act() warnings
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
  })

  it('shows KB Stats heading', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })

    renderHome()
    expect(screen.getByText('KB Stats')).toBeInTheDocument()
    // Flush pending stat fetches to avoid act() warnings
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
  })
})
