/**
 * Home page tests:
 * - Stats card: renders fetched numbers from three endpoints
 * - Stats card: partial failure renders em dash, not an error
 * - No fetch storm: exactly one call per endpoint on mount
 * - The Browse panel renders (tablist), there is no Ask question field, and
 *   nothing on Home fetches the full graph
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

const EMPTY_PAGE = vi.hoisted(() => ({
  items: [],
  next_cursor: null,
  total: 0,
  skipped: 0,
  signal: null,
}))

// api mock — must be declared before import below
vi.mock('../api', () => ({
  getToken: vi.fn(() => 'test-token'),
  listKbProjects: vi.fn(),
  listKbContributors: vi.fn(),
  getMapsIndex: vi.fn(),
  browseRecent: vi.fn(() => Promise.resolve(EMPTY_PAGE)),
  browseTop: vi.fn(() => Promise.resolve(EMPTY_PAGE)),
  browseStale: vi.fn(() => Promise.resolve(EMPTY_PAGE)),
  browseProjects: vi.fn(() => Promise.resolve(EMPTY_PAGE)),
  browseProject: vi.fn(() => Promise.resolve(EMPTY_PAGE)),
  getGraphFull: vi.fn(),
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

import {
  listKbProjects,
  listKbContributors,
  getMapsIndex,
  getGraphFull,
  browseRecent,
  browseProjects,
} from '../api'

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

  it('renders the Browse panel and no Ask field', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })

    renderHome()
    expect(screen.getByText('Browse the KB')).toBeInTheDocument()
    expect(await screen.findByRole('tablist', { name: 'Browse views' })).toBeInTheDocument()
    expect(screen.queryByLabelText(/question/i)).toBeNull()
    // Flush pending fetches to avoid act() warnings
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
    await waitFor(() => expect(screen.getByText('No entries yet.')).toBeInTheDocument())
    expect(vi.mocked(getGraphFull)).not.toHaveBeenCalled()
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
    vi.mocked(browseRecent).mockResolvedValue({ ...EMPTY_PAGE, total: 55 })
    vi.mocked(browseProjects).mockResolvedValue({
      ...EMPTY_PAGE,
      items: [
        { project_ref: 'proj-a', entry_count: 42, counts_by_type: {}, last_activity: '' },
        { project_ref: 'proj-b', entry_count: 8, counts_by_type: {}, last_activity: '' },
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

    // Entries = unfiltered browse total (55); 42 + 8 in projects, 5 without
    await waitFor(() =>
      expect(screen.getByTestId('stat-value-entries').textContent).toBe('55'),
    )
    await waitFor(() =>
      expect(screen.getByTestId('stat-value-no project').textContent).toBe('5'),
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
    vi.mocked(browseRecent).mockRejectedValue(new Error('Network error'))
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
