/**
 * Home page tests:
 * - Stats card: renders fetched numbers from three endpoints
 * - Stats card: partial failure renders em dash, not an error
 * - No fetch storm: exactly one call per endpoint on mount
 * - The Browse panel renders (tablist), there is no Ask question field, and
 *   nothing on Home fetches the full graph
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Home } from '../pages/Home'
import type { IngestFileResult } from '../kbTypes'

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
  ingestFile: vi.fn(),
  ingestUrl: vi.fn(),
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
  ingestFile,
} from '../api'

// ── Helpers ───────────────────────────────────────────────────────────────────

function deferred<T>() {
  let resolve!: (v: T) => void
  const promise = new Promise<T>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

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

    // Two listKbProjects calls: StatsCard + the Add to KB panel's suggestions.
    expect(vi.mocked(listKbProjects)).toHaveBeenCalledTimes(2)
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

  it('shows the Add to KB panel under the stats card', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })

    renderHome()
    expect(screen.getByText('Add to KB')).toBeInTheDocument()
    expect(screen.getByText('Drop files or URLs here')).toBeInTheDocument()
    // Flush pending stat fetches to avoid act() warnings
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
  })

  it('refreshes the stats and the browse list after an ingested job', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})

    const ingested: IngestFileResult = {
      path: 'p',
      action: 'ingested',
      reason: null,
      entry_count: 2,
      entry_ids: ['kb-1', 'kb-2'],
      summary: 's',
      chunks_processed: 3,
      chunks_skipped: 0,
      chunks_flagged: 0,
    }
    vi.mocked(browseRecent).mockResolvedValue(EMPTY_PAGE)
    const fileDefer = deferred<IngestFileResult>()
    vi.mocked(ingestFile).mockImplementation(() => fileDefer.promise)

    renderHome()
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
    await screen.findByText('No entries yet.')

    fireEvent.drop(screen.getByTestId('add-to-kb-drop'), {
      dataTransfer: { files: [new File(['x'], 'a.md')], getData: () => '' },
    })
    await waitFor(() => expect(vi.mocked(ingestFile)).toHaveBeenCalledTimes(1))

    const before = vi.mocked(browseRecent).mock.calls.length
    await act(async () => {
      fileDefer.resolve(ingested)
    })
    expect(info).toHaveBeenCalledWith('[add-to-kb] refresh', {
      action: 'ingested',
      entry_count: 2,
      entry_ids_count: 2,
    })
    await screen.findByText('Ingested · 2 entries · 3 chunks')
    // StatsCard's countEntries re-runs and the 'new' EntryList refetches: +2.
    await waitFor(() => expect(vi.mocked(browseRecent).mock.calls.length).toBe(before + 2))
    vi.restoreAllMocks()
  })

  it('a skipped job does not refresh and does not log the refresh', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    vi.mocked(listKbContributors).mockResolvedValue({ items: [] })
    vi.mocked(getMapsIndex).mockResolvedValue({ projects: [] })
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})

    const skipped: IngestFileResult = {
      path: 'p',
      action: 'skipped',
      reason: 'duplicate',
      entry_count: 0,
      entry_ids: [],
      summary: null,
      chunks_processed: 0,
      chunks_skipped: 0,
      chunks_flagged: 0,
    }
    vi.mocked(browseRecent).mockResolvedValue(EMPTY_PAGE)
    const fileDefer = deferred<IngestFileResult>()
    vi.mocked(ingestFile).mockImplementation(() => fileDefer.promise)

    renderHome()
    await waitFor(() => expect(screen.getByTestId('stat-value-entries')).toBeInTheDocument())
    await screen.findByText('No entries yet.')

    fireEvent.drop(screen.getByTestId('add-to-kb-drop'), {
      dataTransfer: { files: [new File(['x'], 'a.md')], getData: () => '' },
    })
    await waitFor(() => expect(vi.mocked(ingestFile)).toHaveBeenCalledTimes(1))

    const before = vi.mocked(browseRecent).mock.calls.length
    await act(async () => {
      fileDefer.resolve(skipped)
    })
    await screen.findByText('Skipped · duplicate')
    // Give any spurious refetch a chance to fire, then assert none did.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20))
    })
    expect(vi.mocked(browseRecent).mock.calls.length).toBe(before)
    expect(info).not.toHaveBeenCalledWith('[add-to-kb] refresh', expect.anything())
    vi.restoreAllMocks()
  })
})
