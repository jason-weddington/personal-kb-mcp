/**
 * BrowsePanel tests: tabs + URL state, the New/Top/Stale views, the projects
 * grid and project view, paging, keyboard navigation, debounce, errors and
 * the stale-response race guard.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import type { BrowseEntriesResponse, BrowseEntry, BrowseProjectsResponse } from '../kbTypes'

const openEntry = vi.hoisted(() => vi.fn())

vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: () => ({ openEntry, closeEntry: vi.fn() }),
}))

vi.mock('../api', () => ({
  browseRecent: vi.fn(),
  browseTop: vi.fn(),
  browseStale: vi.fn(),
  browseProjects: vi.fn(),
  browseProject: vi.fn(),
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
  ApiError,
  browseProject,
  browseProjects,
  browseRecent,
  browseStale,
  browseTop,
} from '../api'
import { BrowsePanel } from '../components/browse/BrowsePanel'

// ── Fixtures ─────────────────────────────────────────────────────────────────

function row(id: string, short_title: string, over: Partial<BrowseEntry> = {}): BrowseEntry {
  return {
    id,
    short_title,
    long_title: `${short_title} (long)`,
    entry_type: 'factual_reference',
    project_ref: 'p-alpha',
    created_at: '2026-06-30T00:00:00+00:00',
    updated_at: '2026-06-30T00:00:00+00:00',
    last_accessed: null,
    expires_at: null,
    confidence_level: 1,
    effective_confidence: 0.99,
    half_life_days: 90,
    anchor_age_days: 1,
    expired: false,
    signal_value: null,
    ...over,
  }
}

const ALPHA = row('kb-90001', 'Alpha fact')
const UNSCOPED = row('kb-90009', 'Unscoped expired pattern', {
  entry_type: 'pattern_convention',
  project_ref: null,
  expired: true,
  expires_at: '2026-06-20T00:00:00+00:00',
})
const GAMMA = row('kb-90010', 'Gamma decision', {
  entry_type: 'decision',
  project_ref: 'p-gamma',
})
const ALPHA_MAP = row('kb-90005', 'Alpha map', {
  entry_type: 'mental_map',
  half_life_days: null,
  effective_confidence: 1,
})

function page(
  items: BrowseEntry[],
  over: Partial<BrowseEntriesResponse> = {},
): BrowseEntriesResponse {
  return {
    items,
    next_cursor: null,
    total: items.length,
    skipped: 0,
    signal: null,
    ...over,
  }
}

const PROJECTS: BrowseProjectsResponse = {
  items: [
    {
      project_ref: 'p-alpha',
      entry_count: 3,
      counts_by_type: { factual_reference: 1, decision: 1, mental_map: 1 },
      last_activity: '2026-06-30T00:00:00+00:00',
    },
  ],
  next_cursor: null,
  total: 1,
  skipped: 0,
}

function deferred<T>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

// ── Render helpers ───────────────────────────────────────────────────────────

function LocationProbe() {
  return <div data-testid="location">{useLocation().search}</div>
}

function renderPanel(url = '/') {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <BrowsePanel />
      <LocationProbe />
    </MemoryRouter>,
  )
}

function rows(): HTMLElement[] {
  return screen.queryAllByTestId('browse-row')
}

async function chooseType(label: string, optionName: string) {
  fireEvent.mouseDown(screen.getByRole('combobox', { name: label }))
  const listbox = await screen.findByRole('listbox')
  fireEvent.click(within(listbox).getByRole('option', { name: optionName }))
  // Wait for the refetched rows so the resolution happens inside act().
  await screen.findAllByTestId('browse-row')
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('BrowsePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.setSystemTime(new Date('2026-07-01T00:00:00Z'))
    vi.mocked(browseRecent).mockResolvedValue(page([ALPHA, UNSCOPED]))
    vi.mocked(browseTop).mockResolvedValue(page([], { signal: 'degree' }))
    vi.mocked(browseStale).mockResolvedValue(page([]))
    vi.mocked(browseProjects).mockResolvedValue(PROJECTS)
    vi.mocked(browseProject).mockResolvedValue(page([ALPHA_MAP, ALPHA]))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('(a) defaults to New: one browseRecent call, skeletons, then rows', async () => {
    const d = deferred<BrowseEntriesResponse>()
    vi.mocked(browseRecent).mockReturnValue(d.promise)
    renderPanel('/')

    expect(screen.getByRole('tab', { name: 'New' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getAllByTestId('browse-skeleton')).toHaveLength(8)
    expect(browseRecent).toHaveBeenCalledTimes(1)
    expect(browseRecent).toHaveBeenCalledWith({
      sort: 'created',
      entryType: undefined,
      limit: 25,
      cursor: null,
    })
    expect(browseTop).not.toHaveBeenCalled()
    expect(browseStale).not.toHaveBeenCalled()
    expect(browseProjects).not.toHaveBeenCalled()
    expect(browseProject).not.toHaveBeenCalled()

    await act(async () => d.resolve(page([ALPHA, UNSCOPED])))
    expect(screen.getByText('Alpha fact')).toBeInTheDocument()
    const second = rows()[1]
    expect(within(second).getByText('Unscoped expired pattern')).toBeInTheDocument()
    expect(within(second).getByText('Expired')).toBeInTheDocument()
    expect(within(second).getByText('no project')).toBeInTheDocument()
  })

  it('(b) click and Enter open the entry drawer', async () => {
    renderPanel('/')
    const user = userEvent.setup()
    await user.click(await screen.findByText('Alpha fact'))
    expect(openEntry).toHaveBeenCalledWith('kb-90001')

    rows()[1].focus()
    await user.keyboard('{Enter}')
    expect(openEntry).toHaveBeenCalledWith('kb-90009')
  })

  it('(c) ArrowDown moves focus to the next row', async () => {
    renderPanel('/')
    const user = userEvent.setup()
    await screen.findByText('Alpha fact')
    rows()[0].focus()
    await user.keyboard('{ArrowDown}')
    expect(document.activeElement).toBe(rows()[1])
  })

  it('(d) Load more appends the next page and drops repeats', async () => {
    vi.mocked(browseRecent)
      .mockResolvedValueOnce(page([ALPHA, UNSCOPED], { next_cursor: '2', total: 3 }))
      .mockResolvedValueOnce(page([UNSCOPED, GAMMA], { next_cursor: null, total: 3 }))
    renderPanel('/')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Load more' }))
    expect(browseRecent).toHaveBeenLastCalledWith({
      sort: 'created',
      entryType: undefined,
      limit: 25,
      cursor: '2',
    })
    await screen.findByText('Gamma decision')
    expect(rows()).toHaveLength(3)
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull()
  })

  it('(e) Updated sorts by update time', async () => {
    renderPanel('/')
    const user = userEvent.setup()
    await screen.findByText('Alpha fact')
    await user.click(screen.getByRole('button', { name: 'Updated' }))
    await waitFor(() =>
      expect(browseRecent).toHaveBeenLastCalledWith(
        expect.objectContaining({ sort: 'updated' }),
      ),
    )
  })

  it('(f) the Type filter applies and survives a tab switch', async () => {
    renderPanel('/')
    await screen.findByText('Alpha fact')
    await chooseType('Type', 'decision')
    await waitFor(() =>
      expect(browseRecent).toHaveBeenLastCalledWith(
        expect.objectContaining({ entryType: 'decision' }),
      ),
    )
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'Top' }))
    await waitFor(() =>
      expect(browseTop).toHaveBeenCalledWith({
        signal: 'degree',
        entryType: 'decision',
        limit: 25,
        cursor: null,
      }),
    )
  })

  it('(g) Top shows connections and Recently read', async () => {
    vi.mocked(browseTop).mockResolvedValue(
      page([row('kb-90001', 'Alpha fact', { signal_value: 5 })], { signal: 'degree' }),
    )
    renderPanel('/?view=top')
    expect(await screen.findByText('5 connections')).toBeInTheDocument()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Recently read' }))
    await waitFor(() =>
      expect(browseTop).toHaveBeenLastCalledWith(expect.objectContaining({ signal: 'read' })),
    )
    expect(
      screen.getByText(
        'Ranked by last_accessed: the last kb_get by an agent, or an entry opened here.',
      ),
    ).toBeInTheDocument()
  })

  it('(h) Stale shows decay meta, never-decays and the Expired view', async () => {
    vi.mocked(browseStale).mockResolvedValueOnce(
      page([
        row('kb-90003', 'Beta stale fact', {
          effective_confidence: 0.0301,
          half_life_days: 90,
          anchor_age_days: 365,
        }),
        row('kb-90005', 'Alpha map', { entry_type: 'mental_map', half_life_days: null }),
      ]),
    )
    renderPanel('/?view=stale')
    expect(screen.getByRole('tab', { name: 'Stale' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText(/below 30% after decay/)).toBeInTheDocument()
    expect(
      screen.getByText(/Opening an entry counts as a read and resets its decay clock\./),
    ).toBeInTheDocument()

    const staleRow = (await screen.findByText('Beta stale fact')).closest(
      '[data-testid="browse-row"]',
    ) as HTMLElement
    expect(within(staleRow).getByText('3%')).toBeInTheDocument()
    expect(within(staleRow).getByText('half-life 90d · untouched 365d')).toBeInTheDocument()
    expect(screen.getByText('never decays')).toBeInTheDocument()

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Expired' }))
    await waitFor(() =>
      expect(browseStale).toHaveBeenLastCalledWith(expect.objectContaining({ kind: 'expired' })),
    )
    expect(await screen.findByText('No expired entries.')).toBeInTheDocument()
  })

  it('(i) Projects grid opens a project view with sections and a type filter', async () => {
    renderPanel('/')
    const user = userEvent.setup()
    await screen.findByText('Alpha fact')
    await user.click(screen.getByRole('tab', { name: 'Projects' }))
    expect(browseProjects).toHaveBeenCalledWith({ q: undefined, limit: 24, cursor: null })
    const card = await screen.findByTestId('project-card')
    expect(within(card).getByText('p-alpha')).toBeInTheDocument()
    expect(within(card).getByText(/3 entries/)).toBeInTheDocument()
    expect(screen.queryByRole('combobox', { name: 'Type' })).toBeNull()

    await user.click(screen.getByRole('button', { name: 'Open project p-alpha' }))
    expect(browseProject).toHaveBeenCalledWith('p-alpha', {
      sort: 'newest',
      entryType: undefined,
      q: undefined,
      limit: 25,
      cursor: null,
    })
    const maps = await screen.findByText('Mental maps')
    const mapRow = screen.getByText('Alpha map')
    const entries = screen.getByText('Entries')
    const alphaRow = screen.getByText('Alpha fact')
    expect(maps.compareDocumentPosition(mapRow) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(
      entries.compareDocumentPosition(alphaRow) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    expect(screen.getAllByRole('combobox', { name: 'Type' })).toHaveLength(1)
    await chooseType('Type', 'decision')
    await waitFor(() =>
      expect(browseProject).toHaveBeenLastCalledWith(
        'p-alpha',
        expect.objectContaining({ entryType: 'decision' }),
      ),
    )

    await user.click(screen.getByRole('button', { name: 'Back to projects' }))
    expect(await screen.findByTestId('project-card')).toBeInTheDocument()
  })

  it('(j) the in-project filter is debounced 250 ms', async () => {
    vi.useRealTimers()
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-07-01T00:00:00Z'))
    renderPanel('/?view=projects&project=p-alpha')
    await act(async () => {})
    const input = screen.getByLabelText('Filter in project')
    const qs = () => vi.mocked(browseProject).mock.calls.map((c) => c[1].q)

    fireEvent.change(input, { target: { value: 'd' } })
    await act(async () => vi.advanceTimersByTime(100))
    fireEvent.change(input, { target: { value: 'de' } })
    await act(async () => vi.advanceTimersByTime(100))
    fireEvent.change(input, { target: { value: 'dec' } })
    await act(async () => vi.advanceTimersByTime(249))
    expect(qs()).not.toContain('d')
    expect(qs()).not.toContain('de')
    expect(qs()).not.toContain('dec')
    await act(async () => vi.advanceTimersByTime(1))
    expect(qs().filter((q) => q === 'dec')).toHaveLength(1)
  })

  it('(k) an error shows the message and Retry refetches', async () => {
    vi.mocked(browseRecent)
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce(page([ALPHA]))
    renderPanel('/')
    expect(await screen.findByText('Could not load: boom')).toBeInTheDocument()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('Alpha fact')).toBeInTheDocument()
    expect(browseRecent).toHaveBeenCalledTimes(2)
  })

  it('(l) a stale response never overwrites the latest one', async () => {
    const a = deferred<BrowseEntriesResponse>()
    const b = deferred<BrowseEntriesResponse>()
    vi.mocked(browseRecent).mockReturnValueOnce(a.promise).mockReturnValueOnce(b.promise)
    renderPanel('/')
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Updated' }))
    expect(browseRecent).toHaveBeenCalledTimes(2)
    await act(async () => b.resolve(page([GAMMA])))
    await act(async () => a.resolve(page([ALPHA])))
    expect(screen.getByText('Gamma decision')).toBeInTheDocument()
    expect(screen.queryByText('Alpha fact')).toBeNull()
  })

  it('(m) unknown view and type params fall back', async () => {
    const { unmount } = renderPanel('/?view=bogus')
    expect(screen.getByRole('tab', { name: 'New' })).toHaveAttribute('aria-selected', 'true')
    await screen.findByText('Alpha fact')
    unmount()

    vi.mocked(browseRecent).mockClear()
    renderPanel('/?type=bogus')
    await screen.findByText('Alpha fact')
    expect(browseRecent).toHaveBeenCalledWith(
      expect.objectContaining({ entryType: undefined }),
    )
  })

  it('(n) an unknown project shows a warning and keeps the back button', async () => {
    vi.mocked(browseProject).mockRejectedValue(new ApiError(404, 'Unknown project: p-none'))
    renderPanel('/?view=projects&project=p-none')
    expect(await screen.findByText('Project not found: p-none')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Back to projects' })).toBeInTheDocument()
  })

  it('(o) switching tabs keeps type and drops project and view=new', async () => {
    renderPanel('/?view=projects&project=p-alpha&type=decision')
    await screen.findByText('Alpha fact')
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'New' }))
    const search = screen.getByTestId('location').textContent ?? ''
    const params = new URLSearchParams(search)
    expect(params.get('type')).toBe('decision')
    expect(params.has('project')).toBe(false)
    expect(params.has('view')).toBe(false)
  })

  it('(p) the projects filter is debounced 250 ms', async () => {
    vi.useRealTimers()
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-07-01T00:00:00Z'))
    renderPanel('/?view=projects')
    await act(async () => {})
    const qs = () => vi.mocked(browseProjects).mock.calls.map((c) => c[0].q)
    fireEvent.change(screen.getByLabelText('Filter projects'), { target: { value: 'ga' } })
    await act(async () => vi.advanceTimersByTime(249))
    expect(qs()).not.toContain('ga')
    await act(async () => vi.advanceTimersByTime(1))
    expect(qs().filter((q) => q === 'ga')).toHaveLength(1)
  })

  it('(q) the caption shows the total and unreadable rows', async () => {
    vi.mocked(browseRecent).mockResolvedValueOnce(page([ALPHA], { total: 1 }))
    const { unmount } = renderPanel('/')
    expect(await screen.findByText('1 entry')).toBeInTheDocument()
    unmount()

    vi.mocked(browseRecent).mockResolvedValueOnce(page([ALPHA], { total: 1, skipped: 2 }))
    renderPanel('/')
    expect(await screen.findByText(/· 2 unreadable/)).toBeInTheDocument()
  })
})
