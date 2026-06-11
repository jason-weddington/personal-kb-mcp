import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { EntryDrawerProvider, useEntryDrawer } from '../contexts/EntryDrawerContext'
import type { GetResponse } from '../kbTypes'

// Mock api — real EntryDrawerProvider + EntryDrawer used throughout
vi.mock('../api', () => ({
  getEntries: vi.fn(),
  getGraphNeighbors: vi.fn(),
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

import { getEntries, getGraphNeighbors } from '../api'

// Helper entry factory
function makeEntry(
  id: string,
  opts: { short_title?: string; superseded_by?: string | null } = {},
) {
  return {
    id,
    short_title: opts.short_title ?? `Title for ${id}`,
    long_title: opts.short_title ?? `Title for ${id}`,
    knowledge_details: `Details for ${id}`,
    entry_type: 'decision' as const,
    tags: [],
    project_ref: null,
    contributor: null,
    team: null,
    updated_by: null,
    sensitivity: null,
    confidence_level: 0.9,
    hints: {},
    created_at: null,
    updated_at: null,
    last_accessed: null,
    expires_at: null,
    superseded_by: opts.superseded_by ?? null,
    is_active: true,
    has_embedding: true,
    version: 1,
    source_context: null,
  }
}

function makeGetResponse(
  id: string,
  opts: { short_title?: string; superseded_by?: string | null } = {},
): GetResponse {
  return {
    results: [
      {
        id,
        found: true,
        entry: makeEntry(id, opts),
        pointer_rot: [],
      },
    ],
  }
}

// Test consumer: calls openEntry
function OpenButton({ entryId }: { entryId: string }) {
  const { openEntry } = useEntryDrawer()
  return (
    <button onClick={() => openEntry(entryId)}>Open {entryId}</button>
  )
}

function renderWithProvider(children: React.ReactNode) {
  return render(
    <MemoryRouter>
      <EntryDrawerProvider>
        {children}
      </EntryDrawerProvider>
    </MemoryRouter>,
  )
}

describe('EntryDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default: empty neighbors so Connections section stays absent in unrelated tests
    vi.mocked(getGraphNeighbors).mockResolvedValue({ neighbors: [] })
  })

  // (a) Provider mounted, drawer closed → zero getEntries calls (kb-01449)
  it('(a) does not fetch when drawer is closed', () => {
    renderWithProvider(<div>content</div>)
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(0)
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledTimes(0)
  })

  // (b) openEntry → drawer opens, getEntries called once, short_title visible
  it('(b) opens drawer and fetches entry on openEntry', async () => {
    vi.mocked(getEntries).mockResolvedValue(
      makeGetResponse('kb-00001', { short_title: 'My Entry' }),
    )

    renderWithProvider(<OpenButton entryId="kb-00001" />)

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))

    await waitFor(() =>
      expect(vi.mocked(getEntries)).toHaveBeenCalledWith(['kb-00001']),
    )

    // short_title appears in both header (h6) and EntryContent (h5) — use getAllByText
    await waitFor(() =>
      expect(screen.getAllByText('My Entry')[0]).toBeInTheDocument(),
    )
  })

  // (c) clicking superseded_by ref → pushes stack, Back button appears; clicking Back re-shows first
  it('(c) within-drawer navigation: push on superseded_by click, Back returns to first entry', async () => {
    vi.mocked(getEntries).mockImplementation(async (ids: string[]) => {
      const id = ids[0]
      if (id === 'kb-00001') {
        return makeGetResponse('kb-00001', {
          short_title: 'First Entry',
          superseded_by: 'kb-00002',
        })
      }
      return makeGetResponse('kb-00002', { short_title: 'Second Entry' })
    })

    renderWithProvider(<OpenButton entryId="kb-00001" />)

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))

    // First entry loaded — appears in both h6 (header) and h5 (content)
    await waitFor(() =>
      expect(screen.getAllByText('First Entry')[0]).toBeInTheDocument(),
    )

    // Back button NOT visible (stack depth = 1)
    expect(screen.queryByRole('button', { name: 'Back' })).toBeNull()

    // Click the superseded_by button (renders as a MUI Link button)
    await user.click(screen.getByRole('button', { name: 'kb-00002' }))

    // Second entry loads
    await waitFor(() =>
      expect(screen.getAllByText('Second Entry')[0]).toBeInTheDocument(),
    )
    // Back button now visible (stack depth = 2)
    expect(screen.getByRole('button', { name: 'Back' })).toBeInTheDocument()

    // Click Back
    await user.click(screen.getByRole('button', { name: 'Back' }))

    // First entry re-shown
    await waitFor(() =>
      expect(screen.getAllByText('First Entry')[0]).toBeInTheDocument(),
    )
    // Back button hidden again (stack depth = 1)
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Back' })).toBeNull(),
    )
  })

  // (d) Close button → subsequent reopen refetches (kb-01449 gate reset on close)
  it('(d) Close button closes drawer; subsequent reopen triggers refetch', async () => {
    vi.mocked(getEntries).mockResolvedValue(
      makeGetResponse('kb-00001', { short_title: 'Entry One' }),
    )

    renderWithProvider(<OpenButton entryId="kb-00001" />)

    const user = userEvent.setup()

    // Open the drawer
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))
    await waitFor(() =>
      expect(screen.getAllByText('Entry One')[0]).toBeInTheDocument(),
    )
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(1)

    // Close via Close button (aria-label="Close") in drawer header
    await user.click(screen.getByRole('button', { name: 'Close' }))

    // Wait for the drawer to close (MUI removes aria-hidden from siblings)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Open kb-00001' })).toBeInTheDocument(),
    )

    // Reopen — triggers a second fetch since stack was reset to []
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))
    await waitFor(() =>
      expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(2),
    )
    await waitFor(() =>
      expect(screen.getAllByText('Entry One')[0]).toBeInTheDocument(),
    )
  })
})

describe('EntryDrawer — Connections section', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default: empty neighbors so section stays absent unless overridden
    vi.mocked(getGraphNeighbors).mockResolvedValue({ neighbors: [] })
  })

  // (e) Closed drawer — getGraphNeighbors never called (kb-01449)
  it('(e) closed drawer: getGraphNeighbors not called', () => {
    renderWithProvider(<div>content</div>)
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledTimes(0)
  })

  // (f) Entry neighbor chip is clickable and opens the neighbor in the drawer
  it('(f) entry neighbor chip opens in drawer via openEntry', async () => {
    vi.mocked(getEntries).mockImplementation(async (ids: string[]) => {
      // Initial load of kb-00001, and title resolution for kb-00002
      return makeGetResponse(ids[0], {
        short_title: ids[0] === 'kb-00001' ? 'Main Entry' : 'Neighbor Entry',
      })
    })
    vi.mocked(getGraphNeighbors).mockResolvedValue({
      neighbors: [
        { neighbor_id: 'kb-00002', edge_type: 'related', direction: 'outgoing' },
      ],
    })

    renderWithProvider(<OpenButton entryId="kb-00001" />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))

    // Main entry loads
    await waitFor(() =>
      expect(screen.getAllByText('Main Entry')[0]).toBeInTheDocument(),
    )

    // Connections section header appears
    await waitFor(() =>
      expect(screen.getByText('Connections')).toBeInTheDocument(),
    )

    // Neighbor chip visible with resolved title; label: "[kb-00002] Neighbor Entry"
    const neighborChip = await screen.findByRole('button', {
      name: '[kb-00002] Neighbor Entry',
    })
    expect(neighborChip).toBeInTheDocument()

    // Click neighbor chip → opens kb-00002 in the drawer
    await user.click(neighborChip)

    await waitFor(() =>
      expect(screen.getAllByText('Neighbor Entry')[0]).toBeInTheDocument(),
    )
  })

  // (g) Non-entry node chip is not clickable (no button role)
  it('(g) non-entry neighbor chip is not clickable', async () => {
    vi.mocked(getEntries).mockResolvedValue(
      makeGetResponse('kb-00001', { short_title: 'My Entry' }),
    )
    vi.mocked(getGraphNeighbors).mockResolvedValue({
      neighbors: [
        { neighbor_id: 'tag:foo', edge_type: 'tagged', direction: 'outgoing' },
      ],
    })

    renderWithProvider(<OpenButton entryId="kb-00001" />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))

    await waitFor(() =>
      expect(screen.getAllByText('My Entry')[0]).toBeInTheDocument(),
    )

    // Non-entry chip present
    await waitFor(() =>
      expect(screen.getByText('tag:foo')).toBeInTheDocument(),
    )

    // Non-entry chip must NOT be a button
    expect(screen.queryByRole('button', { name: 'tag:foo' })).toBeNull()
  })

  // (h) getGraphNeighbors error → Connections section absent; entry body still renders
  it('(h) fetch error: Connections section absent, entry body unaffected', async () => {
    vi.mocked(getEntries).mockResolvedValue(
      makeGetResponse('kb-00001', { short_title: 'My Entry' }),
    )
    vi.mocked(getGraphNeighbors).mockRejectedValue(new Error('network error'))

    renderWithProvider(<OpenButton entryId="kb-00001" />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))

    // Entry body still renders
    await waitFor(() =>
      expect(screen.getAllByText('My Entry')[0]).toBeInTheDocument(),
    )

    // Connections header must be absent
    expect(screen.queryByText('Connections')).toBeNull()
  })

  // (i) Switching entries (via within-drawer navigation) refreshes getGraphNeighbors
  it('(i) switching entries re-fetches getGraphNeighbors', async () => {
    // kb-00001 supersedes kb-00002 so we can navigate within the drawer
    vi.mocked(getEntries).mockImplementation(async (ids: string[]) => {
      const id = ids[0]
      if (id === 'kb-00001') {
        return makeGetResponse('kb-00001', {
          short_title: 'First Entry',
          superseded_by: 'kb-00002',
        })
      }
      return makeGetResponse(id, { short_title: `Entry ${id}` })
    })
    vi.mocked(getGraphNeighbors).mockResolvedValue({ neighbors: [] })

    renderWithProvider(<OpenButton entryId="kb-00001" />)
    const user = userEvent.setup()

    // Open first entry
    await user.click(screen.getByRole('button', { name: 'Open kb-00001' }))
    await waitFor(() =>
      expect(screen.getAllByText('First Entry')[0]).toBeInTheDocument(),
    )
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledWith('kb-00001', 'both', 50)

    // Navigate to second entry via superseded_by link inside the drawer
    await user.click(screen.getByRole('button', { name: 'kb-00002' }))
    await waitFor(() =>
      expect(screen.getAllByText('Entry kb-00002')[0]).toBeInTheDocument(),
    )
    // Neighbors re-fetched for the new entry
    expect(vi.mocked(getGraphNeighbors)).toHaveBeenCalledWith('kb-00002', 'both', 50)
  })
})
