import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Search } from '../pages/Search'
import type { SearchResponse } from '../kbTypes'

// Mock AuthContext
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

// Mock EntryDrawerContext
const openEntryMock = vi.fn()
vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: vi.fn(),
  EntryDrawerProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

// Mock api wrappers
vi.mock('../api', () => ({
  searchKb: vi.fn(),
  listKbProjects: vi.fn(),
  getToken: vi.fn(() => 'test-token'),
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

import { searchKb, listKbProjects } from '../api'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'

const PROJECTS_RESPONSE = { items: [{ name: 'proj-a', entry_count: 5 }] }

const SEARCH_RESPONSE: SearchResponse = {
  results: [
    {
      entry: {
        id: 'kb-00001',
        short_title: 'My Entry Title',
        long_title: 'Long title',
        entry_type: 'decision',
        tags: ['tag1', 'tag2'],
        project_ref: 'proj-a',
        contributor: null,
        team: null,
        knowledge_details: 'details',
        confidence_level: 0.9,
        is_active: true,
        has_embedding: true,
        version: 1,
        sensitivity: null,
        hints: {},
        source_context: null,
        updated_by: null,
        created_at: null,
        updated_at: null,
        last_accessed: null,
        expires_at: null,
        superseded_by: null,
      },
      score: 0.95,
      effective_confidence: 0.88,
      staleness_warning: null,
      match_source: 'hybrid' as const,
    },
  ],
  filtered_count: 1,
}

const SEARCH_RESPONSE_WITH_STALENESS: SearchResponse = {
  results: [
    {
      ...SEARCH_RESPONSE.results[0],
      staleness_warning: 'This entry is over 6 months old',
    },
  ],
  filtered_count: 1,
}

function renderSearch() {
  return render(
    <MemoryRouter>
      <Search />
    </MemoryRouter>,
  )
}

describe('Search page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(listKbProjects).mockResolvedValue(PROJECTS_RESPONSE)
    vi.mocked(searchKb).mockResolvedValue(SEARCH_RESPONSE)
    vi.mocked(useEntryDrawer).mockReturnValue({
      openEntry: openEntryMock,
      closeEntry: vi.fn(),
    })
  })

  it('renders the search form and project list loads', async () => {
    renderSearch()
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument()
    await waitFor(() => expect(vi.mocked(listKbProjects)).toHaveBeenCalled())
  })

  it('all-blank filters: POST body is exactly {query, limit, include_stale, include_expired}', async () => {
    const user = userEvent.setup()
    renderSearch()

    await waitFor(() => expect(vi.mocked(listKbProjects)).toHaveBeenCalled())

    await user.type(screen.getByLabelText(/query/i), 'test query')
    await user.click(screen.getByRole('button', { name: /^search$/i }))

    await waitFor(() => expect(vi.mocked(searchKb)).toHaveBeenCalled())

    const body = vi.mocked(searchKb).mock.calls[0][0]
    expect(body).toEqual({
      query: 'test query',
      limit: 10,
      include_stale: false,
      include_expired: false,
    })
    // Must NOT include project_ref, entry_type, or tags
    expect(body).not.toHaveProperty('project_ref')
    expect(body).not.toHaveProperty('entry_type')
    expect(body).not.toHaveProperty('tags')
  })

  it('renders result cards with short_title and filtered_count', async () => {
    const user = userEvent.setup()
    renderSearch()

    await user.type(screen.getByLabelText(/query/i), 'test')
    await user.click(screen.getByRole('button', { name: /^search$/i }))

    await waitFor(() =>
      expect(screen.getByText('My Entry Title')).toBeInTheDocument(),
    )
    expect(screen.getByText('1 result')).toBeInTheDocument()
  })

  it('renders staleness_warning as Alert when non-null', async () => {
    vi.mocked(searchKb).mockResolvedValue(SEARCH_RESPONSE_WITH_STALENESS)
    const user = userEvent.setup()
    renderSearch()

    await user.type(screen.getByLabelText(/query/i), 'test')
    await user.click(screen.getByRole('button', { name: /^search$/i }))

    await waitFor(() =>
      expect(
        screen.getByText('This entry is over 6 months old'),
      ).toBeInTheDocument(),
    )
  })

  it('clicking a result card calls openEntry with the entry id', async () => {
    const user = userEvent.setup()
    renderSearch()

    await user.type(screen.getByLabelText(/query/i), 'test')
    await user.click(screen.getByRole('button', { name: /^search$/i }))

    await waitFor(() =>
      expect(screen.getByText('My Entry Title')).toBeInTheDocument(),
    )

    // Click the result card (CardActionArea renders as a button)
    await user.click(screen.getByText('My Entry Title'))

    expect(openEntryMock).toHaveBeenCalledWith('kb-00001')
  })
})
