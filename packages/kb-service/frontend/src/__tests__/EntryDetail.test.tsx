import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { EntryDetail } from '../pages/EntryDetail'

// Mock api
vi.mock('../api', () => ({
  getEntries: vi.fn(),
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

// Mock EntryDrawerContext for isolation
const openEntryMock = vi.fn()
vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: vi.fn(() => ({
    openEntry: openEntryMock,
    closeEntry: vi.fn(),
  })),
  EntryDrawerProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

import { getEntries } from '../api'

function makeEntry(id: string, opts: { superseded_by?: string } = {}) {
  return {
    id,
    short_title: `Entry ${id}`,
    long_title: `Entry ${id}`,
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

describe('EntryDetail superseded_by → openEntry', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getEntries).mockResolvedValue({
      results: [
        {
          id: 'kb-00001',
          found: true,
          entry: makeEntry('kb-00001', { superseded_by: 'kb-00002' }),
          pointer_rot: [],
        },
      ],
    })
  })

  // (g) clicking superseded_by reference calls openEntry with target id
  it('(g) clicking superseded_by reference calls openEntry with the target id', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/entries/kb-00001']}>
        <Routes>
          <Route path="/entries/:id" element={<EntryDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    // Wait for entry to load and render
    await waitFor(() =>
      expect(screen.getByText('Entry kb-00001')).toBeInTheDocument(),
    )

    // The superseded_by renders as a button with text 'kb-00002'
    const supersededBtn = screen.getByRole('button', { name: 'kb-00002' })
    await user.click(supersededBtn)

    expect(openEntryMock).toHaveBeenCalledWith('kb-00002')
  })
})
