import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Chat } from '../pages/Chat'

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

// Mock streaming
vi.mock('../streaming', () => ({
  streamSSE: vi.fn(),
}))

// Mock api
vi.mock('../api', () => ({
  listChats: vi.fn(),
  getChatMessages: vi.fn(),
  deleteChat: vi.fn(),
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

import { listChats, getChatMessages, deleteChat } from '../api'

const SESSIONS = [
  { id: 'chat-1', title: 'First conversation', mode: 'chat', updated_at: '2026-06-09T12:00:00Z' },
  { id: 'chat-2', title: 'Second conversation', mode: 'chat', updated_at: '2026-06-08T10:30:00Z' },
]

function renderChat() {
  return render(
    <MemoryRouter>
      <Chat />
    </MemoryRouter>,
  )
}

describe('Chat page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(listChats).mockResolvedValue(SESSIONS)
  })

  it('renders chat session list', async () => {
    renderChat()

    await waitFor(() =>
      expect(screen.getByText('First conversation')).toBeInTheDocument(),
    )
    expect(screen.getByText('Second conversation')).toBeInTheDocument()
  })

  it('selecting a session loads its messages', async () => {
    const user = userEvent.setup()
    vi.mocked(getChatMessages).mockResolvedValue([
      { role: 'user', content: 'Hello there' },
      { role: 'assistant', content: 'Hi! How can I help?' },
    ])

    renderChat()

    await waitFor(() =>
      expect(screen.getByText('First conversation')).toBeInTheDocument(),
    )

    await user.click(screen.getByText('First conversation'))

    await waitFor(() =>
      expect(screen.getByText('Hello there')).toBeInTheDocument(),
    )
    expect(screen.getByText('Hi! How can I help?')).toBeInTheDocument()
    expect(vi.mocked(getChatMessages)).toHaveBeenCalledWith('chat-1')
  })

  it('delete button calls DELETE /api/chat/{id} after confirm and refreshes list', async () => {
    const user = userEvent.setup()
    vi.mocked(deleteChat).mockResolvedValue({ ok: true, id: null })
    vi.mocked(listChats)
      .mockResolvedValueOnce(SESSIONS)
      .mockResolvedValue([SESSIONS[1]])

    // Mock window.confirm
    vi.spyOn(window, 'confirm').mockReturnValue(true)

    renderChat()

    await waitFor(() =>
      expect(screen.getByText('First conversation')).toBeInTheDocument(),
    )

    // Click the delete button for the first session
    const deleteButtons = screen.getAllByLabelText(/delete/i)
    await user.click(deleteButtons[0])

    await waitFor(() =>
      expect(vi.mocked(deleteChat)).toHaveBeenCalledWith('chat-1'),
    )

    // History is refreshed
    await waitFor(() =>
      expect(vi.mocked(listChats)).toHaveBeenCalledTimes(2),
    )
  })

  it('delete cancelled by user does not call deleteChat', async () => {
    const user = userEvent.setup()
    vi.spyOn(window, 'confirm').mockReturnValue(false)

    renderChat()

    await waitFor(() =>
      expect(screen.getByText('First conversation')).toBeInTheDocument(),
    )

    const deleteButtons = screen.getAllByLabelText(/delete/i)
    await user.click(deleteButtons[0])

    expect(vi.mocked(deleteChat)).not.toHaveBeenCalled()
  })
})
