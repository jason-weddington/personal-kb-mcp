/**
 * AskPanel behaviour tests (formerly Ask page tests — extracted to AskPanel).
 * All original Ask behaviours are covered here against the extracted component.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { AskPanel } from '../components/AskPanel'

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

// Mock streaming
vi.mock('../streaming', () => ({
  streamSSE: vi.fn(),
}))

// Mock api
vi.mock('../api', () => ({
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

import { streamSSE } from '../streaming'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'

type OnEvent = (event: string, data: Record<string, unknown>) => void

function renderAskPanel() {
  return render(
    <MemoryRouter>
      <AskPanel />
    </MemoryRouter>,
  )
}

describe('AskPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useEntryDrawer).mockReturnValue({
      openEntry: openEntryMock,
      closeEntry: vi.fn(),
    })
  })

  it('Ask button is disabled when question is empty', () => {
    renderAskPanel()
    expect(screen.getByRole('button', { name: /^ask$/i })).toBeDisabled()
  })

  it('shows mode Chip, live status line, markdown answer, and entry chip calls openEntry', async () => {
    const user = userEvent.setup()

    // Controlled mock: capture onEvent, resolve on demand
    let capturedOnEvent: OnEvent | null = null
    let resolveStream!: () => void

    vi.mocked(streamSSE).mockImplementation(
      async (_url: string, _body: unknown, onEvent: OnEvent) => {
        capturedOnEvent = onEvent
        return new Promise<void>((resolve) => {
          resolveStream = resolve
        })
      },
    )

    renderAskPanel()

    await user.type(screen.getByLabelText(/question/i), 'What is a decision?')
    await user.click(screen.getByRole('button', { name: /^ask$/i }))

    // Wait for streamSSE to be called (mock captures onEvent)
    await waitFor(() => expect(capturedOnEvent).not.toBeNull())

    // Fire classified and status events
    act(() => {
      capturedOnEvent!('classified', { mode: 'summarize' })
      capturedOnEvent!('status', { message: 'Searching knowledge base...' })
    })

    // Mode Chip should be visible
    await waitFor(() =>
      expect(screen.getByText('summarize')).toBeInTheDocument(),
    )

    // Status line should be visible during streaming
    await waitFor(() =>
      expect(screen.getByText('Searching knowledge base...')).toBeInTheDocument(),
    )

    // Fire synthesis_result and stream_end, then resolve
    act(() => {
      capturedOnEvent!('synthesis_result', {
        answer: 'A decision is a KB entry.',
        question: 'What is a decision?',
        entry_ids: ['kb-00001'],
      })
      capturedOnEvent!('stream_end', {})
      resolveStream()
    })

    // Markdown answer should render
    await waitFor(() =>
      expect(screen.getByText('A decision is a KB entry.')).toBeInTheDocument(),
    )

    // Entry chip is now a button (no longer a link); clicking it calls openEntry
    await waitFor(() => {
      const chip = screen.getByRole('button', { name: 'kb-00001' })
      expect(chip).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: 'kb-00001' }))
    expect(openEntryMock).toHaveBeenCalledWith('kb-00001')

    // Answer is wrapped in a ResponseCard
    await waitFor(() => {
      const answerEl = screen.getByText('A decision is a KB entry.')
      expect(answerEl.closest('[data-testid="response-card"]')).not.toBeNull()
    })

    // Form is re-enabled after stream
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /^ask$/i })).not.toBeDisabled(),
    )
  })

  it('explore mode: renders entry card inside a response-card, title click calls openEntry', async () => {
    const user = userEvent.setup()

    let capturedOnEvent: OnEvent | null = null
    let resolveStream!: () => void

    vi.mocked(streamSSE).mockImplementation(
      async (_url: string, _body: unknown, onEvent: OnEvent) => {
        capturedOnEvent = onEvent
        return new Promise<void>((resolve) => {
          resolveStream = resolve
        })
      },
    )

    renderAskPanel()

    await user.type(screen.getByLabelText(/question/i), 'Show me decisions')
    await user.click(screen.getByRole('button', { name: /^ask$/i }))

    await waitFor(() => expect(capturedOnEvent).not.toBeNull())

    act(() => {
      capturedOnEvent!('classified', { mode: 'explore' })
      capturedOnEvent!('entries', {
        entries: [
          {
            id: 'kb-00002',
            short_title: 'Some entry',
            entry_type: 'decision',
            tags: [],
            context: 'Some context.',
          },
        ],
        turns_used: 1,
      })
      capturedOnEvent!('stream_end', {})
      resolveStream()
    })

    // Entry title renders as a button (MUI Link component="button")
    await waitFor(() => {
      const button = screen.getByRole('button', { name: 'Some entry' })
      expect(button).toBeInTheDocument()
    })

    // Clicking it calls openEntry with the entry id
    await user.click(screen.getByRole('button', { name: 'Some entry' }))
    expect(openEntryMock).toHaveBeenCalledWith('kb-00002')

    // Entry card is wrapped in a ResponseCard
    await waitFor(() => {
      const button = screen.getByRole('button', { name: 'Some entry' })
      expect(button.closest('[data-testid="response-card"]')).not.toBeNull()
    })
  })

  it('shows error Alert when streamSSE rejects', async () => {
    const user = userEvent.setup()

    vi.mocked(streamSSE).mockRejectedValue(new Error('Network failure'))

    renderAskPanel()

    await user.type(screen.getByLabelText(/question/i), 'test question')
    await user.click(screen.getByRole('button', { name: /^ask$/i }))

    await waitFor(() =>
      expect(screen.getByText('Network failure')).toBeInTheDocument(),
    )

    // Form should be re-enabled in finally
    expect(screen.getByRole('button', { name: /^ask$/i })).not.toBeDisabled()
  })

  describe('keyboard submit (Cmd/Ctrl+Enter)', () => {
    beforeEach(() => {
      vi.mocked(streamSSE).mockResolvedValue(undefined)
    })

    it('Ctrl+Enter fires submit when question is non-empty', async () => {
      const user = userEvent.setup()
      renderAskPanel()

      const input = screen.getByLabelText(/question/i)
      await user.type(input, 'my question')
      await user.keyboard('{Control>}{Enter}{/Control}')

      await waitFor(() => expect(vi.mocked(streamSSE)).toHaveBeenCalledTimes(1))
    })

    it('metaKey+Enter fires submit when question is non-empty', async () => {
      const user = userEvent.setup()
      renderAskPanel()

      const input = screen.getByLabelText(/question/i)
      await user.type(input, 'my question')
      await user.keyboard('{Meta>}{Enter}{/Meta}')

      await waitFor(() => expect(vi.mocked(streamSSE)).toHaveBeenCalledTimes(1))
    })

    it('plain Enter does not fire submit', async () => {
      const user = userEvent.setup()
      renderAskPanel()

      const input = screen.getByLabelText(/question/i)
      await user.type(input, 'my question')
      await user.keyboard('{Enter}')

      // Small pause to ensure no async submission triggered
      await new Promise((r) => setTimeout(r, 30))
      expect(vi.mocked(streamSSE)).not.toHaveBeenCalled()
    })

    it('Ctrl+Enter does not fire when question is blank', async () => {
      const user = userEvent.setup()
      renderAskPanel()

      const input = screen.getByLabelText(/question/i)
      await user.click(input)
      await user.keyboard('{Control>}{Enter}{/Control}')

      await new Promise((r) => setTimeout(r, 30))
      expect(vi.mocked(streamSSE)).not.toHaveBeenCalled()
    })

    it('Ctrl+Enter does not fire while a request is in flight', async () => {
      const user = userEvent.setup()

      // Never-resolving stream to simulate in-flight request
      vi.mocked(streamSSE).mockReturnValue(new Promise(() => {}))

      renderAskPanel()

      const input = screen.getByLabelText(/question/i)
      await user.type(input, 'my question')

      // Submit via button to start the in-flight request
      await user.click(screen.getByRole('button', { name: /^ask$/i }))

      // Wait until the button is in submitting state (disabled)
      await waitFor(() =>
        expect(screen.getByRole('button', { name: /thinking/i })).toBeDisabled(),
      )

      // Clear call count so we only count keyboard attempt
      vi.mocked(streamSSE).mockClear()

      // Try keyboard shortcut while in-flight
      await user.keyboard('{Control>}{Enter}{/Control}')

      await new Promise((r) => setTimeout(r, 30))
      expect(vi.mocked(streamSSE)).not.toHaveBeenCalled()
    })
  })
})
