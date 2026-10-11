import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GraphChatPanel } from '../components/GraphChatPanel'
import { CitedMarkdown } from '../components/CitedMarkdown'
import { exploreSeedAnswer } from '../graph/queryModel'
import type { ChatSeed } from '../graph/chatModel'

/* eslint-disable @typescript-eslint/no-explicit-any */
const h = vi.hoisted(() => ({ streamSSE: vi.fn<(...a: any[]) => Promise<void>>() }))
vi.mock('../streaming', () => ({ streamSSE: h.streamSSE }))
vi.mock('../api', () => ({ getToken: () => 'test-token' }))

const seedS: ChatSeed = {
  question: 'Why alpha',
  answer: 'Alpha is decided [kb-00001].',
  entryIds: ['kb-00001'],
  mode: 'summarize',
}
const seedE: ChatSeed = {
  question: 'Find alpha',
  answer: exploreSeedAnswer(['kb-00001', 'kb-00002']),
  entryIds: ['kb-00001', 'kb-00002'],
  mode: 'explore',
}

type Ev = [string, Record<string, unknown>]
const CHAT: Ev[] = [
  ['chat_session', { session_id: 's1' }],
  ['chat_thinking', {}],
  ['chat_done', { new_entries: ['kb-00002'] }],
  ['chat_response', { answer: 'Beta too [kb-00002].', session_id: 's1' }],
  ['stream_end', {}],
]
const script = (events: Ev[]) =>
  h.streamSSE.mockImplementation((_u, _b, onEvent) => {
    for (const [e, d] of events) onEvent(e, d)
    return Promise.resolve()
  })

let onNewEntries: ReturnType<typeof vi.fn>
let onClose: ReturnType<typeof vi.fn>
let onCitation: ReturnType<typeof vi.fn>
const input = () => screen.getByRole('textbox', { name: 'Follow-up message' })
const view = (seed: ChatSeed = seedS) =>
  render(
    <GraphChatPanel
      seed={seed}
      onCitation={onCitation as any}
      onNewEntries={onNewEntries as any}
      onClose={onClose as any}
    />,
  )
async function say(text: string) {
  await userEvent.type(input(), `${text}{Enter}`)
}

beforeEach(() => {
  h.streamSSE.mockReset()
  h.streamSSE.mockImplementation(() => new Promise<void>(() => {}))
  onNewEntries = vi.fn(() => [])
  onClose = vi.fn()
  onCitation = vi.fn()
  window.HTMLElement.prototype.scrollIntoView = vi.fn()
})
afterEach(() => vi.restoreAllMocks())

describe('GraphChatPanel', () => {
  it('p1 renders the seed', () => {
    view()
    expect(screen.getByRole('complementary', { name: 'Graph chat' })).toBeInTheDocument()
    expect(screen.getByText('summarize')).toBeInTheDocument()
    const all = screen.getAllByText('Why alpha')
    expect(all).toHaveLength(2)
    expect(all.some((e) => e.getAttribute('title') === 'Why alpha')).toBe(true)
    expect(screen.getByRole('button', { name: 'Show kb-00001 on the graph' })).toBeInTheDocument()
  })

  it('p2/p3 seeds the first message then reuses the session', async () => {
    script(CHAT)
    view()
    await say('why?')
    expect(h.streamSSE).toHaveBeenCalledWith(
      '/api/chat/stream?token=test-token',
      {
        message: 'why?',
        mode: 'summarize',
        seed_question: 'Why alpha',
        seed_answer: 'Alpha is decided [kb-00001].',
        seed_entry_ids: ['kb-00001'],
      },
      expect.any(Function),
    )
    expect(input()).toHaveValue('')
    expect(screen.getByText('why?')).toBeInTheDocument()
    await say('more')
    expect(h.streamSSE.mock.calls[1][1]).toEqual({ message: 'more', mode: 'summarize', session_id: 's1' })
  })

  it('p4 thinking, answer and new entries', async () => {
    let emit!: (e: string, d: Record<string, unknown>) => void
    let done!: () => void
    h.streamSSE.mockImplementation((_u, _b, onEvent) => {
      emit = onEvent
      return new Promise<void>((r) => (done = r))
    })
    view()
    await say('go')
    act(() => emit('chat_thinking', {}))
    expect(screen.getByText('Thinking…')).toBeInTheDocument()
    act(() => emit('chat_done', { new_entries: ['kb-00002'] }))
    act(() => emit('chat_response', { answer: 'Beta too [kb-00002].', session_id: 's1' }))
    expect(screen.queryByText('Thinking…')).toBeNull()
    expect(screen.getByText(/Beta too/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Show kb-00002 on the graph' })).toBeInTheDocument()
    expect(onNewEntries).toHaveBeenCalledTimes(1)
    expect(onNewEntries).toHaveBeenCalledWith(['kb-00002'])
    await act(async () => done())
    h.streamSSE.mockImplementation((_u, _b, onEvent) => {
      onEvent('chat_done', { new_entries: [] })
      onEvent('chat_response', { answer: 'x', session_id: 's1' })
      return Promise.resolve()
    })
    await say('again')
    expect(onNewEntries).toHaveBeenCalledTimes(1)
  })

  it('p5 shows errors and recovers', async () => {
    script([['error', { message: 'ValueError: boom' }]])
    view()
    await say('a')
    expect(screen.getByRole('alert')).toHaveTextContent('ValueError: boom')
    h.streamSSE.mockRejectedValue(new Error('500: Internal'))
    await say('b')
    expect(screen.getAllByRole('alert').at(-1)).toHaveTextContent('500: Internal')
    script(CHAT)
    await say('c')
    expect(screen.getByText(/Beta too/)).toBeInTheDocument()
  })

  it('p6 input rules', async () => {
    view()
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    await userEvent.type(input(), 'x{Shift>}{Enter}{/Shift}')
    expect(h.streamSSE).not.toHaveBeenCalled()
    await userEvent.clear(input())
    await userEvent.type(input(), '   {Enter}')
    expect(h.streamSSE).not.toHaveBeenCalled()
    await userEvent.clear(input())
    await say('first')
    expect(h.streamSSE).toHaveBeenCalledTimes(1)
    await userEvent.type(input(), 'again{Enter}')
    expect(h.streamSSE).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
  })

  it('p7 Escape blurs the input without closing', async () => {
    view()
    input().focus()
    expect(document.activeElement).toBe(input())
    fireEvent.keyDown(input(), { key: 'Escape' })
    expect(document.activeElement).not.toBe(input())
    expect(onClose).not.toHaveBeenCalled()
  })

  it('p8 close button', async () => {
    view()
    await userEvent.click(screen.getByRole('button', { name: 'Close chat' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('p9 Summarize these', async () => {
    script(CHAT)
    view(seedE)
    await userEvent.click(screen.getByRole('button', { name: 'Summarize these' }))
    expect(h.streamSSE.mock.calls[0][1]).toEqual({
      message: 'Summarize these entries for me.',
      mode: 'explore',
      seed_question: 'Find alpha',
      seed_answer: 'Found 2 entries: [kb-00001], [kb-00002]. Ask me anything about these results.',
      seed_entry_ids: ['kb-00001', 'kb-00002'],
    })
    expect(screen.queryByRole('button', { name: 'Summarize these' })).toBeNull()
  })

  it('p9b no Summarize these in summarize mode', () => {
    view()
    expect(screen.queryByRole('button', { name: 'Summarize these' })).toBeNull()
  })

  it('p10 citations and external links', async () => {
    view()
    await userEvent.click(screen.getByRole('button', { name: 'Show kb-00001 on the graph' }))
    expect(onCitation).toHaveBeenCalledWith('kb-00001')
    render(<CitedMarkdown onCitation={vi.fn()}>{'[docs](https://example.com) `[kb-00009]`'}</CitedMarkdown>)
    const a = screen.getByRole('link', { name: 'docs' })
    expect(a).toHaveAttribute('href', 'https://example.com')
    expect(a).toHaveAttribute('target', '_blank')
    expect(a).toHaveAttribute('rel', 'noopener noreferrer')
    expect(screen.queryByRole('button', { name: 'Show kb-00009 on the graph' })).toBeNull()
  })

  it('p11 drops events after unmount', async () => {
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {})
    let emit!: (e: string, d: Record<string, unknown>) => void
    h.streamSSE.mockImplementation((_u, _b, onEvent) => {
      emit = onEvent
      return new Promise<void>(() => {})
    })
    const { unmount } = view()
    await say('x')
    unmount()
    emit('chat_done', { new_entries: ['kb-00002'] })
    emit('chat_response', { answer: 'late', session_id: 's1' })
    expect(onNewEntries).not.toHaveBeenCalled()
    expect(debug).toHaveBeenCalledWith('[graph-chat] dropped event after unmount', { event: 'chat_done' })
  })

  it('p12 stream without a response', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    script([
      ['chat_session', { session_id: 's1' }],
      ['chat_thinking', {}],
    ])
    view()
    await say('x')
    expect(screen.getByRole('alert')).toHaveTextContent('ended without a response')
    expect(warn).toHaveBeenCalledTimes(1)
    expect(warn).toHaveBeenCalledWith('[graph-chat] stream ended without response', {
      session_id: 's1',
      events_seen: ['chat_session', 'chat_thinking'],
    })
    script(CHAT)
    await say('y')
    expect(screen.getByText(/Beta too/)).toBeInTheDocument()
  })

  it('p13 adopts the server session id', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    script([['chat_response', { answer: 'ok', session_id: 's1' }]])
    view()
    await say('x')
    expect(warn).toHaveBeenCalledWith('[graph-chat] session id mismatch', {
      state_session_id: null,
      response_session_id: 's1',
    })
    await say('y')
    expect(h.streamSSE.mock.calls[1][1]).toEqual({ message: 'y', mode: 'summarize', session_id: 's1' })
    expect(warn.mock.calls.some((c) => String(c[0]).includes('re-seeding'))).toBe(false)
  })

  it('p14 notice for hidden entries', async () => {
    onNewEntries = vi.fn(() => ['kb-77777'])
    script([
      ['chat_done', { new_entries: ['kb-77777'] }],
      ['chat_response', { answer: 'ok', session_id: 's1' }],
    ])
    view()
    await say('x')
    expect(screen.getByText('1 new entry is not on the graph')).toBeInTheDocument()
  })

  it('p14b plural notice', async () => {
    onNewEntries = vi.fn(() => ['kb-77777', 'kb-77778'])
    script([
      ['chat_done', { new_entries: ['kb-77777', 'kb-77778'] }],
      ['chat_response', { answer: 'ok', session_id: 's1' }],
    ])
    view()
    await say('x')
    expect(screen.getByText('2 new entries are not on the graph')).toBeInTheDocument()
  })

  it('p15 unlinked citation tripwire', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    view({ ...seedS, answer: 'See kb-00001 and [kb-00002].' })
    expect(info).toHaveBeenCalledWith('[graph-chat] unlinked citations', {
      message_index: 1,
      linked: 1,
      mentioned: 2,
    })
  })

  it('p15b no tripwire when linked', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {})
    view()
    expect(info).not.toHaveBeenCalled()
  })
})
