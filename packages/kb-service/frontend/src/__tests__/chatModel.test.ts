import { describe, it, expect } from 'vitest'
import {
  buildChatStreamBody,
  chatReducer,
  chatSeedDecision,
  citationCoverage,
  initialChatState,
  remarkKbCitations,
  splitCitationText,
} from '../graph/chatModel'
import type { ChatAction, ChatSeed, GraphChatState } from '../graph/chatModel'
import { exploreSeedAnswer } from '../graph/queryModel'
import type { AskEntry } from '../kbTypes'

const t = (text: string) => ({ kind: 'text', text })
const c = (id: string) => ({ kind: 'cite', id })
const entry = (id: string): AskEntry => ({
  id,
  short_title: id,
  entry_type: 'decision',
  tags: [],
  context: '',
})

describe('splitCitationText', () => {
  it('splits single and list citations', () => {
    expect(splitCitationText('See [kb-00042].')).toEqual([t('See ['), c('kb-00042'), t('].')])
    const list = [t('['), c('kb-00001'), t(', '), c('kb-00002'), t(']')]
    expect(splitCitationText('[kb-00001, kb-00002]')).toEqual(list)
    expect(splitCitationText('[kb-00001,kb-00002]')).toEqual(list)
    expect(splitCitationText('a [kb-00001] b [kb-00002]')).toEqual([
      t('a ['),
      c('kb-00001'),
      t('] b ['),
      c('kb-00002'),
      t(']'),
    ])
  })
  it('leaves non-citations alone', () => {
    const s = '[kb-0001] [kb-000011] kb-00042 [KB-00042]'
    expect(splitCitationText(s)).toEqual([t(s)])
    expect(splitCitationText('')).toEqual([])
  })
  it('is repeatable', () => {
    expect(splitCitationText('x [kb-00001]')).toEqual(splitCitationText('x [kb-00001]'))
  })
})

describe('remarkKbCitations', () => {
  it('replaces text citations outside links and code', () => {
    const tree = {
      type: 'root',
      children: [
        {
          type: 'paragraph',
          children: [
            { type: 'text', value: 'x [kb-00001] y' },
            { type: 'inlineCode', value: '[kb-00002]' },
            { type: 'link', url: 'u', children: [{ type: 'text', value: '[kb-00003]' }] },
          ],
        },
      ],
    }
    remarkKbCitations()(tree)
    expect(tree.children[0].children).toEqual([
      { type: 'text', value: 'x [' },
      { type: 'link', url: '#cite-kb-00001', children: [{ type: 'text', value: 'kb-00001' }] },
      { type: 'text', value: '] y' },
      { type: 'inlineCode', value: '[kb-00002]' },
      { type: 'link', url: 'u', children: [{ type: 'text', value: '[kb-00003]' }] },
    ])
  })
})

describe('chatSeedDecision', () => {
  it('decides per result shape', () => {
    expect(
      chatSeedDecision({ question: 'Q', answer: 'A', sourceIds: ['kb-00001', 'kb-00003'], entries: null }),
    ).toEqual({
      seed: { question: 'Q', mode: 'summarize', answer: 'A', entryIds: ['kb-00001', 'kb-00003'] },
      reason: null,
      mode: 'summarize',
    })
    expect(chatSeedDecision({ question: 'Q', answer: '', sourceIds: [], entries: null })).toEqual({
      seed: null,
      reason: 'empty_answer',
      mode: 'summarize',
    })
    expect(chatSeedDecision({ question: 'Q', answer: null, sourceIds: [], entries: [] })).toEqual({
      seed: null,
      reason: 'no_entries',
      mode: 'explore',
    })
    expect(
      chatSeedDecision({
        question: 'Q',
        answer: null,
        sourceIds: [],
        entries: [entry('kb-00001'), entry('kb-00002')],
      }),
    ).toEqual({
      seed: {
        question: 'Q',
        mode: 'explore',
        answer: exploreSeedAnswer(['kb-00001', 'kb-00002']),
        entryIds: ['kb-00001', 'kb-00002'],
      },
      reason: null,
      mode: 'explore',
    })
    expect(chatSeedDecision({ question: 'Q', answer: null, sourceIds: [], entries: null })).toBeNull()
  })
})

describe('initialChatState', () => {
  it('seeds the transcript', () => {
    const seed: ChatSeed = { question: 'Q', answer: 'A', entryIds: [], mode: 'summarize' }
    const s = initialChatState(seed)
    expect(s.messages).toEqual([
      { role: 'user', content: 'Q' },
      { role: 'assistant', content: 'A' },
    ])
    expect(s.followUps).toBe(0)
    expect(initialChatState({ ...seed, question: '' }).messages).toEqual([
      { role: 'assistant', content: 'A' },
    ])
  })
})

describe('buildChatStreamBody', () => {
  const seedQ: ChatSeed = { question: 'Q', answer: 'A', entryIds: ['kb-00001'], mode: 'summarize' }
  it('seeds only the first message', () => {
    const first = buildChatStreamBody(seedQ, null, 'why?')
    expect(first).toEqual({
      message: 'why?',
      mode: 'summarize',
      seed_question: 'Q',
      seed_answer: 'A',
      seed_entry_ids: ['kb-00001'],
    })
    expect('session_id' in first).toBe(false)
    const next = buildChatStreamBody(seedQ, 's1', 'more')
    expect(next).toEqual({ message: 'more', mode: 'summarize', session_id: 's1' })
    expect('seed_question' in next).toBe(false)
  })
})

describe('chatReducer', () => {
  const base: GraphChatState = {
    sessionId: null,
    messages: [{ role: 'assistant', content: 'A' }],
    busy: false,
    thinking: true,
    followUps: 0,
  }
  it('applies every action', () => {
    let s = chatReducer(base, { type: 'send', message: 'hi' })
    expect(s).toMatchObject({ busy: true, thinking: false, followUps: 1 })
    expect(s.messages[1]).toEqual({ role: 'user', content: 'hi' })
    s = chatReducer(s, { type: 'session', sessionId: 's1' })
    expect(s.sessionId).toBe('s1')
    s = chatReducer(s, { type: 'thinking' })
    expect(s.thinking).toBe(true)
    s = chatReducer(s, { type: 'response', answer: 'ok' })
    expect(s.thinking).toBe(false)
    expect(s.messages.at(-1)).toEqual({ role: 'assistant', content: 'ok' })
    s = chatReducer(s, { type: 'error', message: 'bad' })
    expect(s.messages.at(-1)).toEqual({ role: 'assistant', content: 'bad', error: true })
    s = chatReducer(s, { type: 'notice', content: 'n' })
    expect(s.messages.at(-1)).toEqual({ role: 'notice', content: 'n' })
    s = chatReducer(s, { type: 'end' })
    expect(s).toMatchObject({ busy: false, thinking: false })
  })
  it('never mutates a frozen state', () => {
    const frozen = Object.freeze({
      ...base,
      messages: Object.freeze(base.messages.map((m) => Object.freeze({ ...m }))) as never,
    })
    const actions: ChatAction[] = [
      { type: 'send', message: 'x' },
      { type: 'session', sessionId: 's' },
      { type: 'thinking' },
      { type: 'response', answer: 'a' },
      { type: 'error', message: 'e' },
      { type: 'notice', content: 'n' },
      { type: 'end' },
    ]
    for (const a of actions) expect(chatReducer(frozen, a)).not.toBe(frozen)
  })
})

describe('citationCoverage', () => {
  it('counts linked and mentioned ids', () => {
    expect(citationCoverage('a [kb-00001] b kb-00002 (KB-00003)')).toEqual({ linked: 1, mentioned: 3 })
    expect(citationCoverage('none')).toEqual({ linked: 0, mentioned: 0 })
  })
})
