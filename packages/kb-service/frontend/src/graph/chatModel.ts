import type { QuerySeed, QuerySeedInput } from './queryModel'
import { querySeed } from './queryModel'

export const GRAPH_CHAT_W = 420
export const CITATION_RE = /\[(kb-\d{5}(?:\s*,\s*kb-\d{5})*)\]/g
export const CITE_HREF_PREFIX = '#cite-'
export const SUMMARIZE_THESE_MESSAGE = 'Summarize these entries for me.'
export const STREAM_NO_RESPONSE_MESSAGE = 'The chat stream ended without a response.'

export type ChatSeed = QuerySeed
export type ChatMode = QuerySeed['mode']
export type SeedSkipReason = 'empty_answer' | 'no_entries'

export interface ChatSeedDecision {
  seed: ChatSeed | null
  reason: SeedSkipReason | null
  mode: ChatMode
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'notice'
  content: string
  error?: true
}

export interface GraphChatState {
  sessionId: string | null
  messages: ChatMessage[]
  busy: boolean
  thinking: boolean
  followUps: number
}

export type ChatAction =
  | { type: 'send'; message: string }
  | { type: 'session'; sessionId: string }
  | { type: 'thinking' }
  | { type: 'response'; answer: string }
  | { type: 'error'; message: string }
  | { type: 'notice'; content: string }
  | { type: 'end' }

export type ChatStreamBody = {
  message: string
  mode: ChatMode
  session_id?: string
  seed_question?: string
  seed_answer?: string
  seed_entry_ids?: string[]
}

export type CitationPart = { kind: 'text'; text: string } | { kind: 'cite'; id: string }

export function splitCitationText(text: string): CitationPart[] {
  const parts: CitationPart[] = []
  const pushText = (t: string) => {
    if (t === '') return
    const last = parts[parts.length - 1]
    if (last && last.kind === 'text') last.text += t
    else parts.push({ kind: 'text', text: t })
  }
  let pos = 0
  for (const m of text.matchAll(CITATION_RE)) {
    const start = m.index
    pushText(text.slice(pos, start))
    pushText('[')
    m[1].split(',').forEach((raw, i) => {
      if (i > 0) pushText(', ')
      parts.push({ kind: 'cite', id: raw.trim() })
    })
    pushText(']')
    pos = start + m[0].length
  }
  pushText(text.slice(pos))
  return parts
}

interface MdNode {
  type: string
  value?: string
  url?: string
  children?: MdNode[]
}

function transformNode(node: MdNode): void {
  if (!node.children) return
  const out: MdNode[] = []
  for (const child of node.children) {
    if (child.type === 'text' && typeof child.value === 'string') {
      for (const part of splitCitationText(child.value)) {
        if (part.kind === 'text') out.push({ type: 'text', value: part.text })
        else
          out.push({
            type: 'link',
            url: CITE_HREF_PREFIX + part.id,
            children: [{ type: 'text', value: part.id }],
          })
      }
    } else {
      if (child.type !== 'link' && child.type !== 'linkReference') transformNode(child)
      out.push(child)
    }
  }
  node.children = out
}

export function remarkKbCitations() {
  return (tree: MdNode): void => {
    transformNode(tree)
  }
}

export function chatSeedDecision(q: QuerySeedInput): ChatSeedDecision | null {
  if (q.answer === null && q.entries === null) return null
  const seed = querySeed(q)
  if (seed) return { seed, reason: null, mode: seed.mode }
  if (q.answer !== null) return { seed: null, reason: 'empty_answer', mode: 'summarize' }
  return { seed: null, reason: 'no_entries', mode: 'explore' }
}

export function initialChatState(seed: ChatSeed): GraphChatState {
  const messages: ChatMessage[] = []
  if (seed.question.trim() !== '') messages.push({ role: 'user', content: seed.question })
  messages.push({ role: 'assistant', content: seed.answer })
  return { sessionId: null, messages, busy: false, thinking: false, followUps: 0 }
}

export function buildChatStreamBody(
  seed: ChatSeed,
  sessionId: string | null,
  message: string,
): ChatStreamBody {
  if (sessionId === null) {
    return {
      message,
      mode: seed.mode,
      seed_question: seed.question,
      seed_answer: seed.answer,
      seed_entry_ids: [...seed.entryIds],
    }
  }
  return { message, mode: seed.mode, session_id: sessionId }
}

export function chatReducer(state: GraphChatState, action: ChatAction): GraphChatState {
  switch (action.type) {
    case 'send':
      return {
        ...state,
        messages: [...state.messages, { role: 'user', content: action.message }],
        busy: true,
        thinking: false,
        followUps: state.followUps + 1,
      }
    case 'session':
      return { ...state, sessionId: action.sessionId }
    case 'thinking':
      return { ...state, thinking: true }
    case 'response':
      return {
        ...state,
        messages: [...state.messages, { role: 'assistant', content: action.answer }],
        thinking: false,
      }
    case 'error':
      return {
        ...state,
        messages: [
          ...state.messages,
          { role: 'assistant', content: action.message, error: true },
        ],
        thinking: false,
      }
    case 'notice':
      return {
        ...state,
        messages: [...state.messages, { role: 'notice', content: action.content }],
      }
    case 'end':
      return { ...state, busy: false, thinking: false }
  }
}

export function citationCoverage(text: string): { linked: number; mentioned: number } {
  const linked = splitCitationText(text).filter((p) => p.kind === 'cite').length
  const mentioned = [...text.matchAll(/kb-\d{5}/gi)].length
  return { linked, mentioned }
}
