import { describe, it, expect } from 'vitest'
import { INSPECTOR_PILL_MAX, entrySummary, groupNeighbors, normalizeDirection } from '../graph/inspectorModel'
import type { ConnInput } from '../graph/inspectorModel'

const conn = (nodeId: string, edgeType: string, dir: 'in' | 'out'): ConnInput => ({
  nodeId,
  edgeType,
  dir,
})

describe('groupNeighbors', () => {
  it('keys each group by `${edgeType}:${dir}`', () => {
    const groups = groupNeighbors([
      conn('tag:alpha', 'has_tag', 'out'),
      conn('kb-00001', 'has_tag', 'in'),
      conn('kb-00002', 'supersedes', 'out'),
    ])
    expect(groups.map((g) => g.key)).toEqual(['has_tag:out', 'has_tag:in', 'supersedes:out'])
    expect(groups[0]).toEqual({
      key: 'has_tag:out',
      edgeType: 'has_tag',
      dir: 'out',
      members: [{ nodeId: 'tag:alpha' }],
    })
    expect(groups[1]).toEqual({
      key: 'has_tag:in',
      edgeType: 'has_tag',
      dir: 'in',
      members: [{ nodeId: 'kb-00001' }],
    })
  })

  it('keeps first-appearance order of the input', () => {
    const groups = groupNeighbors([
      conn('kb-00003', 'related_to', 'in'),
      conn('tag:alpha', 'has_tag', 'out'),
      conn('kb-00002', 'supersedes', 'out'),
      conn('kb-00001', 'related_to', 'in'),
    ])
    expect(groups.map((g) => g.key)).toEqual(['related_to:in', 'has_tag:out', 'supersedes:out'])
    expect(groups[0].members).toEqual([{ nodeId: 'kb-00003' }, { nodeId: 'kb-00001' }])
  })

  it('a duplicate (nodeId, edgeType, dir) tuple does not create a second member', () => {
    const groups = groupNeighbors([
      conn('kb-00002', 'related_to', 'out'),
      conn('tag:x', 'has_tag', 'out'),
      conn('kb-00002', 'related_to', 'out'),
    ])
    expect(groups[0].members).toEqual([{ nodeId: 'kb-00002' }])
  })

  it('the same node may appear in two groups with different edge or direction', () => {
    const groups = groupNeighbors([
      conn('kb-00002', 'related_to', 'out'),
      conn('kb-00002', 'related_to', 'in'),
      conn('kb-00002', 'supersedes', 'out'),
    ])
    expect(groups.map((g) => g.key)).toEqual(['related_to:out', 'related_to:in', 'supersedes:out'])
    for (const g of groups) expect(g.members).toEqual([{ nodeId: 'kb-00002' }])
  })
})

describe('INSPECTOR_PILL_MAX', () => {
  it('is 12 members', () => {
    expect(INSPECTOR_PILL_MAX).toBe(12)
  })
})

describe('entrySummary', () => {
  it('formats the pinned worked example', () => {
    expect(entrySummary(['decision', 'decision', 'factual_reference'])).toBe(
      '3 entries · decision 2, factual reference 1',
    )
  })

  it("returns '' for an empty array", () => {
    expect(entrySummary([])).toBe('')
  })

  it('always uses the plural "entries"', () => {
    expect(entrySummary(['decision'])).toBe('1 entries · decision 1')
  })

  it('appends · N other only when more than two distinct types exist', () => {
    expect(entrySummary(['decision', 'factual_reference'])).toBe(
      '2 entries · factual reference 1, decision 1',
    )
    expect(entrySummary(['decision', 'decision', 'factual_reference', 'lesson_learned'])).toBe(
      '4 entries · decision 2, factual reference 1 · 1 other',
    )
    expect(
      entrySummary(['decision', 'decision', 'factual_reference', 'lesson_learned', 'lesson_learned']),
    ).toBe('5 entries · decision 2, lesson learned 2 · 1 other')
  })

  it('breaks count ties by ENTRY_TYPES declaration order', () => {
    expect(entrySummary(['lesson_learned', 'decision', 'factual_reference'])).toBe(
      '3 entries · factual reference 1, decision 1 · 1 other',
    )
  })
})

describe('normalizeDirection', () => {
  it('maps the neighbors API direction values', () => {
    expect(normalizeDirection('outgoing')).toBe('out')
    expect(normalizeDirection('incoming')).toBe('in')
  })
})
