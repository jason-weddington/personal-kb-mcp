import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  FACET_TOP_N,
  INSPECTOR_PILL_MAX,
  computeFacets,
  entrySummary,
  excludedMemberIds,
  groupNeighbors,
  memberMatches,
  normalizeDirection,
} from '../graph/inspectorModel'
import type { ConnectionGroup, ConnInput, Facet, FilterMember } from '../graph/inspectorModel'

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

// ── Group filters (item C) ─────────────────────────────────────────────────────

afterEach(() => {
  vi.restoreAllMocks()
})

describe('FACET_TOP_N', () => {
  it('is 8 facets', () => {
    expect(FACET_TOP_N).toBe(8)
  })
})

describe('memberMatches', () => {
  const member = (over: Partial<FilterMember> = {}): FilterMember => ({
    nodeId: 'kb-00001',
    label: 'Decision doc',
    longTitle: null,
    nodeType: 'entry',
    entryType: 'decision',
    projectRef: 'p1',
    ...over,
  })
  const noNeighbors: ReadonlySet<string> = new Set()

  it('matches text case-insensitively over the label', () => {
    expect(memberMatches(member(), { text: 'DECI', facets: [] }, noNeighbors)).toBe(true)
    expect(memberMatches(member(), { text: 'deci', facets: [] }, noNeighbors)).toBe(true)
    expect(memberMatches(member(), { text: 'zzz', facets: [] }, noNeighbors)).toBe(false)
  })

  it("matches text over the member's longTitle", () => {
    const m = member({ label: 'x', longTitle: 'long thing' })
    expect(memberMatches(m, { text: 'long thi', facets: [] }, noNeighbors)).toBe(true)
  })

  it('empty and whitespace-only text matches everything', () => {
    expect(memberMatches(member(), { text: '', facets: [] }, noNeighbors)).toBe(true)
    expect(memberMatches(member(), { text: '   ', facets: [] }, noNeighbors)).toBe(true)
  })

  it('an entry_type facet matches only entry members of that type', () => {
    const other = member({ entryType: 'factual_reference' })
    const tag = member({ nodeType: 'tag', entryType: null })
    const f = { text: '', facets: ['entry_type:decision'] }
    expect(memberMatches(member(), f, noNeighbors)).toBe(true)
    expect(memberMatches(other, f, noNeighbors)).toBe(false)
    expect(memberMatches(tag, f, noNeighbors)).toBe(false)
  })

  it('a project facet matches only members of that project ref', () => {
    const other = member({ projectRef: 'p2', nodeId: 'kb-00002' })
    const f = { text: '', facets: ['project:p1'] }
    expect(memberMatches(member(), f, noNeighbors)).toBe(true)
    expect(memberMatches(other, f, noNeighbors)).toBe(false)
  })

  it('a node facet matches via the member neighbor ids and via the member own nodeId', () => {
    const viaNeighbor = member({ nodeId: 'kb-00001' })
    const f = { text: '', facets: ['node:tag:rust'] }
    expect(memberMatches(viaNeighbor, f, new Set(['tag:rust', 'tag:uv']))).toBe(true)
    expect(memberMatches(viaNeighbor, f, new Set(['tag:uv']))).toBe(false)
    const selfIsTag = member({ nodeId: 'tag:rust', nodeType: 'tag', entryType: null, projectRef: null })
    expect(memberMatches(selfIsTag, f, noNeighbors)).toBe(true)
  })

  it('two selected facets AND with each other', () => {
    const both = member()
    const onlyDecision = member({ projectRef: 'p2', nodeId: 'kb-00002' })
    const f = { text: '', facets: ['entry_type:decision', 'project:p1'] }
    expect(memberMatches(both, f, noNeighbors)).toBe(true)
    expect(memberMatches(onlyDecision, f, noNeighbors)).toBe(false)
  })

  it('the text filter ANDs with the facets', () => {
    const f = { text: 'DECI', facets: ['project:p1'] }
    expect(memberMatches(member(), f, noNeighbors)).toBe(true)
    expect(memberMatches(member({ label: 'other' }), f, noNeighbors)).toBe(false)
  })
})

describe('computeFacets', () => {
  const member = (id: string, over: Partial<FilterMember> = {}): FilterMember => ({
    nodeId: id,
    label: id,
    longTitle: null,
    nodeType: 'entry',
    entryType: null,
    projectRef: null,
    ...over,
  })

  it('returns the pinned worked example in the pinned order', () => {
    // 15 entry members: 10 decision (8 with p1), 4 factual_reference (2 with
    // p2), 1 lesson_learned.
    const members: FilterMember[] = []
    for (let i = 0; i < 10; i++) {
      members.push(member(`kb-d${i}`, { entryType: 'decision', projectRef: i < 8 ? 'p1' : null }))
    }
    for (let i = 0; i < 4; i++) {
      members.push(
        member(`kb-f${i}`, { entryType: 'factual_reference', projectRef: i < 2 ? 'p2' : null }),
      )
    }
    members.push(member('kb-l0', { entryType: 'lesson_learned' }))

    const neighborByMember = new Map<string, Array<{ id: string; type: string }>>()
    for (const m of members) neighborByMember.set(m.nodeId, [])
    // tag:rust adjacent to 7, note:n1 to 2, concept:rag to 2, tool:uv to 2,
    // tag:vllm to 1 member — assigned in member order.
    const adjacencyPlan: Array<[string, string]> = [
      ['tag:rust', 'tag'],
      ['tag:rust', 'tag'],
      ['tag:rust', 'tag'],
      ['tag:rust', 'tag'],
      ['tag:rust', 'tag'],
      ['tag:rust', 'tag'],
      ['tag:rust', 'tag'],
      ['note:n1', 'note'],
      ['note:n1', 'note'],
      ['concept:rag', 'concept'],
      ['concept:rag', 'concept'],
      ['tool:uv', 'tool'],
      ['tool:uv', 'tool'],
      ['tag:vllm', 'tag'],
    ]
    let idx = 0
    for (const [id, type] of adjacencyPlan) {
      const m = members[idx]
      neighborByMember.get(m.nodeId)!.push({ id, type })
      idx++
    }

    const nodeTypeById = new Map<string, string>([
      ...members.map((m) => [m.nodeId, 'entry'] as [string, string]),
      ['tag:rust', 'tag'],
      ['tag:vllm', 'tag'],
      ['note:n1', 'note'],
      ['concept:rag', 'concept'],
      ['tool:uv', 'tool'],
      ['tag:mass', 'tag'],
    ])
    const nodeLabelById = new Map<string, string>([
      ['tag:rust', 'rust'],
      ['tag:vllm', 'vllm'],
      ['note:n1', 'n1'],
      ['concept:rag', 'rag'],
      ['tool:uv', 'uv'],
      ['tag:mass', 'mass'],
    ])

    const facets = computeFacets({
      inspectedNodeId: 'tag:mass',
      members,
      neighborByMember,
      nodeTypeById,
      nodeLabelById,
    })
    expect(facets.map((f) => [f.key, f.count])).toEqual([
      ['entry_type:decision', 10],
      ['project:p1', 8],
      ['node:tag:rust', 7],
      ['entry_type:factual_reference', 4],
      ['node:note:n1', 2],
      ['project:p2', 2],
      ['node:concept:rag', 2],
      ['node:tool:uv', 2],
      ['entry_type:lesson_learned', 1],
      ['node:tag:vllm', 1],
    ])
    expect(facets.map((f) => f.label)).toEqual([
      'decision',
      'p1',
      'rust',
      'factual reference',
      'n1',
      'p2',
      'rag',
      'uv',
      'lesson learned',
      'vllm',
    ])
  })

  it('excludes the inspected node and entry-typed neighbours; a missing graph node falls back to the raw id', () => {
    const members = [member('kb-00001', { entryType: 'decision', projectRef: 'p1' })]
    const neighborByMember = new Map([
      ['kb-00001', [
        { id: 'tag:mass', type: 'tag' },
        { id: 'kb-00002', type: 'entry' },
        { id: 'tag:ghost', type: 'tag' },
      ]],
    ])
    const facets = computeFacets({
      inspectedNodeId: 'tag:mass',
      members,
      neighborByMember,
      nodeTypeById: new Map([
        ['kb-00001', 'entry'],
        ['kb-00002', 'entry'],
        ['tag:mass', 'tag'],
        ['tag:ghost', 'tag'],
      ]),
      nodeLabelById: new Map(),
    })
    expect(facets.map((f) => f.key)).toEqual(['entry_type:decision', 'project:p1', 'node:tag:ghost'])
    expect(facets[2].label).toBe('tag:ghost')
    expect(facets[2].nodeType).toBe('tag')
    expect(facets[0].nodeType).toBeNull()
    for (const f of facets as Facet[]) {
      expect(f.label).not.toMatch(/^\[kb-/)
    }
  })

  it('is scoped to the members it is given, never pooled across groups', () => {
    const groupA = [member('kb-a0', { entryType: 'decision', projectRef: 'pa' })]
    const groupB = [member('kb-b0', { entryType: 'lesson_learned', projectRef: 'pb' })]
    const nodeTypeById = new Map([['kb-a0', 'entry'], ['kb-b0', 'entry']])
    const neighborByMember = new Map<string, Array<{ id: string; type: string }>>()
    const input = {
      inspectedNodeId: 'tag:x',
      neighborByMember,
      nodeTypeById,
      nodeLabelById: new Map<string, string>(),
    }
    // each invocation sees only ITS group's members
    const facetsA = computeFacets({ ...input, members: groupA })
    expect(facetsA.map((f) => f.key)).toEqual(['entry_type:decision', 'project:pa'])
    const facetsB = computeFacets({ ...input, members: groupB })
    expect(facetsB.map((f) => f.key)).toEqual(['entry_type:lesson_learned', 'project:pb'])
  })
})

describe('excludedMemberIds', () => {
  const group = (key: string, nodeIds: string[]): ConnectionGroup => {
    const [edgeType, dir] = key.split(':')
    const g = groupNeighbors(nodeIds.map((id) => conn(id, edgeType, dir as 'in' | 'out')))
    return g[0]
  }
  const byProject = (projectOf: Map<string, string>) => (nodeId: string) =>
    projectOf.get(nodeId) === 'p1'

  it('subtracts the failing members of an actively filtered has_tag:in group', () => {
    const nodeIds = Array.from({ length: 15 }, (_, i) => `kb-001${String(i).padStart(2, '0')}`)
    const projectOf = new Map<string, string>()
    nodeIds.forEach((id, i) => projectOf.set(id, i < 6 ? 'p1' : 'p2'))
    const groups = [group('has_tag:in', nodeIds)]
    const excluded = excludedMemberIds(
      'tag:mass',
      groups,
      { 'has_tag:in': { text: '', facets: ['project:p1'] } },
      byProject(projectOf),
    )
    expect([...excluded].sort()).toEqual(nodeIds.filter((id) => projectOf.get(id) !== 'p1').sort())
  })

  it('a group with an empty filter contributes nothing', () => {
    const nodeIds = ['kb-00001', 'kb-00002']
    const groups = [group('has_tag:in', nodeIds)]
    const excluded = excludedMemberIds(
      'tag:mass',
      groups,
      { 'has_tag:in': { text: '', facets: [] } },
      () => false,
    )
    expect(excluded.size).toBe(0)
  })

  it('an unfiltered group contributes nothing', () => {
    const nodeIds = ['kb-00001', 'kb-00002']
    const groups = [group('has_tag:in', nodeIds)]
    const excluded = excludedMemberIds('tag:mass', groups, {}, () => false)
    expect(excluded.size).toBe(0)
  })

  it('the focused member is never excluded and the tripwire warn fires', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const groups = [group('has_tag:in', ['kb-00001', 'tag:mass'])]
    const excluded = excludedMemberIds(
      'tag:mass',
      groups,
      { 'has_tag:in': { text: '', facets: ['project:p1'] } },
      () => false,
    )
    expect(excluded.has('tag:mass')).toBe(false)
    expect([...excluded]).toEqual(['kb-00001'])
    expect(warn).toHaveBeenCalledWith('[graph-inspector] focused member excluded', {
      node_id: 'tag:mass',
      group_key: 'has_tag:in',
    })
  })

  it('does not warn when the focused member matches the filter', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const groups = [group('has_tag:in', ['kb-00001', 'tag:mass'])]
    const excluded = excludedMemberIds(
      'tag:mass',
      groups,
      { 'has_tag:in': { text: '', facets: ['project:p1'] } },
      () => true,
    )
    expect(excluded.size).toBe(0)
    expect(warn).not.toHaveBeenCalled()
  })
})
