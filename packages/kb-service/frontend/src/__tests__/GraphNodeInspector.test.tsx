import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import {
  DEFAULT_INSPECTOR_VIEW,
  GraphNodeInspector,
  InspectorConnections,
} from '../components/GraphNodeInspector'
import type { GraphNodeInspectorProps, InspectorMember } from '../components/GraphNodeInspector'
import type { InspectorViewState } from '../components/GraphExplorer'
import { drawNode } from '../graph/drawNode'
import type { NodeRenderState } from '../graph/drawNode'
import { EMPTY_TRAVERSAL, labelOpacity } from '../graph/explorerModel'
import type { ExplorerNode } from '../graph/explorerModel'
import { INSPECTOR_PILL_MAX, FACET_TOP_N, groupNeighbors } from '../graph/inspectorModel'
import type { ConnInput } from '../graph/inspectorModel'
import type { GetResponse, KnowledgeEntry } from '../kbTypes'

const openNodeDrawerMock = vi.fn()

vi.mock('../api', () => ({ getEntries: vi.fn() }))
vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: () => ({
    openEntry: vi.fn(),
    closeEntry: vi.fn(),
    openNode: openNodeDrawerMock,
    setNodeHandler: vi.fn(),
  }),
}))

import { getEntries } from '../api'

// ── Fixtures ─────────────────────────────────────────────────────────────────

const ENTRY_PROPS = {
  short_title: 'replaced below',
  long_title: 'replaced below',
  entry_type: 'decision',
  project_ref: 'p1',
  confidence_level: 0.9,
}

function entryNode(id: string, over: Partial<ExplorerNode> = {}): ExplorerNode {
  return {
    id,
    type: 'entry',
    label: `Title ${id}`,
    val: 1,
    properties: { ...ENTRY_PROPS, short_title: `Title ${id}`, long_title: `Long ${id}` },
    ...over,
  }
}

function makeEntry(id: string, over: Partial<KnowledgeEntry> = {}): KnowledgeEntry {
  return {
    id,
    project_ref: 'p1',
    short_title: `Title ${id}`,
    long_title: `Long ${id}`,
    knowledge_details: `Body for ${id}`,
    entry_type: 'decision',
    source_context: null,
    contributor: null,
    team: null,
    updated_by: null,
    sensitivity: null,
    confidence_level: 0.9,
    tags: [],
    hints: {},
    created_at: null,
    updated_at: null,
    last_accessed: null,
    expires_at: null,
    superseded_by: null,
    is_active: true,
    has_embedding: true,
    version: 1,
    ...over,
  }
}

function getResponse(entry: KnowledgeEntry): GetResponse {
  return { results: [{ id: entry.id, found: true, entry, pointer_rot: [] }] }
}

const tagNode = (id: string): ExplorerNode => ({
  id,
  type: 'tag',
  label: id.slice('tag:'.length),
  val: 1,
  properties: {},
})

const projectNode = (ref: string): ExplorerNode => ({
  id: `project:${ref}`,
  type: 'project',
  label: ref,
  val: 1,
  properties: {},
})

const conn = (nodeId: string, edgeType: string, dir: 'in' | 'out'): ConnInput => ({
  nodeId,
  edgeType,
  dir,
})

const nodeMap = (nodes: ExplorerNode[]) => new Map(nodes.map((n) => [n.id, n]))

const entryMember = (node: ExplorerNode, over: Partial<InspectorMember> = {}): InspectorMember => ({
  nodeId: node.id,
  label: `Title ${node.id}`,
  nodeType: 'entry',
  edgeType: 'related_to',
  dir: 'out',
  tooltip: `Long ${node.id}`,
  entryType: 'decision',
  projectRef: 'p1',
  ...over,
})

const memberMap = (entries: Array<[string, InspectorMember]>) => new Map(entries)

type RenderProps = Partial<GraphNodeInspectorProps> & { node: ExplorerNode }

function renderInspector(props: RenderProps) {
  const { node, ...rest } = props
  const all: GraphNodeInspectorProps = {
    groups: [],
    nodeById: nodeMap([node]),
    view: DEFAULT_INSPECTOR_VIEW,
    onViewChange: () => {},
    onOpenNode: () => {},
    onHover: () => {},
    onClose: () => {},
    ...rest,
    node,
  }
  return render(<GraphNodeInspector {...all} />)
}

const region = () => screen.getByRole('region', { name: 'Node inspector' })

beforeEach(() => {
  vi.mocked(getEntries).mockReset()
  openNodeDrawerMock.mockClear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ── Headers ──────────────────────────────────────────────────────────────────

describe('GraphNodeInspector headers', () => {
  it('non-entry header: type chip, label, id subtitle and the entry summary; no Val or Properties dump', () => {
    const a = entryNode('kb-00001')
    const b = entryNode('kb-00002', {
      properties: { ...ENTRY_PROPS, short_title: 'b', long_title: 'b', entry_type: 'factual_reference' },
    })
    const tag: ExplorerNode = { ...tagNode('tag:alpha'), properties: { rot: 'never-rendered' } }
    renderInspector({
      node: tag,
      groups: groupNeighbors([conn('kb-00001', 'has_tag', 'in'), conn('kb-00002', 'has_tag', 'in')]),
      nodeById: nodeMap([tag, a, b]),
    })
    expect(screen.getByText('alpha')).toBeInTheDocument()
    expect(screen.getByText('tag:alpha')).toBeInTheDocument()
    expect(screen.getByText('2 entries · factual reference 1, decision 1')).toBeInTheDocument()
    expect(screen.queryByText(/Val:/)).toBeNull()
    expect(screen.queryByText('Properties')).toBeNull()
    expect(screen.queryByText(/rot/)).toBeNull()
  })

  it('entry header: type chip, short title, long title, meta line, tag pills and project pill', () => {
    const node = entryNode('kb-00001')
    const other = entryNode('kb-00002')
    const tag = tagNode('tag:alpha')
    const proj = projectNode('p1')
    renderInspector({
      node,
      groups: groupNeighbors([
        conn('tag:alpha', 'has_tag', 'out'),
        conn('project:p1', 'in_project', 'out'),
        conn('kb-00002', 'related_to', 'out'),
      ]),
      nodeById: nodeMap([node, other, tag, proj]),
    })
    expect(screen.getByText('decision')).toBeInTheDocument()
    expect(screen.getByText('Title kb-00001')).toBeInTheDocument()
    expect(screen.getByText('Long kb-00001')).toBeInTheDocument()
    expect(screen.getByText('p1 · 0.90')).toBeInTheDocument()
    // the has_tag:out members are the header tag pills, in_project:out the project pill
    expect(screen.getByRole('button', { name: 'alpha' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'p1' })).toBeInTheDocument()
    // ... and those two groups are NOT repeated in the connections section
    expect(screen.queryByText(/has_tag:out/)).toBeNull()
    expect(screen.queryByText(/in_project:out/)).toBeNull()
    expect(screen.getByText('related_to → 1')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Title kb-00002' })).toBeInTheDocument()
    // graph-variant labels carry no '[kb-...]' prefix
    expect(/\[kb-/.test(region().textContent ?? '')).toBe(false)
  })

  it('header tag pills and the project pill call onOpenNode with that node id', () => {
    const node = entryNode('kb-00001')
    const tag = tagNode('tag:alpha')
    const proj = projectNode('p1')
    const onOpenNode = vi.fn()
    renderInspector({
      node,
      groups: groupNeighbors([conn('tag:alpha', 'has_tag', 'out'), conn('project:p1', 'in_project', 'out')]),
      nodeById: nodeMap([node, tag, proj]),
      onOpenNode,
    })
    fireEvent.click(screen.getByRole('button', { name: 'alpha' }))
    expect(onOpenNode).toHaveBeenCalledWith('tag:alpha')
    fireEvent.click(screen.getByRole('button', { name: 'p1' }))
    expect(onOpenNode).toHaveBeenCalledWith('project:p1')
  })
})

// ── Pills vs rows ────────────────────────────────────────────────────────────

describe('InspectorConnections rendering', () => {
  const manyEntries = (n: number) => Array.from({ length: n }, (_, i) => entryNode(`kb-9${String(i).padStart(4, '0')}`))

  it(`a group with <= ${INSPECTOR_PILL_MAX} members renders always-visible pills`, () => {
    const members = manyEntries(3)
    const groups = groupNeighbors(members.map((m) => conn(m.id, 'related_to', 'out')))
    render(
      <InspectorConnections
        groups={groups}
        membersById={memberMap(members.map((m) => [m.id, entryMember(m)]))}
        onOpenNode={() => {}}
      />,
    )
    expect(screen.getByText('related_to → 3')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Title kb-90000' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Title kb-90002' })).toBeInTheDocument()
  })

  it('an in-direction header renders the ← arrow and the count', () => {
    const groups = groupNeighbors([conn('kb-00001', 'has_tag', 'in'), conn('kb-00002', 'has_tag', 'in')])
    render(<InspectorConnections groups={groups} membersById={new Map()} onOpenNode={() => {}} />)
    expect(screen.getByText('has_tag ← 2')).toBeInTheDocument()
  })

  it('a 31-member group renders a collapsed accordion whose key, arrow and count are in the control text and expands to all 31 rows', () => {
    const members = manyEntries(31)
    const groups = groupNeighbors(members.map((m) => conn(m.id, 'related_to', 'out')))
    render(
      <InspectorConnections
        groups={groups}
        membersById={memberMap(members.map((m) => [m.id, entryMember(m)]))}
        onOpenNode={() => {}}
      />,
    )
    const control = screen.getByRole('button', { name: 'related_to:out → 31' })
    expect(control).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(control)
    expect(screen.getByRole('button', { name: 'related_to:out → 31' })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    expect(screen.getAllByText(/Title kb-9\d{4}/)).toHaveLength(31)
    expect(screen.queryByText(/\.\.\. and/)).toBeNull()
  })

  it('rows are clickable and every row calls onOpenNode with the member node id', () => {
    const members = manyEntries(13)
    const groups = groupNeighbors(members.map((m) => conn(m.id, 'related_to', 'out')))
    const onOpenNode = vi.fn()
    render(
      <InspectorConnections
        groups={groups}
        membersById={memberMap(members.map((m) => [m.id, entryMember(m)]))}
        onOpenNode={onOpenNode}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'related_to:out → 13' }))
    fireEvent.click(screen.getByRole('button', { name: /Title kb-90005/ }))
    expect(onOpenNode).toHaveBeenCalledWith('kb-90005')
  })

  it('a missing member warns and renders the raw node id as its label', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const groups = groupNeighbors([conn('kb-90001', 'related_to', 'out')])
    render(<InspectorConnections groups={groups} membersById={new Map()} onOpenNode={() => {}} />)
    expect(warn).toHaveBeenCalledWith('[graph-inspector] member not in payload', {
      group_key: 'related_to:out',
      node_id: 'kb-90001',
    })
    expect(screen.getByRole('button', { name: 'kb-90001' })).toBeInTheDocument()
  })

  it('Tooltip wraps a member only when its tooltip is a non-empty string', async () => {
    const e1 = entryNode('kb-90001')
    const e2 = entryNode('kb-90002')
    const groups = groupNeighbors([conn(e1.id, 'related_to', 'out'), conn(e2.id, 'related_to', 'out')])
    render(
      <InspectorConnections
        groups={groups}
        membersById={memberMap([
          [e1.id, entryMember(e1)],
          [e2.id, entryMember(e2, { tooltip: '', entryType: null, projectRef: null })],
        ])}
        onOpenNode={() => {}}
      />,
    )
    const pill1 = screen.getByRole('button', { name: 'Title kb-90001' })
    fireEvent.mouseOver(pill1)
    await waitFor(() => expect(screen.getByText('Long kb-90001')).toBeInTheDocument())
    fireEvent.mouseOut(pill1)
    const pill2 = screen.getByRole('button', { name: 'Title kb-90002' })
    fireEvent.mouseOver(pill2)
    await new Promise((r) => setTimeout(r, 150))
    expect(screen.queryByText('Long kb-90002')).toBeNull()
  })

  it('an entry row shows the bold one-line title, clamped long title, type dot and project; a non-entry row its label and type chip', () => {
    const e = entryNode('kb-90001')
    const t = tagNode('tag:x')
    const padding = Array.from({ length: INSPECTOR_PILL_MAX }, (_, i) => `pad:${i}`)
    const groups = groupNeighbors([
      conn(e.id, 'related_to', 'out'),
      conn(t.id, 'related_to', 'out'),
      ...padding.map((id) => conn(id, 'related_to', 'out')),
    ])
    const membersById = memberMap([
      [e.id, entryMember(e)],
      [
        t.id,
        {
          nodeId: t.id,
          label: 'tag:x',
          nodeType: 'tag',
          edgeType: 'related_to',
          dir: 'out',
          entryType: null,
          projectRef: null,
        },
      ],
      ...padding.map(
        (id) =>
          [
            id,
            {
              nodeId: id,
              label: id,
              nodeType: 'tag',
              edgeType: 'related_to',
              dir: 'out',
              entryType: null,
              projectRef: null,
            },
          ] as [string, InspectorMember],
      ),
    ])
    render(
      <InspectorConnections groups={groups} membersById={membersById} onOpenNode={() => {}} />,
    )
    fireEvent.click(screen.getByRole('button', { name: /related_to:out/ }))
    expect(screen.getByText('Title kb-90001')).toBeInTheDocument()
    expect(screen.getByText('Long kb-90001')).toBeInTheDocument()
    expect(screen.getByText('p1')).toBeInTheDocument()
    expect(screen.getByText('tag:x')).toBeInTheDocument()
    expect(screen.getAllByText('tag').length).toBeGreaterThan(0)
  })
})

// ── Hover wiring ─────────────────────────────────────────────────────────────

describe('inspector hover', () => {
  it('pills drive onHover with the member id and null on mouse out', () => {
    const onHover = vi.fn()
    const e = entryNode('kb-90001')
    const groups = groupNeighbors([conn(e.id, 'related_to', 'out')])
    render(
      <InspectorConnections
        groups={groups}
        membersById={memberMap([[e.id, entryMember(e)]])}
        onOpenNode={() => {}}
        onHover={onHover}
      />,
    )
    const pill = screen.getByRole('button', { name: 'Title kb-90001' })
    fireEvent.mouseOver(pill)
    expect(onHover).toHaveBeenCalledWith('kb-90001')
    fireEvent.mouseOut(pill)
    expect(onHover).toHaveBeenCalledWith(null)
  })

  it('the hoveredId write is the one drawNode highlights: the label draws at full opacity', () => {
    const e = entryNode('kb-90001')
    const groups = groupNeighbors([conn(e.id, 'related_to', 'out')])
    const rs: NodeRenderState = {
      hoveredId: null,
      highlightedId: null,
      focusSet: null,
      soloType: null,
      traversal: EMPTY_TRAVERSAL,
      activeNeighbors: new Set(),
      now: 0,
    }
    render(
      <InspectorConnections
        groups={groups}
        membersById={memberMap([[e.id, entryMember(e)]])}
        onOpenNode={() => {}}
        onHover={(id) => {
          rs.hoveredId = id
        }}
      />,
    )
    fireEvent.mouseOver(screen.getByRole('button', { name: 'Title kb-90001' }))
    expect(rs.hoveredId).toBe('kb-90001')
    expect(
      labelOpacity({
        hovered: true,
        highlighted: false,
        nodeState: undefined,
        queryMode: null,
        inFocus: false,
        solo: false,
        globalScale: 0.5,
      }),
    ).toBe(1)
    const log: Array<[string, ...unknown[]]> = []
    const ctx = new Proxy(
      {
        beginPath: () => log.push(['beginPath']),
        arc: () => log.push(['arc']),
        roundRect: () => log.push(['roundRect']),
        fill: () => log.push(['fill']),
        fillText: (...a: unknown[]) => log.push(['fillText', ...a]),
        measureText: () => ({ width: 40 }),
      } as Record<string, unknown>,
      {
        set(target, prop, value) {
          log.push(['set', String(prop), value])
          target[String(prop)] = value
          return true
        },
      },
    ) as unknown as CanvasRenderingContext2D
    drawNode(e, ctx, 0.5, rs)
    expect(log).toContainEqual(['fillText', 'Title kb-90001', expect.any(Number), expect.any(Number)])
    fireEvent.mouseOut(screen.getByRole('button', { name: 'Title kb-90001' }))
    expect(rs.hoveredId).toBe(null)
  })
})

// ── Details section ──────────────────────────────────────────────────────────

describe('entry inspector Details', () => {
  it('is collapsed by default and fetches the entry exactly once on expand', async () => {
    vi.mocked(getEntries).mockResolvedValue(
      getResponse(makeEntry('kb-00001', { updated_at: '2026-03-03T12:00:00Z' })),
    )
    const node = entryNode('kb-00001')
    const view = renderInspector({ node })
    expect(vi.mocked(getEntries)).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('Details'))
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(1)
    expect(vi.mocked(getEntries)).toHaveBeenCalledWith(['kb-00001'], { touch: false })
    await act(async () => {})
    await waitFor(() => expect(screen.getByText('Body for kb-00001')).toBeInTheDocument())
    const date = new Date('2026-03-03T12:00:00Z').toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
    expect(screen.getByText(`p1 · 0.90 · updated ${date}`)).toBeInTheDocument()
    // a re-render does not refetch
    view.rerender(
      <GraphNodeInspector
        node={node}
        groups={[]}
        nodeById={nodeMap([node])}
        view={DEFAULT_INSPECTOR_VIEW}
        onViewChange={() => {}}
        onOpenNode={() => {}}
        onHover={() => {}}
        onClose={() => {}}
      />,
    )
    expect(vi.mocked(getEntries)).toHaveBeenCalledTimes(1)
  })

  it('renders the superseded button and pointer-rot alerts after the fetch', async () => {
    vi.mocked(getEntries).mockResolvedValue({
      results: [
        {
          id: 'kb-00001',
          found: true,
          entry: makeEntry('kb-00001', { superseded_by: 'kb-00002' }),
          pointer_rot: [{ target_id: 'kb-00003', superseded_by: 'kb-00004' }],
        },
      ],
    })
    const node = entryNode('kb-00001')
    renderInspector({ node })
    fireEvent.click(screen.getByText('Details'))
    await act(async () => {})
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Superseded by kb-00002' })).toBeInTheDocument(),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Superseded by kb-00002' }))
    expect(openNodeDrawerMock).toHaveBeenCalledWith('kb-00002')
    expect(screen.getByText(/kb-00003 superseded by kb-00004/)).toBeInTheDocument()
  })

  it('a failed fetch warns and renders a warning Alert inside Details without unmounting', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.mocked(getEntries).mockRejectedValue(new Error('boom'))
    const node = entryNode('kb-00001')
    renderInspector({ node })
    fireEvent.click(screen.getByText('Details'))
    await act(async () => {})
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(warn).toHaveBeenCalledWith('[graph-inspector] details fetch failed', {
      node_id: 'kb-00001',
      error: 'boom',
    })
    expect(region()).toBeInTheDocument()
    expect(screen.getByText('Title kb-00001')).toBeInTheDocument()
  })

  it('a not-found entry warns and renders a warning Alert inside Details', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.mocked(getEntries).mockResolvedValue({
      results: [{ id: 'kb-00001', found: false, entry: null, pointer_rot: [] }],
    })
    const node = entryNode('kb-00001')
    renderInspector({ node })
    fireEvent.click(screen.getByText('Details'))
    await act(async () => {})
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(warn).toHaveBeenCalledWith('[graph-inspector] details fetch failed', {
      node_id: 'kb-00001',
      error: 'not found',
    })
    expect(region()).toBeInTheDocument()
  })
})

// ── View state ───────────────────────────────────────────────────────────────

describe('InspectorViewState', () => {
  it('has the pinned defaults and renders no filter UI', () => {
    expect(DEFAULT_INSPECTOR_VIEW).toEqual({ expanded: {}, filters: {}, scrollTop: 0 })
    const node = entryNode('kb-00001')
    const others = manyEntriesOf(13)
    const onViewChange = vi.fn()
    renderInspector({
      node,
      groups: groupNeighbors(others.map((m) => conn(m.id, 'related_to', 'out'))),
      nodeById: nodeMap([node, ...others]),
      view: { ...DEFAULT_INSPECTOR_VIEW },
      onViewChange,
    })
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.queryByRole('combobox')).toBeNull()
    fireEvent.click(screen.getByText(/related_to:out/))
    expect(onViewChange).toHaveBeenCalled()
    for (const v of onViewChange.mock.calls.map((c) => c[0] as InspectorViewState)) {
      expect(v.filters).toEqual({})
    }
  })

  it('re-targeting the inspector resets expanded and scrollTop to the defaults', () => {
    const a = tagNode('tag:hub-a')
    const b = tagNode('tag:hub-b')
    const membersOfA = Array.from({ length: 13 }, (_, i) => entryNode(`kb-a${String(i).padStart(4, '0')}`))
    const membersOfB = Array.from({ length: 13 }, (_, i) => entryNode(`kb-b${String(i).padStart(4, '0')}`))
    const nodeById = nodeMap([a, b, ...membersOfA, ...membersOfB])
    const connsByNode: Record<string, ConnInput[]> = {
      'tag:hub-a': [...membersOfA.map((m) => conn(m.id, 'related_to', 'in')), conn('tag:hub-b', 'related_to', 'out')],
      'tag:hub-b': membersOfB.map((m) => conn(m.id, 'related_to', 'in')),
    }

    // Mirrors GraphExplorer: the view reset happens in the same update that
    // changes the target (showDetails), never in an effect.
    function Harness() {
      const [node, setNode] = useState<ExplorerNode>(a)
      const [view, setView] = useState<InspectorViewState>(DEFAULT_INSPECTOR_VIEW)
      const retarget = (n: ExplorerNode) => {
        setNode(n)
        setView(DEFAULT_INSPECTOR_VIEW)
      }
      return (
        <GraphNodeInspector
          node={node}
          groups={groupNeighbors(connsByNode[node.id] ?? [])}
          nodeById={nodeById}
          view={view}
          onViewChange={setView}
          onOpenNode={(id) => {
            const n = nodeById.get(id)
            if (n) retarget(n)
          }}
          onHover={() => {}}
          onClose={() => {}}
        />
      )
    }

    render(<Harness />)
    const el = region()
    fireEvent.click(screen.getByRole('button', { name: /related_to:in/ }))
    expect(screen.getByRole('button', { name: /related_to:in/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    act(() => {
      el.scrollTop = 150
    })
    fireEvent.scroll(el)
    expect(el.scrollTop).toBe(150)
    // walk to node B through the same select path a member click uses
    fireEvent.click(screen.getByRole('button', { name: 'hub-b' }))
    expect(screen.getByRole('button', { name: /related_to:in/ })).toHaveAttribute(
      'aria-expanded',
      'false',
    )
    expect(el).toBeInTheDocument()
    expect(el.scrollTop).toBe(0)
  })
})

function manyEntriesOf(n: number): ExplorerNode[] {
  return Array.from({ length: n }, (_, i) => entryNode(`kb-9${String(i).padStart(4, '0')}`))
}

// ── Group filters (item C) ─────────────────────────────────────────────────

type MemberSpec = {
  id: string
  label?: string
  longTitle?: string | null
  entryType?: string | null
  projectRef?: string | null
  neighbors?: string[]
}

/** A big-group fixture: the members, their group, the graph nodes and adjacency. */
function buildGroup(specs: MemberSpec[], edgeType = 'has_tag', dir: 'in' | 'out' = 'in') {
  const groups = groupNeighbors(specs.map((s) => conn(s.id, edgeType, dir)))
  const membersById = new Map<string, InspectorMember>()
  const nodeById = new Map<string, { id: string; label: string; type: string }>()
  const adjacency = new Map<string, Array<{ node: string; type: string }>>()
  for (const s of specs) {
    const longTitle = s.longTitle ?? null
    membersById.set(s.id, {
      nodeId: s.id,
      label: s.label ?? s.id,
      nodeType: 'entry',
      edgeType,
      dir,
      tooltip: longTitle !== null && longTitle.length > 0 ? longTitle : undefined,
      entryType: s.entryType ?? null,
      projectRef: s.projectRef ?? null,
      longTitle,
    })
    nodeById.set(s.id, { id: s.id, label: s.label ?? s.id, type: 'entry' })
    adjacency.set(
      s.id,
      (s.neighbors ?? []).map((n) => ({ node: n, type: edgeType })),
    )
    for (const n of s.neighbors ?? []) {
      if (nodeById.has(n)) continue
      const sep = n.indexOf(':')
      nodeById.set(n, {
        id: n,
        label: sep === -1 ? n : n.slice(sep + 1),
        type: sep === -1 ? 'unknown' : n.slice(0, sep),
      })
    }
  }
  return { groups, membersById, nodeById, adjacency }
}

const massSpecs = (): MemberSpec[] => {
  const out: MemberSpec[] = []
  for (let i = 0; i < 15; i++) {
    out.push({
      id: `kb-001${String(i).padStart(2, '0')}`,
      label: `Mass entry ${i}`,
      entryType: i <= 5 ? 'decision' : i <= 12 ? 'factual_reference' : 'lesson_learned',
      projectRef: i <= 5 ? 'p1' : 'p2',
    })
  }
  return out
}

const KEY = 'has_tag:in'
const SUMMARY = `${KEY} ← 15`
const rowButtons = (pattern: RegExp = /Mass entry \d+/) =>
  screen.queryAllByRole('button', { name: pattern })
const chipButtons = () =>
  screen
    .queryAllByRole('button', { name: /\S.* \d+$/ })
    .filter((c) => c.getAttribute('aria-pressed') !== null)
const expandedView = (filters: InspectorViewState['filters'] = {}): InspectorViewState => ({
  expanded: { [KEY]: true },
  filters,
  scrollTop: 0,
})

function renderFiltered(
  specs: MemberSpec[],
  opts: { initialView?: InspectorViewState; filtering?: boolean } = {},
) {
  const { groups, membersById, nodeById, adjacency } = buildGroup(specs)
  const views: InspectorViewState[] = []
  const filtering = opts.filtering ?? true
  function Harness() {
    const [view, setView] = useState(opts.initialView ?? DEFAULT_INSPECTOR_VIEW)
    const [revealed, setRevealed] = useState<ReadonlySet<string>>(() => new Set())
    return (
      <InspectorConnections
        groups={groups}
        membersById={membersById}
        onOpenNode={() => {}}
        view={view}
        onViewChange={(v: InspectorViewState) => {
          views.push(v)
          setView(v)
        }}
        filtering={
          filtering
            ? {
                viewState: view,
                onViewChange: (v: InspectorViewState) => {
                  views.push(v)
                  setView(v)
                },
                nodeById,
                adjacency,
                inspectedNodeId: 'tag:mass',
              }
            : undefined
        }
        revealedFacetGroups={revealed}
        onRevealFacetGroup={(key) => setRevealed(new Set([...revealed, key]))}
      />
    )
  }
  const result = render(<Harness />)
  return { views, container: result.container }
}

describe('InspectorConnections group filters', () => {
  it('a 15-member group WITHOUT the filtering prop renders no filter UI and no of-N header', () => {
    const { container } = renderFiltered(massSpecs(), { filtering: false })
    expect(screen.getByRole('button', { name: SUMMARY })).toBeInTheDocument()
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(container.querySelectorAll('[aria-pressed]')).toHaveLength(0)
    expect(screen.queryByText(/^\d+ of \d+$/)).toBeNull()
  })

  it('a group with <= INSPECTOR_PILL_MAX members renders no filter UI even with filtering present', () => {
    const { container } = renderFiltered(massSpecs().slice(0, 5))
    expect(container.querySelectorAll('[role="textbox"], input[type="text"]')).toHaveLength(0)
    expect(container.querySelectorAll('[aria-pressed]')).toHaveLength(0)
  })

  it('a 15-member group with filtering renders the filter input and the pinned facet order', () => {
    renderFiltered(massSpecs(), { initialView: expandedView() })
    const input = screen.getByTestId(`group-filter-${KEY}`)
    expect(input).toBeInTheDocument()
    expect(screen.queryAllByRole('textbox')).toHaveLength(1)
    expect(chipButtons().map((c) => c.textContent)).toEqual([
      'p2 9',
      'factual reference 7',
      'decision 6',
      'p1 6',
      'lesson learned 2',
    ])
    expect(screen.queryByText(/^\+?\d+$/)).toBeNull()
    expect(screen.queryByText(/^\d+ of \d+$/)).toBeNull()
  })

  it('shows the top FACET_TOP_N facets plus a +N chip that reveals the rest', () => {
    const specs = Array.from({ length: 15 }, (_, i) => ({
      id: `kb-002${String(i).padStart(2, '0')}`,
      label: `Deep ${i}`,
      entryType: 'decision',
      projectRef: `p${String(i + 1).padStart(2, '0')}`,
    }))
    renderFiltered(specs, { initialView: expandedView() })
    const chip = (name: string) => screen.getByRole('button', { name })
    expect(chip('decision 15')).toBeInTheDocument()
    expect(chip('p01 1')).toBeInTheDocument()
    expect(chip('p07 1')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'p08 1' })).toBeNull()
    expect(chip(`+${16 - FACET_TOP_N}`)).toBeInTheDocument()
    fireEvent.click(chip(`+${16 - FACET_TOP_N}`))
    expect(chip('p08 1')).toBeInTheDocument()
    expect(chip('p15 1')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: `+${16 - FACET_TOP_N}` })).toBeNull()
  })

  it('facet chips carry the legend colour of their node type and the exact label-count text', () => {
    const specs = massSpecs().map((s, i) => (i < 7 ? { ...s, neighbors: ['tag:rust'] } : s))
    renderFiltered(specs, { initialView: expandedView() })
    const project = screen.getByRole('button', { name: 'p2 9' })
    expect(project).toHaveStyle({ backgroundColor: '#ff9800', color: '#0a0a0f' })
    expect(project).toHaveTextContent('p2 9')
    const tag = screen.getByRole('button', { name: 'rust 7' })
    expect(tag).toHaveStyle({ backgroundColor: '#00bcd4', color: '#0a0a0f' })
    expect(tag).toHaveTextContent('rust 7')
    const entryType = screen.getByRole('button', { name: 'factual reference 7' })
    expect(entryType).toHaveTextContent('factual reference 7')
  })

  it('clicking a facet chip toggles aria-pressed and writes the facet key through onViewChange', () => {
    const { views } = renderFiltered(massSpecs(), { initialView: expandedView() })
    const chip = screen.getByRole('button', { name: 'decision 6' })
    expect(chip).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(chip)
    expect(views.at(-1)!.filters).toEqual({ [KEY]: { text: '', facets: ['entry_type:decision'] } })
    expect(chip).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(chip)
    expect(views.at(-1)!.filters).toEqual({ [KEY]: { text: '', facets: [] } })
    expect(chip).toHaveAttribute('aria-pressed', 'false')
  })

  it('selected facets AND with each other and with the text filter over the rendered rows', () => {
    const specs: MemberSpec[] = [
      { id: 'kb-01000', label: 'ollama alpha', entryType: 'decision', projectRef: 'p1' },
      { id: 'kb-01001', label: 'beta', entryType: 'decision', projectRef: 'p1' },
      { id: 'kb-01002', label: 'ollama gamma', entryType: 'decision', projectRef: 'p2' },
      { id: 'kb-01003', label: 'ollama delta', entryType: 'factual_reference', projectRef: 'p1' },
      ...Array.from({ length: 11 }, (_, i) => ({
        id: `kb-011${String(i).padStart(2, '0')}`,
        label: `padding ${i}`,
        entryType: 'decision',
        projectRef: 'p2',
      })),
    ]
    renderFiltered(specs, {
      initialView: expandedView({
        [KEY]: { text: 'ollama', facets: ['entry_type:decision', 'project:p1'] },
      }),
    })
    expect(rowButtons(/ollama alpha|beta|ollama gamma|ollama delta|padding \d+/)).toHaveLength(1)
    expect(screen.getByRole('button', { name: /ollama alpha/ })).toBeInTheDocument()
    expect(screen.queryByTestId(`inspector-empty-${KEY}`)).toBeNull()
  })

  it('after selecting a facet chip, the rendered row count equals the chip count', () => {
    renderFiltered(massSpecs(), { initialView: expandedView() })
    fireEvent.click(screen.getByRole('button', { name: 'decision 6' }))
    expect(rowButtons()).toHaveLength(6)
  })

  it('zero matches renders No matching members. and zero rows', () => {
    renderFiltered(massSpecs(), { initialView: expandedView({ [KEY]: { text: 'zzz', facets: [] } }) })
    const empty = screen.getByTestId(`inspector-empty-${KEY}`)
    expect(empty).toHaveTextContent('No matching members.')
    expect(rowButtons()).toHaveLength(0)
  })

  it('the header shows <matching> of <total> while a filter is active and Clear resets it', () => {
    const { views } = renderFiltered(massSpecs(), { initialView: expandedView() })
    fireEvent.click(screen.getByRole('button', { name: 'decision 6' }))
    expect(screen.getByText('6 of 15')).toBeInTheDocument()
    const clear = screen.getByTestId(`group-clear-${KEY}`)
    expect(clear).toHaveAccessibleName('Clear')
    fireEvent.click(clear)
    expect(views.at(-1)!.filters).toEqual({ [KEY]: { text: '', facets: [] } })
    expect(screen.queryByText(/^\d+ of \d+$/)).toBeNull()
    expect(screen.queryByTestId(`group-clear-${KEY}`)).toBeNull()
    expect(rowButtons()).toHaveLength(15)
  })

  it('filter edits write only filters[groupKey]: expanded and scrollTop are untouched', () => {
    const { views } = renderFiltered(massSpecs(), { initialView: expandedView() })
    const input = screen.getByTestId(`group-filter-${KEY}`) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'mass' } })
    fireEvent.click(screen.getByRole('button', { name: 'decision 6' }))
    fireEvent.click(screen.getByRole('button', { name: 'p1 6' }))
    for (const v of views) {
      expect(v.expanded).toEqual({ [KEY]: true })
      expect(v.scrollTop).toBe(0)
      expect(Object.keys(v.filters)).toEqual([KEY])
    }
  })

  it('typing three characters over a missing member warns once, not three times', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const specs = massSpecs()
    const { groups, membersById, nodeById, adjacency } = buildGroup(specs)
    membersById.delete('kb-00107')
    function Harness() {
      const [view, setView] = useState<InspectorViewState>(expandedView())
      return (
        <InspectorConnections
          groups={groups}
          membersById={membersById}
          onOpenNode={() => {}}
          filtering={{
            viewState: view,
            onViewChange: setView,
            nodeById,
            adjacency,
            inspectedNodeId: 'tag:mass',
          }}
        />
      )
    }
    render(<Harness />)
    const input = screen.getByTestId(`group-filter-${KEY}`)
    for (const value of ['m', 'ma', 'mas']) {
      fireEvent.change(input, { target: { value } })
    }
    expect(
      warn.mock.calls.filter((c: unknown[]) => c[0] === '[graph-inspector] member not in payload'),
    ).toHaveLength(1)
  })
})

describe('GraphNodeInspector group filters', () => {
  function massFixture() {
    const nodes: ExplorerNode[] = [
      { id: 'tag:mass', type: 'tag', label: 'mass', val: 15, properties: {}, x: 40, y: 70 },
      ...Array.from({ length: 15 }, (_, i) => {
        const id = `kb-001${String(i).padStart(2, '0')}`
        const deep = i === 0 ? 'Deep background for mass entry 0' : `Mass entry ${i} detail`
        return {
          id,
          type: 'entry',
          label: `Mass entry ${i}`,
          val: 1,
          properties: {
            project_ref: i <= 5 ? 'p1' : 'p2',
            short_title: `Mass entry ${i}`,
            long_title: deep,
            entry_type: i <= 5 ? 'decision' : i <= 12 ? 'factual_reference' : 'lesson_learned',
          },
          x: 20 + i * 2,
          y: 70,
        }
      }),
    ]
    const links = Array.from({ length: 15 }, (_, i) => ({
      source: `kb-001${String(i).padStart(2, '0')}`,
      target: 'tag:mass',
      type: 'has_tag',
    }))
    const adjacency = new Map<string, Array<{ node: string; type: string }>>()
    for (const l of links) {
      adjacency.set(l.source, [{ node: l.target, type: l.type }])
      adjacency.set(l.target, [
        ...(adjacency.get(l.target) ?? []),
        { node: l.source, type: l.type },
      ])
    }
    return { nodes, links, adjacency }
  }

  function renderMassHarness(
    connsByNode: Record<string, ConnInput[]>,
    nodeById: Map<string, ExplorerNode>,
    adjacency: Map<string, Array<{ node: string; type: string }>>,
    start: ExplorerNode,
  ) {
    function Harness() {
      const [node, setNode] = useState(start)
      const [view, setView] = useState<InspectorViewState>(DEFAULT_INSPECTOR_VIEW)
      const retarget = (n: ExplorerNode) => {
        setNode(n)
        setView(DEFAULT_INSPECTOR_VIEW)
      }
      return (
        <GraphNodeInspector
          node={node}
          groups={groupNeighbors(connsByNode[node.id] ?? [])}
          nodeById={nodeById}
          adjacency={adjacency}
          view={view}
          onViewChange={setView}
          onOpenNode={(id) => {
            const n = nodeById.get(id)
            if (n) retarget(n)
          }}
          onHover={() => {}}
          onClose={() => {}}
        />
      )
    }
    return render(<Harness />)
  }

  it('typing in the title filter narrows the rows via the member longTitle from node.properties', () => {
    const { nodes, adjacency } = massFixture()
    const nodeById = nodeMap(nodes)
    const conns: Record<string, ConnInput[]> = {
      'tag:mass': Array.from({ length: 15 }, (_, i) =>
        conn(`kb-001${String(i).padStart(2, '0')}`, 'has_tag', 'in'),
      ),
    }
    renderMassHarness(conns, nodeById, adjacency, nodes[0])
    fireEvent.click(screen.getByRole('button', { name: SUMMARY }))
    const input = screen.getByTestId(`group-filter-${KEY}`) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'deep background' } })
    expect(input.value).toBe('deep background')
    expect(rowButtons()).toHaveLength(1)
    expect(screen.getByRole('button', { name: /Mass entry 0/ })).toBeInTheDocument()
  })

  it('re-targeting clears the filter text, the facet selection and the +N reveal', () => {
    // 15 members with 15 distinct project refs: 16 facets, so the +8 reveal
    // chip exists and carries over only if the reset is broken.
    const members = Array.from({ length: 15 }, (_, i) => {
      const id = `kb-003${String(i).padStart(2, '0')}`
      return {
        id,
        type: 'entry' as const,
        label: `Member ${i}`,
        val: 1,
        properties: {
          project_ref: `p${String(i + 1).padStart(2, '0')}`,
          short_title: `Member ${i}`,
          long_title: `Member ${i} detail`,
          entry_type: 'decision',
        },
        x: 20,
        y: 70,
      }
    })
    const hubA: ExplorerNode = { id: 'tag:hub-a', type: 'tag', label: 'hub-a', val: 15, properties: {} }
    const hubB: ExplorerNode = { id: 'tag:hub-b', type: 'tag', label: 'hub-b', val: 15, properties: {} }
    const nodeById = nodeMap([hubA, hubB, ...members])
    const memberConns = members.map((m) => conn(m.id, 'has_tag', 'in'))
    const conns: Record<string, ConnInput[]> = {
      'tag:hub-a': [...memberConns, conn('tag:hub-b', 'related_to', 'out')],
      'tag:hub-b': memberConns,
    }
    renderMassHarness(conns, nodeById, new Map(), hubA)
    fireEvent.click(screen.getByRole('button', { name: 'has_tag:in ← 15' }))
    fireEvent.click(screen.getByRole('button', { name: `+${16 - FACET_TOP_N}` }))
    expect(screen.getByRole('button', { name: 'p15 1' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'p01 1' }))
    const input = screen.getByTestId(`group-filter-${KEY}`) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'member' } })
    expect(input.value).toBe('member')
    expect(screen.getByRole('button', { name: 'p01 1' })).toHaveAttribute('aria-pressed', 'true')
    // walk to hub-b through the same select path a member click uses
    fireEvent.click(screen.getByRole('button', { name: 'hub-b' }))
    // hub-b's group renders collapsed again after the reset: expand it
    fireEvent.click(screen.getByRole('button', { name: 'has_tag:in ← 15' }))
    const inputB = screen.getByTestId(`group-filter-${KEY}`) as HTMLInputElement
    expect(inputB.value).toBe('')
    expect(screen.queryByText(/^\d+ of \d+$/)).toBeNull()
    for (const c of chipButtons()) expect(c).toHaveAttribute('aria-pressed', 'false')
    // the reveal reset: the +8 chip is back and only the top 8 facets show
    expect(screen.getByRole('button', { name: `+${16 - FACET_TOP_N}` })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'p15 1' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'p08 1' })).toBeNull()
    expect(screen.getByRole('button', { name: 'p07 1' })).toBeInTheDocument()
  })
})
