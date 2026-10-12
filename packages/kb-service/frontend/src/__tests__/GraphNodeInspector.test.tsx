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
import { INSPECTOR_PILL_MAX, groupNeighbors } from '../graph/inspectorModel'
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
