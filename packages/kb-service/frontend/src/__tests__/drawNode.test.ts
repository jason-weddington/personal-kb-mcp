import { describe, it, expect } from 'vitest'
import { drawNode } from '../graph/drawNode'
import type { NodeRenderState } from '../graph/drawNode'
import { EMPTY_TRAVERSAL } from '../graph/explorerModel'
import type { ExplorerNode, TraversalState } from '../graph/explorerModel'

type Rec = [string, ...unknown[]]

function recordingCtx() {
  const log: Rec[] = []
  const target: Record<string, unknown> = {
    beginPath: () => log.push(['beginPath']),
    arc: (...a: unknown[]) => log.push(['arc', ...a]),
    roundRect: (...a: unknown[]) => log.push(['roundRect', ...a]),
    fill: () => log.push(['fill']),
    fillText: (...a: unknown[]) => log.push(['fillText', ...a]),
    measureText: () => ({ width: 40 }),
  }
  const ctx = new Proxy(target, {
    set(t, prop, value) {
      log.push(['set', String(prop), value])
      t[String(prop)] = value
      return true
    },
  })
  return { ctx: ctx as unknown as CanvasRenderingContext2D, log, state: target }
}

function rs(traversal: TraversalState = EMPTY_TRAVERSAL): NodeRenderState {
  return {
    hoveredId: null,
    highlightedId: null,
    focusSet: null,
    soloType: null,
    traversal,
    activeNeighbors: new Set(),
    now: 0,
  }
}

const sets = (log: Rec[], prop: string) =>
  log.filter((r) => r[0] === 'set' && r[1] === prop).map((r) => r[2])

const nodeA: ExplorerNode = { id: 'a', x: 0, y: 0, val: 4, type: 'entry', label: 'abc', properties: {} }

describe('drawNode', () => {
  it('draws an idle node with a label pill', () => {
    const { ctx, log, state } = recordingCtx()
    drawNode(nodeA, ctx, 4, rs())
    expect(log).toContainEqual(['arc', 0, 0, 5, 0, 2 * Math.PI])
    expect(sets(log, 'fillStyle')[0]).toBe('#e0e0e0')
    expect(sets(log, 'font')).toEqual(['2.5px system-ui, sans-serif'])
    expect(sets(log, 'globalAlpha')).toEqual([0.3, 0.5, 1])
    expect(log).toContainEqual(['roundRect', -20.75, 6.625, 41.5, 3.25, 0.75])
    expect(log).toContainEqual(['fillText', 'abc', 0, 7])
    expect(state.globalAlpha).toBe(1)
  })

  const traversal: TraversalState = {
    queryMode: 'explore',
    nodeStates: new Map([['b', 'result']]),
    pulseStarts: new Map(),
  }

  it('dims an inactive node in query mode with no label', () => {
    const { ctx, log } = recordingCtx()
    drawNode(nodeA, ctx, 4, rs(traversal))
    expect(sets(log, 'globalAlpha')).toEqual([0.15, 1])
    expect(sets(log, 'fillStyle')).toEqual(['#555'])
    expect(log.some((r) => r[0] === 'fillText')).toBe(false)
  })

  it('glows a result node', () => {
    const { ctx, log } = recordingCtx()
    drawNode({ ...nodeA, id: 'b' }, ctx, 4, rs(traversal))
    expect(log).toContainEqual(['arc', 0, 0, 8, 0, 2 * Math.PI])
    expect(sets(log, 'shadowColor')).toEqual(['#00ff88'])
    expect(sets(log, 'shadowBlur')).toEqual([30, 0])
    expect(log.some((r) => r[0] === 'fillText')).toBe(true)
  })
})
