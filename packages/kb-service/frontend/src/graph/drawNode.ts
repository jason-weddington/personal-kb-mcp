// Canvas node renderer, a line-for-line port of the old explorer.js
// nodeCanvasObject (personal_kb 47afec7, lines 203-263).
import {
  BACKGROUND,
  DIMMED_FILL,
  DIM_ALPHA,
  GLOW_HIGHLIGHT,
  GLOW_RESULT,
  GLOW_VISITED,
  HIGHLIGHT_EXTRA_R,
  LABEL_MIN_OPACITY,
  RESULT_COLOR,
  RESULT_EXTRA_R,
  VISITED_COLOR,
  labelOpacity,
  nodeColor,
  nodeRadius,
  pulseRadius,
  truncateLabel,
} from './explorerModel'
import type { ExplorerNode, TraversalState } from './explorerModel'

export type NodeRenderState = {
  hoveredId: string | null
  highlightedId: string | null
  focusSet: Set<string> | null
  soloType: string | null
  traversal: TraversalState
  activeNeighbors: Set<string>
  /** performance.now() of the current frame, written in onRenderFramePre. */
  now: number
}

export function drawNode(
  node: ExplorerNode,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  rs: NodeRenderState,
): void {
  const { traversal } = rs
  const id = node.id
  const x = node.x ?? 0
  const y = node.y ?? 0
  const nState = traversal.nodeStates.get(id)
  const isResult = nState === 'result'
  const isVisited = nState === 'visited'
  const isHL = id === rs.highlightedId
  const active = isResult || isVisited || isHL
  const queryOn = traversal.queryMode !== null
  const neighbor = queryOn && !active && rs.activeNeighbors.has(id)
  const dimmed = queryOn && !active && !neighbor

  const extraR = isResult ? RESULT_EXTRA_R : isHL ? HIGHLIGHT_EXTRA_R : 0
  const r = nodeRadius(node.val, extraR, pulseRadius(traversal.pulseStarts.get(id), rs.now))
  const color = isResult ? RESULT_COLOR : isVisited ? VISITED_COLOR : nodeColor(node.type)

  if (active) {
    ctx.shadowColor = color
    ctx.shadowBlur = isResult ? GLOW_RESULT : isVisited ? GLOW_VISITED : GLOW_HIGHLIGHT
  }
  if (dimmed) ctx.globalAlpha = DIM_ALPHA

  ctx.beginPath()
  ctx.arc(x, y, r, 0, 2 * Math.PI)
  ctx.fillStyle = dimmed ? DIMMED_FILL : color
  ctx.fill()
  ctx.shadowBlur = 0

  if (dimmed) {
    ctx.globalAlpha = 1
    return
  }

  const opacity = labelOpacity({
    hovered: id === rs.hoveredId,
    highlighted: isHL,
    nodeState: nState,
    queryMode: traversal.queryMode,
    inFocus: rs.focusSet?.has(id) ?? false,
    solo: rs.soloType !== null,
    globalScale,
  })
  if (opacity < LABEL_MIN_OPACITY) return

  const fontSize = Math.max(10 / globalScale, 2)
  ctx.font = `${fontSize}px system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  const text = truncateLabel(node.label || id, globalScale)
  const textW = ctx.measureText(text).width
  const padX = 3 / globalScale
  const padY = 1.5 / globalScale
  const pillY = y + r + 2

  ctx.globalAlpha = opacity * 0.6
  ctx.fillStyle = BACKGROUND
  ctx.beginPath()
  ctx.roundRect(x - textW / 2 - padX, pillY - padY, textW + 2 * padX, fontSize + 2 * padY, 3 / globalScale)
  ctx.fill()

  ctx.globalAlpha = opacity
  ctx.fillStyle = '#ffffff'
  ctx.fillText(text, x, pillY)
  ctx.globalAlpha = 1
}
