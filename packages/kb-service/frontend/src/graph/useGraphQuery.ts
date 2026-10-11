import { useCallback, useEffect, useLayoutEffect, useMemo, useReducer, useRef } from 'react'
import type { RefObject } from 'react'
import { getToken } from '../api'
import type { GraphExplorerHandle } from '../components/GraphExplorer'
import { streamSSE } from '../streaming'
import type { TraversalState } from './explorerModel'
import {
  REVEAL_DELAY_MS,
  initialQueryState,
  queryReducer,
  stepEffect,
  summarizeQuery,
  toTraversal,
} from './queryModel'
import type { QueryState, QuerySummary } from './queryModel'

export interface GraphIds {
  rendered: ReadonlySet<string>
  all: ReadonlySet<string>
}

export function useGraphQuery(
  explorerRef: RefObject<GraphExplorerHandle | null>,
  graphIds: GraphIds,
): {
  state: QueryState
  traversal: TraversalState
  start: (question: string) => void
  close: () => void
  reveal: (ids: string[]) => void
} {
  const [state, dispatch] = useReducer(queryReducer, initialQueryState)
  const controllerRef = useRef<AbortController | null>(null)
  const runIdRef = useRef(0)
  const loggedRunRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingLenRef = useRef(0)
  const stateRef = useRef(state)
  const graphIdsRef = useRef(graphIds)

  useLayoutEffect(() => {
    pendingLenRef.current = state.pending.length
    stateRef.current = state
    graphIdsRef.current = graphIds
  })

  const traversal = useMemo(
    () => toTraversal(state.queryMode, state.nodeStates, state.pulseStarts),
    [state.queryMode, state.nodeStates, state.pulseStarts],
  )

  const emitSummary = useCallback(
    (outcome: QuerySummary['outcome'], abortReason: QuerySummary['abort_reason']) => {
      if (loggedRunRef.current === runIdRef.current) return
      loggedRunRef.current = runIdRef.current
      const s = summarizeQuery(stateRef.current, {
        outcome,
        abortReason,
        renderedIds: graphIdsRef.current.rendered,
        graphIds: graphIdsRef.current.all,
      })
      console.info('[graph-query]', s)
      for (const event of s.unknown_events) {
        console.warn('[graph-query] unhandled event', { event })
      }
      for (const [key, count] of Object.entries(s.malformed)) {
        console.warn('[graph-query] malformed event', { key, count })
      }
      if (s.marked_nodes > 0 && s.marked_rendered === 0) {
        console.warn('[graph-query] no marked node is in the graph', {
          marked_nodes: s.marked_nodes,
          sample: s.not_in_graph_sample,
        })
      }
    },
    [],
  )

  const start = useCallback(
    (question: string) => {
      const q = question.trim()
      if (q === '') return
      if (stateRef.current.phase !== 'idle') emitSummary('aborted', 'superseded')
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      runIdRef.current += 1
      const runId = runIdRef.current
      explorerRef.current?.clearSelection()
      dispatch({ type: 'start', question: q })
      const token = getToken()
      const url = `/api/kb/query/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`
      streamSSE(
        url,
        { question: q },
        (event, data) => {
          if (runIdRef.current === runId) dispatch({ type: 'event', event, data })
        },
        controller.signal,
      ).then(
        () => {
          if (runIdRef.current === runId) dispatch({ type: 'streamClosed' })
        },
        (err: unknown) => {
          if (runIdRef.current !== runId || controller.signal.aborted) return
          dispatch({
            type: 'fail',
            message: err instanceof Error ? err.message : String(err),
          })
        },
      )
    },
    [emitSummary, explorerRef],
  )

  const close = useCallback(() => {
    if (stateRef.current.phase !== 'idle') emitSummary('aborted', 'close')
    controllerRef.current?.abort()
    runIdRef.current += 1
    dispatch({ type: 'close' })
    explorerRef.current?.resetAll()
  }, [emitSummary, explorerRef])

  const reveal = useCallback((ids: string[]) => dispatch({ type: 'reveal', ids }), [])

  // Drain: apply one queued reveal step per REVEAL_DELAY_MS.
  useEffect(() => {
    if (state.pending.length === 0 || timerRef.current !== null) return
    function run() {
      if (pendingLenRef.current === 0) {
        timerRef.current = null
        return
      }
      dispatch({ type: 'tick', now: performance.now() })
      timerRef.current = setTimeout(run, REVEAL_DELAY_MS)
    }
    timerRef.current = setTimeout(run, 0)
  }, [state.pending])

  // Camera and particles follow the step that was just applied.
  useEffect(() => {
    if (state.lastApplied === null) return
    const fx = stepEffect(state.lastApplied.step, state.nodeStates)
    if (fx.particlesFor) explorerRef.current?.emitParticlesFor(fx.particlesFor)
    if (fx.fit) explorerRef.current?.fitNodes(fx.fit.ids, fx.fit.durationMs, fx.fit.padding)
  }, [state.lastApplied, state.nodeStates, explorerRef])

  useEffect(() => {
    if (
      (state.phase === 'done' || state.phase === 'error') &&
      state.streamEnded &&
      state.pending.length === 0
    ) {
      emitSummary(state.phase, null)
    }
  }, [state.phase, state.streamEnded, state.pending, emitSummary])

  useEffect(
    () => () => {
      if (stateRef.current.phase !== 'idle') emitSummary('aborted', 'unmount')
      controllerRef.current?.abort()
      if (timerRef.current !== null) clearTimeout(timerRef.current)
      timerRef.current = null
    },
    [emitSummary],
  )

  return { state, traversal, start, close, reveal }
}
