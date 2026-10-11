import { useCallback, useEffect, useRef, useState } from 'react'

export interface PageResponse<T> {
  items: T[]
  next_cursor: string | null
  total: number
  skipped: number
}

export type PagedStatus = 'loading' | 'ready' | 'error'

export interface PagedList<T> {
  items: T[]
  total: number
  skipped: number
  nextCursor: string | null
  status: PagedStatus
  error: unknown | null
  errorMessage: string | null
  loadingMore: boolean
  fetchedAt: number | null
  loadMore: () => void
  retry: () => void
}

/** Internal state, tagged with the token (key + retry nonce) it belongs to. */
interface TaggedState<T> {
  token: string | null
  items: T[]
  total: number
  skipped: number
  nextCursor: string | null
  status: PagedStatus
  error: unknown | null
  loadingMore: boolean
  fetchedAt: number | null
}

const EMPTY: TaggedState<never> = {
  token: null,
  items: [],
  total: 0,
  skipped: 0,
  nextCursor: null,
  status: 'loading',
  error: null,
  loadingMore: false,
  fetchedAt: null,
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/**
 * Cursor-paged fetching keyed by `key`.
 *
 * A new key (or a retry) discards the loaded items and fetches the first
 * page; loading is derived from the stored state's token not matching the
 * current one, so no state is reset synchronously in an effect. Responses
 * for a superseded key or request are ignored. `loadMore` appends the next
 * page, dropping items whose `keyOf` value is already present.
 */
export function usePagedList<T>(
  key: string,
  fetchPage: (cursor: string | null) => Promise<PageResponse<T>>,
  keyOf: (item: T) => string,
): PagedList<T> {
  const [nonce, setNonce] = useState(0)
  const [state, setState] = useState<TaggedState<T>>(EMPTY)
  const token = `${nonce}:${key}`

  // Latest fetchPage, updated after render so effects and handlers use it.
  const fetchRef = useRef(fetchPage)
  useEffect(() => {
    fetchRef.current = fetchPage
  })

  // Monotonic request id: only the latest request may write state.
  const requestRef = useRef(0)

  useEffect(() => {
    const id = ++requestRef.current
    fetchRef
      .current(null)
      .then((page) => {
        if (requestRef.current !== id) return
        setState({
          token,
          items: page.items,
          total: page.total,
          skipped: page.skipped,
          nextCursor: page.next_cursor,
          status: 'ready',
          error: null,
          loadingMore: false,
          fetchedAt: Date.now(),
        })
      })
      .catch((err: unknown) => {
        if (requestRef.current !== id) return
        setState({ ...EMPTY, token, status: 'error', error: err })
      })
    return () => {
      // Invalidate this request when the key changes or on unmount.
      if (requestRef.current === id) requestRef.current += 1
    }
  }, [token])

  const current = state.token === token
  const cursor = current ? state.nextCursor : null
  const busy = current && state.loadingMore

  const loadMore = useCallback(() => {
    if (cursor === null || busy) return
    const id = ++requestRef.current
    setState((s) => (s.token === token ? { ...s, loadingMore: true } : s))
    fetchRef
      .current(cursor)
      .then((page) => {
        if (requestRef.current !== id) return
        setState((s) => {
          if (s.token !== token) return s
          const seen = new Set(s.items.map(keyOf))
          const fresh = page.items.filter((item) => !seen.has(keyOf(item)))
          return {
            ...s,
            items: [...s.items, ...fresh],
            total: page.total,
            skipped: page.skipped,
            nextCursor: page.next_cursor,
            status: 'ready',
            error: null,
            loadingMore: false,
            fetchedAt: Date.now(),
          }
        })
      })
      .catch((err: unknown) => {
        if (requestRef.current !== id) return
        setState((s) =>
          s.token === token
            ? { ...s, status: 'error', error: err, loadingMore: false }
            : s,
        )
      })
  }, [cursor, busy, token, keyOf])

  const retry = useCallback(() => {
    setNonce((n) => n + 1)
  }, [])

  const view = current ? state : EMPTY
  return {
    items: view.items,
    total: view.total,
    skipped: view.skipped,
    nextCursor: view.nextCursor,
    status: view.status,
    error: view.error,
    errorMessage: view.status === 'error' ? messageOf(view.error) : null,
    loadingMore: view.loadingMore,
    fetchedAt: view.fetchedAt,
    loadMore,
    retry,
  }
}
