import { createContext, useContext, useState, useCallback, useMemo, useRef } from 'react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { EntryDrawer } from '../components/EntryDrawer'

interface EntryDrawerContextValue {
  openEntry: (id: string) => void
  closeEntry: () => void
  /**
   * Open a connection chip's node. A page that registered a node handler
   * (the Graph page) gets every id; otherwise an entry id opens in the drawer
   * and any other node opens the Graph focused on it.
   */
  openNode: (id: string) => void
  /** Register (or clear with null) the page's node handler. */
  setNodeHandler: (fn: ((id: string) => void) | null) => void
}

const ENTRY_ID = /^kb-\d+$/

const EntryDrawerContext = createContext<EntryDrawerContextValue | null>(null)

export function EntryDrawerProvider({ children }: { children: ReactNode }) {
  const [stack, setStack] = useState<string[]>([])

  const openEntry = useCallback((id: string) => {
    setStack((s) => {
      if (s.length === 0) return [id]
      if (id === s[s.length - 1]) return s // no-op when id === top of stack
      return [...s, id]
    })
  }, [])

  const closeEntry = useCallback(() => {
    setStack([])
  }, [])

  const navigate = useNavigate()
  const nodeHandlerRef = useRef<((id: string) => void) | null>(null)

  const setNodeHandler = useCallback((fn: ((id: string) => void) | null) => {
    nodeHandlerRef.current = fn
  }, [])

  const openNode = useCallback(
    (id: string) => {
      if (nodeHandlerRef.current) {
        nodeHandlerRef.current(id)
      } else if (ENTRY_ID.test(id)) {
        openEntry(id)
      } else {
        navigate(`/graph?focus=${encodeURIComponent(id)}`)
        setStack([])
      }
    },
    [openEntry, navigate],
  )

  const value = useMemo(
    () => ({ openEntry, closeEntry, openNode, setNodeHandler }),
    [openEntry, closeEntry, openNode, setNodeHandler],
  )

  const popStack = useCallback(() => {
    setStack((s) => s.slice(0, -1))
  }, [])

  return (
    <EntryDrawerContext.Provider value={value}>
      {children}
      <EntryDrawer
        stack={stack}
        onBack={popStack}
        onClose={closeEntry}
        onOpenEntry={openEntry}
        onOpenNode={openNode}
      />
    </EntryDrawerContext.Provider>
  )
}

export function useEntryDrawer(): EntryDrawerContextValue {
  const ctx = useContext(EntryDrawerContext)
  if (!ctx) {
    throw new Error('useEntryDrawer must be used within EntryDrawerProvider')
  }
  return ctx
}
