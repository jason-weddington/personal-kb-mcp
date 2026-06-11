import { createContext, useContext, useState, useCallback } from 'react'
import type { ReactNode } from 'react'
import { EntryDrawer } from '../components/EntryDrawer'

interface EntryDrawerContextValue {
  openEntry: (id: string) => void
  closeEntry: () => void
}

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

  const popStack = useCallback(() => {
    setStack((s) => s.slice(0, -1))
  }, [])

  return (
    <EntryDrawerContext.Provider value={{ openEntry, closeEntry }}>
      {children}
      <EntryDrawer
        stack={stack}
        onBack={popStack}
        onClose={closeEntry}
        onOpenEntry={openEntry}
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
