import { useEffect, useState } from 'react'

/**
 * Returns `value` after it has stopped changing for `delayMs`. The initial
 * value is returned immediately; each change restarts the timer.
 */
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])
  return debounced
}
