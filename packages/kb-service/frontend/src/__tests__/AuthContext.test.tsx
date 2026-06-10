import { useEffect } from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, waitFor } from '@testing-library/react'
import { AuthProvider, useAuth } from '../contexts/AuthContext'

// Mock the api module
vi.mock('../api', () => ({
  api: {
    auth: {
      me: vi.fn(),
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    },
  },
  ApiError: class ApiError extends Error {
    status: number
    detail: string
    constructor(status: number, detail: string) {
      super(detail)
      this.status = status
      this.detail = detail
    }
  },
}))

import { api } from '../api'

type CapturedState = { user: unknown; loading: boolean }

// TestConsumer captures auth state via a ref each render.
// Using useEffect so we don't mutate the external ref directly in render
// (avoids react-hooks/immutability lint error).
function TestConsumer({
  stateRef,
}: {
  stateRef: React.MutableRefObject<CapturedState | null>
}) {
  const { user, loading } = useAuth()
  useEffect(() => {
    stateRef.current = { user, loading }
  })
  return null
}

describe('AuthContext bootstrap (kb-01449 #2)', () => {
  const originalLocation = window.location

  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      writable: true,
      configurable: true,
    })
  })

  it('when api.auth.me rejects: user is null, loading is false, kb-token removed, no redirect', async () => {
    localStorage.setItem('kb-token', 'bad-token')
    localStorage.setItem('kb-user', JSON.stringify({ id: '1', email: 'a@b.com' }))

    const locationStub = { pathname: '/some-page', href: '' }
    Object.defineProperty(window, 'location', {
      value: locationStub,
      writable: true,
      configurable: true,
    })

    vi.mocked(api.auth.me).mockRejectedValue(new Error('network error'))

    // Use a plain object ref — TypeScript doesn't narrow object properties
    // the same way it narrows let variables, avoiding the 'never' issue.
    const stateRef = { current: null as CapturedState | null }

    render(
      <AuthProvider>
        <TestConsumer stateRef={stateRef as React.MutableRefObject<CapturedState | null>} />
      </AuthProvider>,
    )

    await waitFor(() => {
      expect(stateRef.current?.loading).toBe(false)
    })

    expect(stateRef.current?.user).toBeNull()
    expect(localStorage.getItem('kb-token')).toBeNull()
    expect(localStorage.getItem('kb-user')).toBeNull()
    // kb-01449 guard #2: no redirect on bootstrap failure
    expect(locationStub.href).toBe('')
  })
})
