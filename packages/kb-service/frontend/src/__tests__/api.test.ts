import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { ApiError, api } from '../api'

describe('api 401 guard (kb-01449 #1)', () => {
  const originalLocation = window.location

  beforeEach(() => {
    localStorage.setItem('kb-token', 'test-token')
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    localStorage.clear()
    vi.unstubAllGlobals()
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      writable: true,
      configurable: true,
    })
  })

  function mockFetch401() {
    vi.mocked(fetch).mockResolvedValue({
      status: 401,
      ok: false,
      statusText: 'Unauthorized',
      json: async () => ({ detail: 'Unauthorized' }),
    } as Response)
  }

  it('removes kb-token and kb-user on 401', async () => {
    localStorage.setItem('kb-user', JSON.stringify({ id: '1', email: 'a@b.com' }))
    Object.defineProperty(window, 'location', {
      value: { pathname: '/', href: '' },
      writable: true,
      configurable: true,
    })
    mockFetch401()

    await expect(api.auth.me()).rejects.toThrow(ApiError)
    expect(localStorage.getItem('kb-token')).toBeNull()
    expect(localStorage.getItem('kb-user')).toBeNull()
  })

  it('redirects to /login when pathname is not /login', async () => {
    const locationStub = { pathname: '/', href: '' }
    Object.defineProperty(window, 'location', {
      value: locationStub,
      writable: true,
      configurable: true,
    })
    mockFetch401()

    await expect(api.auth.me()).rejects.toThrow(ApiError)
    expect(locationStub.href).toBe('/login')
  })

  it('does NOT redirect when pathname is /login', async () => {
    const locationStub = { pathname: '/login', href: '' }
    Object.defineProperty(window, 'location', {
      value: locationStub,
      writable: true,
      configurable: true,
    })
    mockFetch401()

    await expect(api.auth.me()).rejects.toThrow(ApiError)
    expect(locationStub.href).toBe('')
  })

  it('throws ApiError with status 401 in both cases', async () => {
    Object.defineProperty(window, 'location', {
      value: { pathname: '/', href: '' },
      writable: true,
      configurable: true,
    })
    mockFetch401()

    try {
      await api.auth.me()
      expect.fail('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(401)
    }
  })
})
