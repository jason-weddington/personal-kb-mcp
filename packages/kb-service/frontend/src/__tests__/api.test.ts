import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  ApiError,
  api,
  browseProject,
  browseRecent,
  getEntries,
} from '../api'

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

describe('KB browse + get request shapes', () => {
  beforeEach(() => {
    localStorage.setItem('kb-token', 'test-token')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        status: 200,
        ok: true,
        json: async () => ({}),
      } as Response),
    )
  })

  afterEach(() => {
    localStorage.clear()
    vi.unstubAllGlobals()
  })

  function lastUrl(): string {
    return vi.mocked(fetch).mock.calls.at(-1)![0] as string
  }

  function lastBody(): string {
    return (vi.mocked(fetch).mock.calls.at(-1)![1] as RequestInit).body as string
  }

  it('getEntries with touch false sends touch: false', async () => {
    await getEntries(['kb-1'], { touch: false })
    expect(lastBody()).toBe('{"ids":["kb-1"],"touch":false}')
  })

  it('getEntries without opts sends ids only', async () => {
    await getEntries(['kb-1'])
    expect(lastBody()).toBe('{"ids":["kb-1"]}')
  })

  it('browseProject encodes the ref and orders params', async () => {
    await browseProject('p alpha', { sort: 'newest', q: 'x', cursor: '25' })
    expect(lastUrl()).toBe('/api/kb/browse/project/p%20alpha?sort=newest&q=x&cursor=25')
  })

  it('browseProject encodes a slash in the ref', async () => {
    await browseProject('team/sub', { sort: 'newest' })
    expect(lastUrl()).toBe('/api/kb/browse/project/team%2Fsub?sort=newest')
  })

  it('browseRecent skips a null cursor', async () => {
    await browseRecent({ sort: 'created', entryType: 'decision', limit: 25, cursor: null })
    expect(lastUrl()).toBe('/api/kb/browse/recent?sort=created&entry_type=decision&limit=25')
  })
})
