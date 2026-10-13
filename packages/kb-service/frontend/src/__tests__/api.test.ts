import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  ApiError,
  api,
  browseProject,
  browseRecent,
  getEntries,
  ingestFile,
  ingestUrl,
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

describe('ingest helpers', () => {
  const originalLocation = window.location

  const mdFile = () => new File(['content'], 'a.md', { type: 'text/markdown' })

  const okResponse = {
    status: 200,
    ok: true,
    json: async () => ({}),
  } as Response

  function errorResponse(status: number, statusText: string, detail?: string) {
    return {
      status,
      ok: false,
      statusText,
      json: async () => (detail === undefined ? {} : { detail }),
    } as Response
  }

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

  function lastCall(): [string, RequestInit] {
    return vi.mocked(fetch).mock.calls.at(-1) as [string, RequestInit]
  }

  // ── ingestFile ────────────────────────────────────────────────────────────

  it('posts FormData as the fetch body with no manual Content-Type', async () => {
    vi.mocked(fetch).mockResolvedValue(okResponse)
    await ingestFile(mdFile())
    const [url, init] = lastCall()
    expect(url).toBe('/api/kb/ingest/file')
    expect(init.method).toBe('POST')
    expect(init.headers).toEqual({ Authorization: 'Bearer test-token' })
    expect(init.body).toBeInstanceOf(FormData)
    const form = init.body as FormData
    expect((form.get('file') as File).name).toBe('a.md')
    expect(form.has('project_ref')).toBe(false)
    expect(form.has('dry_run')).toBe(false)
  })

  it('appends project_ref only for a non-empty string', async () => {
    vi.mocked(fetch).mockResolvedValue(okResponse)
    await ingestFile(mdFile(), 'proj-a')
    let form = lastCall()[1].body as FormData
    expect(form.get('project_ref')).toBe('proj-a')

    await ingestFile(mdFile(), '')
    form = lastCall()[1].body as FormData
    expect(form.has('project_ref')).toBe(false)
  })

  it('sends no Authorization header when no token is stored', async () => {
    localStorage.removeItem('kb-token')
    vi.mocked(fetch).mockResolvedValue(okResponse)
    await ingestFile(mdFile())
    expect(lastCall()[1].headers).toEqual({})
  })

  it('401 removes the token, redirects and rejects ApiError(401)', async () => {
    const locationStub = { pathname: '/', href: '' }
    Object.defineProperty(window, 'location', {
      value: locationStub,
      writable: true,
      configurable: true,
    })
    localStorage.setItem('kb-user', JSON.stringify({ id: '1', email: 'a@b.com' }))
    vi.mocked(fetch).mockResolvedValue(errorResponse(401, 'Unauthorized', 'Unauthorized'))

    await expect(ingestFile(mdFile())).rejects.toThrow(ApiError)
    expect(localStorage.getItem('kb-token')).toBeNull()
    expect(localStorage.getItem('kb-user')).toBeNull()
    expect(locationStub.href).toBe('/login')
    try {
      await ingestUrl('https://x.example')
      expect.fail('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(401)
      expect((err as ApiError).detail).toBe('Unauthorized')
    }
  })

  it('maps a 403 detail through both helpers', async () => {
    vi.mocked(fetch).mockResolvedValue(
      errorResponse(403, 'Forbidden', 'Interactive surface required.'),
    )
    await expect(ingestFile(mdFile())).rejects.toMatchObject({
      status: 403,
      detail: 'Interactive surface required.',
    })
    await expect(ingestUrl('https://x.example')).rejects.toMatchObject({
      status: 403,
      detail: 'Interactive surface required.',
    })
  })

  it('maps a 413 detail (size cap) through both helpers', async () => {
    const detail =
      'Upload exceeds maximum file size of 10485760 bytes. Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads.'
    vi.mocked(fetch).mockResolvedValue(errorResponse(413, 'Payload Too Large', detail))
    await expect(ingestFile(mdFile())).rejects.toMatchObject({ status: 413, detail })
    await expect(ingestUrl('https://x.example')).rejects.toMatchObject({
      status: 413,
      detail,
    })
  })

  it('maps a 422 detail (empty filename) through both helpers', async () => {
    vi.mocked(fetch).mockResolvedValue(
      errorResponse(422, 'Unprocessable Entity', 'A non-empty filename is required.'),
    )
    await expect(ingestFile(mdFile())).rejects.toMatchObject({
      status: 422,
      detail: 'A non-empty filename is required.',
    })
    await expect(ingestUrl('https://x.example')).rejects.toMatchObject({
      status: 422,
      detail: 'A non-empty filename is required.',
    })
  })

  it('falls back to statusText when the body has no detail', async () => {
    vi.mocked(fetch).mockResolvedValue(errorResponse(503, 'Service Unavailable'))
    await expect(ingestFile(mdFile())).rejects.toMatchObject({
      status: 503,
      detail: 'Service Unavailable',
    })
  })

  // ── ingestUrl ─────────────────────────────────────────────────────────────

  it('posts {"url":...} JSON with Content-Type application/json and no dry_run', async () => {
    vi.mocked(fetch).mockResolvedValue(okResponse)
    await ingestUrl('https://x.example')
    const [url, init] = lastCall()
    expect(url).toBe('/api/kb/ingest/url')
    expect(init.method).toBe('POST')
    expect(init.body).toBe('{"url":"https://x.example"}')
    expect((init.headers as Record<string, string>)['Content-Type']).toBe(
      'application/json',
    )
    expect((init.headers as Record<string, string>).Authorization).toBe(
      'Bearer test-token',
    )
    expect(JSON.parse(init.body as string)).not.toHaveProperty('dry_run')
  })

  it('adds project_ref to the JSON body only for a non-empty string', async () => {
    vi.mocked(fetch).mockResolvedValue(okResponse)
    await ingestUrl('https://x.example', 'proj-a')
    expect(lastCall()[1].body).toBe('{"url":"https://x.example","project_ref":"proj-a"}')

    await ingestUrl('https://x.example', '')
    expect(lastCall()[1].body).toBe('{"url":"https://x.example"}')
  })
})
