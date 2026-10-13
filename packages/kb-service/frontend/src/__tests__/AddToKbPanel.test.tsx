/**
 * AddToKbPanel tests:
 * - Drop zone: accepted vs rejected file types and sizes, dropped URLs
 *   (uri-list precedence, multi-line, files+URL at once), ignored drops
 * - Browse picker: same classifyFile path, input reset
 * - URL field: Enter and button submit, empty-field disabled
 * - Project field: suggestions from listKbProjects, project_ref pass-through
 * - Queue: one at a time in submission order, continues past failures
 * - Results: action alerts, entry links via the drawer, thrown errors
 * - Structured logs: [add-to-kb] enqueue/reject/done/refresh/error/…
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import type { MockInstance } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { IngestFileResult } from '../kbTypes'

const openEntry = vi.hoisted(() => vi.fn())

const MockApiError = vi.hoisted(
  () =>
    class ApiError extends Error {
      status: number
      detail: string
      constructor(status: number, detail: string) {
        super(detail)
        this.status = status
        this.detail = detail
        this.name = 'ApiError'
      }
    },
)

vi.mock('../api', () => ({
  ApiError: MockApiError,
  getToken: vi.fn(() => 'test-token'),
  listKbProjects: vi.fn(),
  ingestFile: vi.fn(),
  ingestUrl: vi.fn(),
}))

vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: () => ({ openEntry, closeEntry: vi.fn(), openNode: vi.fn(), setNodeHandler: vi.fn() }),
}))

import { ApiError, ingestFile, ingestUrl, listKbProjects } from '../api'
import { MAX_FILE_SIZE_BYTES } from '../components/addToKbModel'
import { AddToKbPanel } from '../components/AddToKbPanel'

// ── Helpers ──────────────────────────────────────────────────────────────────

const result = (over: Partial<IngestFileResult> = {}): IngestFileResult => ({
  path: 'a.md',
  action: 'ingested',
  reason: null,
  entry_count: 1,
  entry_ids: ['kb-00001'],
  summary: null,
  chunks_processed: 1,
  chunks_skipped: 0,
  chunks_flagged: 0,
  ...over,
})

function fileOf(name: string, size = 3): File {
  const f = new File(['abc'], name, { type: 'text/plain' })
  Object.defineProperty(f, 'size', { value: size })
  return f
}

function drop(el: HTMLElement, dataTransfer: { files?: File[]; getData?: (t: string) => string; items?: Array<{ type: string }> }) {
  fireEvent.drop(el, {
    dataTransfer: {
      files: [],
      getData: () => '',
      ...dataTransfer,
    },
  })
}

function dropFiles(el: HTMLElement, files: File[]) {
  drop(el, { files })
}

function deferred<T>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function renderPanel(onIngested?: () => void) {
  return render(<AddToKbPanel onIngested={onIngested} />)
}

const dropZone = () => screen.getByTestId('add-to-kb-drop')
const fileInput = () => screen.getByTestId('add-to-kb-file-input') as HTMLInputElement

/** Document order of two elements: -1 when a comes first. */
function orderOf(a: HTMLElement, b: HTMLElement): number {
  return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('AddToKbPanel', () => {
  let info: MockInstance
  let warn: MockInstance
  let debug: MockInstance

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(listKbProjects).mockResolvedValue({ items: [] })
    info = vi.spyOn(console, 'info').mockImplementation(() => {})
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    debug = vi.spyOn(console, 'debug').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // ── Drop zone: file classification ──────────────────────────────────────

  it('queues an accepted dropped file and never sends a rejected one', async () => {
    vi.mocked(ingestFile).mockResolvedValue(result())
    renderPanel()
    const md = fileOf('notes.md')
    dropFiles(dropZone(), [md])
    await waitFor(() =>
      expect(screen.getByText('Ingested · 1 entries · 1 chunks')).toBeInTheDocument(),
    )
    expect(ingestFile).toHaveBeenCalledTimes(1)
    expect(ingestFile).toHaveBeenCalledWith(md, undefined)
    // The reject tripwire did not fire for an accepted file.
    expect(info).not.toHaveBeenCalledWith('[add-to-kb] reject', expect.anything())
    expect(screen.queryByText(/Rejected:/)).toBeNull()
  })

  it('rejects a dropped file of an unsupported type client-side', () => {
    renderPanel()
    const exe = fileOf('notes.exe')
    dropFiles(dropZone(), [exe])
    expect(screen.getByText('Rejected: notes.exe — Unsupported file type')).toBeInTheDocument()
    expect(ingestFile).not.toHaveBeenCalled()
    expect(info).toHaveBeenCalledWith('[add-to-kb] reject', {
      name: 'notes.exe',
      size: 3,
      kind: 'type',
      message: 'Unsupported file type',
    })
  })

  it('rejects a dropped file over the 10 MiB cap client-side', () => {
    renderPanel()
    dropFiles(dropZone(), [fileOf('big.md', MAX_FILE_SIZE_BYTES + 1)])
    expect(
      screen.getByText('Rejected: big.md — Exceeds the 10485760-byte limit'),
    ).toBeInTheDocument()
    expect(ingestFile).not.toHaveBeenCalled()
    expect(info).toHaveBeenCalledWith('[add-to-kb] reject', {
      name: 'big.md',
      size: 10485761,
      kind: 'size',
      message: 'Exceeds the 10485760-byte limit',
    })
  })

  it('mixed drop: accepted file queues, rejected file shows the reject line', () => {
    renderPanel()
    dropFiles(dropZone(), [fileOf('notes.docx')])
    expect(screen.getByText('Rejected: notes.docx — Unsupported file type')).toBeInTheDocument()
    expect(ingestFile).not.toHaveBeenCalled()
  })

  // ── Drop zone: URLs ─────────────────────────────────────────────────────

  it('a dropped uri-list becomes one job per http(s) line, in line order', async () => {
    renderPanel()
    vi.mocked(ingestUrl).mockResolvedValue(result({ action: 'skipped', reason: 'dup' }))
    drop(dropZone(), {
      getData: (t: string) =>
        t === 'text/uri-list' ? 'https://a.example/x\nhttps://b.example/y' : '',
    })
    await waitFor(() => expect(ingestUrl).toHaveBeenCalledTimes(2))
    expect(ingestUrl).toHaveBeenNthCalledWith(1, 'https://a.example/x', undefined)
    expect(ingestUrl).toHaveBeenNthCalledWith(2, 'https://b.example/y', undefined)
    expect(
      orderOf(screen.getByText('https://a.example/x'), screen.getByText('https://b.example/y')),
    ).toBe(-1)
  })

  it('falls back to text/plain when the uri-list is empty', async () => {
    renderPanel()
    vi.mocked(ingestUrl).mockResolvedValue(result())
    drop(dropZone(), {
      getData: (t: string) => (t === 'text/plain' ? 'https://plain.example' : ''),
    })
    await waitFor(() =>
      expect(ingestUrl).toHaveBeenCalledWith('https://plain.example', undefined),
    )
  })

  it('a drop carrying both files and URL text queues the files first, then the URLs', () => {
    renderPanel()
    const fileDefer = deferred<IngestFileResult>()
    const urlDefer = deferred<IngestFileResult>()
    vi.mocked(ingestFile).mockImplementation(() => fileDefer.promise)
    vi.mocked(ingestUrl).mockImplementation(() => urlDefer.promise)
    drop(dropZone(), {
      files: [fileOf('a.md')],
      getData: (t: string) => (t === 'text/uri-list' ? 'https://z.example' : ''),
    })
    expect(ingestFile).toHaveBeenCalledTimes(1)
    expect(ingestUrl).not.toHaveBeenCalled()
    expect(
      orderOf(screen.getByText('a.md'), screen.getByText('https://z.example')),
    ).toBe(-1)
  })

  it('ignores dropped text without an http(s) prefix and logs the drop kinds', () => {
    renderPanel()
    drop(dropZone(), {
      getData: (t: string) => (t === 'text/plain' ? 'just some words' : ''),
      items: [{ type: 'text/plain' }],
    })
    expect(ingestFile).not.toHaveBeenCalled()
    expect(ingestUrl).not.toHaveBeenCalled()
    expect(screen.queryByText(/Rejected:/)).toBeNull()
    expect(debug).toHaveBeenCalledWith('[add-to-kb] drop ignored', {
      kinds: ['text/plain'],
    })
  })

  // ── Browse picker ───────────────────────────────────────────────────────

  it('the Browse input change uses the same classifyFile path and resets the input', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockResolvedValue(result())
    const input = fileInput()
    fireEvent.change(input, { target: { files: [fileOf('notes.pdf')] } })
    await waitFor(() => expect(ingestFile).toHaveBeenCalledTimes(1))
    expect(input.value).toBe('')
    // The same file can be picked again immediately.
    fireEvent.change(input, { target: { files: [fileOf('notes.pdf')] } })
    await waitFor(() => expect(ingestFile).toHaveBeenCalledTimes(2))
  })

  it('a rejected picker selection shows the reject line and creates no job', () => {
    renderPanel()
    fireEvent.change(fileInput(), { target: { files: [fileOf('notes.docx')] } })
    expect(screen.getByText('Rejected: notes.docx — Unsupported file type')).toBeInTheDocument()
    expect(ingestFile).not.toHaveBeenCalled()
  })

  // ── URL field ───────────────────────────────────────────────────────────

  it('Enter in the URL field submits, clears the field and queues a job', async () => {
    renderPanel()
    vi.mocked(ingestUrl).mockResolvedValue(result())
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('URL'), 'https://typed.example')
    await user.keyboard('{Enter}')
    expect(ingestUrl).toHaveBeenCalledWith('https://typed.example', undefined)
    await waitFor(() => expect(screen.getByLabelText('URL')).toHaveValue(''))
    expect(screen.getByText('https://typed.example')).toBeInTheDocument()
  })

  it('the Ingest URL button submits and is disabled while the field is empty', async () => {
    renderPanel()
    vi.mocked(ingestUrl).mockResolvedValue(result())
    const user = userEvent.setup()
    const button = screen.getByRole('button', { name: 'Ingest URL' })
    expect(button).toBeDisabled()
    await user.type(screen.getByLabelText('URL'), 'https://clicked.example')
    expect(button).toBeEnabled()
    await user.click(button)
    expect(ingestUrl).toHaveBeenCalledWith('https://clicked.example', undefined)
  })

  // ── Project field ───────────────────────────────────────────────────────

  it('offers listKbProjects names as suggestions', async () => {
    vi.mocked(listKbProjects).mockResolvedValue({
      items: [
        { name: 'proj-a', entry_count: 3 },
        { name: 'proj-b', entry_count: 1 },
      ],
    })
    renderPanel()
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Project (optional)'), 'proj')
    expect(await screen.findByText('proj-a')).toBeInTheDocument()
    expect(screen.getByText('proj-b')).toBeInTheDocument()
  })

  it('a failed project fetch leaves the field free-text and warns', async () => {
    vi.mocked(listKbProjects).mockRejectedValue(new Error('boom'))
    renderPanel()
    await waitFor(() =>
      expect(warn).toHaveBeenCalledWith('[add-to-kb] project suggestions failed', {
        error: 'Error: boom',
      }),
    )
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Project (optional)'), 'free-text')
    expect(screen.getByLabelText('Project (optional)')).toHaveValue('free-text')
  })

  it('passes a non-empty project as projectRef on every job', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockResolvedValue(result())
    vi.mocked(ingestUrl).mockResolvedValue(result())
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Project (optional)'), 'proj-a')
    const md = fileOf('a.md')
    dropFiles(dropZone(), [md])
    await user.type(screen.getByLabelText('URL'), 'https://x.example')
    await user.click(screen.getByRole('button', { name: 'Ingest URL' }))
    await waitFor(() => expect(ingestFile).toHaveBeenCalledWith(md, 'proj-a'))
    await waitFor(() => expect(ingestUrl).toHaveBeenCalledWith('https://x.example', 'proj-a'))
    const enqueueCalls = info.mock.calls.filter((c) => c[0] === '[add-to-kb] enqueue')
    expect(enqueueCalls).toHaveLength(2)
    expect(enqueueCalls[0][1]).toMatchObject({ target: 'file', project_ref: 'proj-a' })
    expect(enqueueCalls[1][1]).toMatchObject({ target: 'url', project_ref: 'proj-a' })
  })

  it('a blank project calls the helpers with no projectRef', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockResolvedValue(result())
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Project (optional)'), '   ')
    const md = fileOf('a.md')
    dropFiles(dropZone(), [md])
    await waitFor(() => expect(ingestFile).toHaveBeenCalledWith(md, undefined))
    const enqueueCalls = info.mock.calls.filter((c) => c[0] === '[add-to-kb] enqueue')
    expect(enqueueCalls[0][1]).toMatchObject({ project_ref: null })
  })

  // ── Queue order ─────────────────────────────────────────────────────────

  it('processes jobs one at a time in submission order', async () => {
    renderPanel()
    const fileDefer = deferred<IngestFileResult>()
    const urlDefer = deferred<IngestFileResult>()
    vi.mocked(ingestFile).mockImplementation(() => fileDefer.promise)
    vi.mocked(ingestUrl).mockImplementation(() => urlDefer.promise)

    dropFiles(dropZone(), [fileOf('a.md')])
    await waitFor(() => expect(ingestFile).toHaveBeenCalledTimes(1))

    const user = userEvent.setup()
    await user.type(screen.getByLabelText('URL'), 'https://b.example')
    await user.click(screen.getByRole('button', { name: 'Ingest URL' }))

    // A is ingesting (indeterminate progress), B is queued and not started.
    expect(screen.getByText('a.md')).toBeInTheDocument()
    expect(screen.getByText('Ingesting…')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
    expect(screen.getByText('Queued')).toBeInTheDocument()
    expect(screen.getByText('https://b.example')).toBeInTheDocument()
    expect(ingestFile).toHaveBeenCalledTimes(1)
    expect(ingestUrl).not.toHaveBeenCalled()

    await act(async () => {
      fileDefer.resolve(result())
    })
    await waitFor(() => expect(ingestUrl).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Ingested · 1 entries · 1 chunks')).toBeInTheDocument()
    expect(screen.getByText('Ingesting…')).toBeInTheDocument()
    expect(screen.queryByText('Queued')).toBeNull()

    await act(async () => {
      urlDefer.resolve(result())
    })
    await waitFor(() =>
      expect(screen.getAllByText('Ingested · 1 entries · 1 chunks')).toHaveLength(2),
    )
    // A settled job never reorders the list.
    expect(
      orderOf(screen.getByText('a.md'), screen.getByText('https://b.example')),
    ).toBe(-1)
  })

  it('the queue continues past a failed job', async () => {
    renderPanel()
    vi.mocked(ingestFile)
      .mockImplementationOnce(() =>
        Promise.reject(new ApiError(403, 'Interactive surface required.')),
      )
      .mockImplementationOnce(() => Promise.resolve(result()))
    vi.mocked(ingestUrl).mockResolvedValue(result())

    dropFiles(dropZone(), [fileOf('one.md'), fileOf('two.md')])
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('URL'), 'https://three.example')
    await user.click(screen.getByRole('button', { name: 'Ingest URL' }))

    await waitFor(() =>
      expect(
        screen.getByText('Failed · Interactive surface required.'),
      ).toBeInTheDocument(),
    )
    await waitFor(() =>
      expect(screen.getAllByText('Ingested · 1 entries · 1 chunks')).toHaveLength(2),
    )
    expect(ingestFile).toHaveBeenCalledTimes(2)
    expect(ingestUrl).toHaveBeenCalledTimes(1)
  })

  // ── Result rendering ─────────────────────────────────────────────────────

  it('renders ingested results with entry links that open the drawer', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockResolvedValue(
      result({
        action: 'ingested',
        entry_count: 2,
        entry_ids: ['kb-00001', 'kb-00002'],
        summary: 'A short summary',
        chunks_processed: 3,
      }),
    )
    dropFiles(dropZone(), [fileOf('a.md')])
    expect(
      await screen.findByText('Ingested · 2 entries · 3 chunks'),
    ).toBeInTheDocument()
    expect(screen.getByText('A short summary')).toBeInTheDocument()
    fireEvent.click(screen.getByText('kb-00001'))
    expect(openEntry).toHaveBeenCalledWith('kb-00001')
    fireEvent.click(screen.getByText('kb-00002'))
    expect(openEntry).toHaveBeenCalledWith('kb-00002')
  })

  it('renders skipped, unchanged and flagged results as non-failures', async () => {
    renderPanel()
    vi.mocked(ingestFile)
      .mockResolvedValueOnce(result({ action: 'skipped', reason: 'duplicate source' }))
      .mockResolvedValueOnce(result({ action: 'unchanged', reason: 'same content hash' }))
      .mockResolvedValueOnce(result({ action: 'flagged', reason: 'secrets detected' }))
    dropFiles(dropZone(), [fileOf('a.md'), fileOf('b.md'), fileOf('c.md')])
    expect(await screen.findByText('Skipped · duplicate source')).toBeInTheDocument()
    expect(await screen.findByText('Unchanged · same content hash')).toBeInTheDocument()
    expect(await screen.findByText('Flagged · secrets detected')).toBeInTheDocument()
    const alerts = screen.getAllByRole('alert')
    expect(alerts).toHaveLength(3)
    expect(alerts.map((a) => a.className)).toMatchObject([
      expect.stringContaining('MuiAlert-standardSuccess'),
      expect.stringContaining('MuiAlert-standardSuccess'),
      expect.stringContaining('MuiAlert-standardWarning'),
    ])
  })

  it('an HTTP-200 action error renders as a failure alert, warns, and the queue continues', async () => {
    renderPanel()
    vi.mocked(ingestFile)
      .mockResolvedValueOnce(
        result({ action: 'error', reason: 'Failed to fetch: 404' }),
      )
      .mockResolvedValueOnce(result())
    dropFiles(dropZone(), [fileOf('bad.md'), fileOf('good.md')])
    expect(await screen.findByText('Failed · Failed to fetch: 404')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('Ingested · 1 entries · 1 chunks')).toBeInTheDocument())
    expect(ingestFile).toHaveBeenCalledTimes(2)
    expect(warn).toHaveBeenCalledWith('[add-to-kb] ingest error', {
      name: 'bad.md',
      reason: 'Failed to fetch: 404',
    })
  })

  it('a thrown ApiError renders the server message and does not stop the queue', async () => {
    renderPanel()
    vi.mocked(ingestFile)
      .mockRejectedValueOnce(new ApiError(422, 'A non-empty filename is required.'))
      .mockResolvedValueOnce(result())
    dropFiles(dropZone(), [fileOf('bad.md'), fileOf('good.md')])
    expect(
      await screen.findByText('Failed · A non-empty filename is required.'),
    ).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('Ingested · 1 entries · 1 chunks')).toBeInTheDocument())
    expect(warn).toHaveBeenCalledWith('[add-to-kb] ingest error', {
      name: 'bad.md',
      reason: 'A non-empty filename is required.',
    })
  })

  it('a server 413 that reached the wire trips the size-gate warn', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockRejectedValue(
      new ApiError(
        413,
        'Upload exceeds maximum file size of 10485760 bytes. Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads.',
      ),
    )
    dropFiles(dropZone(), [fileOf('sneaky.md')])
    expect(
      await screen.findByText(
        'Failed · Upload exceeds maximum file size of 10485760 bytes. Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads.',
      ),
    ).toBeInTheDocument()
    expect(warn).toHaveBeenCalledWith('[add-to-kb] size gate bypassed', {
      name: 'sneaky.md',
      size: 3,
      client_max: MAX_FILE_SIZE_BYTES,
      server_detail:
        'Upload exceeds maximum file size of 10485760 bytes. Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads.',
    })
    // A local classifyFile rejection never reaches the wire: no tripwire.
    expect(warn).not.toHaveBeenCalledWith('[add-to-kb] size gate bypassed', {
      name: 'big.md',
      size: 10485761,
      client_max: MAX_FILE_SIZE_BYTES,
      server_detail: expect.anything(),
    })
  })

  // ── Clear finished ──────────────────────────────────────────────────────

  it('Clear finished removes done and failed jobs, keeps queued and ingesting', async () => {
    renderPanel()
    const fileDefer = deferred<IngestFileResult>()
    vi.mocked(ingestFile)
      .mockImplementationOnce(() => Promise.reject(new ApiError(413, 'too big')))
      .mockImplementation(() => fileDefer.promise)
    const clear = screen.getByRole('button', { name: 'Clear finished' })
    expect(clear).toBeDisabled()

    dropFiles(dropZone(), [fileOf('fails.md'), fileOf('stays.md')])
    await waitFor(() => expect(screen.getByText('Failed · too big')).toBeInTheDocument())
    // stays.md is ingesting on its deferred promise.
    await waitFor(() => expect(screen.getByText('Ingesting…')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: 'Clear finished' }))
    await waitFor(() => expect(screen.queryByText('fails.md')).toBeNull())
    expect(screen.getByText('stays.md')).toBeInTheDocument()
    // Nothing settled is left: the control is disabled again.
    expect(screen.getByRole('button', { name: 'Clear finished' })).toBeDisabled()
  })

  // ── Refresh + structured logs ───────────────────────────────────────────

  it('fires onIngested only for action ingested and logs the refresh', async () => {
    const onIngested = vi.fn()
    vi.mocked(ingestFile)
      .mockResolvedValueOnce(
        result({
          action: 'ingested',
          entry_count: 2,
          entry_ids: ['kb-1', 'kb-2'],
        }),
      )
      .mockResolvedValueOnce(result({ action: 'skipped', reason: 'dup' }))
    renderPanel(onIngested)
    dropFiles(dropZone(), [fileOf('a.md'), fileOf('b.md')])
    await waitFor(() => expect(screen.getByText('Skipped · dup')).toBeInTheDocument())
    expect(onIngested).toHaveBeenCalledTimes(1)
    expect(info).toHaveBeenCalledWith('[add-to-kb] refresh', {
      action: 'ingested',
      entry_count: 2,
      entry_ids_count: 2,
    })
  })

  it('logs enqueue and done with the pinned payloads', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockResolvedValue(
      result({ entry_count: 2, chunks_processed: 5 }),
    )
    const f = fileOf('a.md')
    dropFiles(dropZone(), [f])
    expect(info).toHaveBeenCalledWith('[add-to-kb] enqueue', {
      target: 'file',
      name: 'a.md',
      size: 3,
      project_ref: null,
    })
    await waitFor(() => expect(screen.getByText('Ingested · 2 entries · 5 chunks')).toBeInTheDocument())
    const doneCalls = info.mock.calls.filter((c) => c[0] === '[add-to-kb] done')
    expect(doneCalls).toHaveLength(1)
    const payload = doneCalls[0][1] as Record<string, unknown>
    expect(payload.target).toBe('file')
    expect(payload.action).toBe('ingested')
    expect(payload.entry_count).toBe(2)
    expect(payload.chunks_processed).toBe(5)
    expect(typeof payload.duration_ms).toBe('number')
    expect(payload.duration_ms).toBeGreaterThanOrEqual(0)
  })

  it('never violates the queue invariant', async () => {
    renderPanel()
    vi.mocked(ingestFile).mockResolvedValue(result())
    dropFiles(dropZone(), [fileOf('a.md'), fileOf('b.md')])
    await waitFor(() =>
      expect(screen.getAllByText('Ingested · 1 entries · 1 chunks')).toHaveLength(2),
    )
    expect(warn).not.toHaveBeenCalledWith('[add-to-kb] invariant violated', expect.anything())
  })
})
