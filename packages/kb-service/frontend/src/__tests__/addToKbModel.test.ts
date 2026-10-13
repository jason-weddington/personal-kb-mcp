import { describe, it, expect } from 'vitest'
import { ApiError } from '../api'
import {
  ALLOWED_EXTENSIONS,
  ALLOWED_NAMES,
  MAX_FILE_SIZE_BYTES,
  classifyFile,
  jobErrorMessage,
  queueInvariantError,
} from '../components/addToKbModel'
import type { IngestFileResult, Job } from '../components/addToKbModel'

// Drift guards: the literal allowlists spelled out from
// kb_core/ingest/ingester.py:34-91 (56 extensions) and :94-104 (9 names).
const EXTENSIONS = [
  '.md',
  '.markdown',
  '.txt',
  '.rst',
  '.org',
  '.adoc',
  '.tex',
  '.py',
  '.js',
  '.ts',
  '.jsx',
  '.tsx',
  '.rb',
  '.go',
  '.rs',
  '.java',
  '.kt',
  '.c',
  '.cpp',
  '.h',
  '.hpp',
  '.cs',
  '.swift',
  '.sh',
  '.bash',
  '.zsh',
  '.fish',
  '.yaml',
  '.yml',
  '.toml',
  '.ini',
  '.cfg',
  '.conf',
  '.json',
  '.xml',
  '.html',
  '.css',
  '.scss',
  '.sql',
  '.r',
  '.R',
  '.jl',
  '.lua',
  '.vim',
  '.el',
  '.clj',
  '.ex',
  '.exs',
  '.erl',
  '.hs',
  '.ml',
  '.nix',
  '.tf',
  '.Dockerfile',
  '.Makefile',
  '.pdf',
]

const NAMES = [
  'Dockerfile',
  'Makefile',
  'Rakefile',
  'Gemfile',
  'Procfile',
  'README',
  'CHANGELOG',
  'LICENSE',
  'NOTES',
]

const job = (id: number, state: Job['state']): Job => ({
  id,
  name: `job-${id}`,
  target: 'file',
  state,
  result: null,
  error: null,
})

describe('addToKbModel allowlists', () => {
  it('ALLOWED_EXTENSIONS is the exact 56-entry set from ingester.py', () => {
    expect(EXTENSIONS).toHaveLength(56)
    expect(ALLOWED_EXTENSIONS.size).toBe(56)
    expect([...ALLOWED_EXTENSIONS].sort()).toEqual([...EXTENSIONS].sort())
  })

  it('ALLOWED_NAMES is the exact 9-entry set from ingester.py', () => {
    expect(NAMES).toHaveLength(9)
    expect(ALLOWED_NAMES.size).toBe(9)
    expect([...ALLOWED_NAMES].sort()).toEqual([...NAMES].sort())
  })

  it('MAX_FILE_SIZE_BYTES matches the server default (10 MiB)', () => {
    expect(MAX_FILE_SIZE_BYTES).toBe(10485760)
  })
})

describe('classifyFile', () => {
  it('accepts .pdf, .md and .txt', () => {
    expect(classifyFile('notes.pdf', 10)).toEqual({ ok: true })
    expect(classifyFile('notes.md', 10)).toEqual({ ok: true })
    expect(classifyFile('notes.txt', 10)).toEqual({ ok: true })
  })

  it('accepts a README with no extension via ALLOWED_NAMES', () => {
    expect(classifyFile('README', 10)).toEqual({ ok: true })
  })

  it("accepts 'X.PDF' — the suffix rule is case-insensitive like Path.suffix.lower()", () => {
    expect(classifyFile('X.PDF', 10)).toEqual({ ok: true })
  })

  it("rejects '.md' as a bare dotfile (Path.suffix is '' for it)", () => {
    const verdict = classifyFile('.md', 10)
    expect(verdict).toEqual({ ok: false, kind: 'type', message: 'Unsupported file type' })
  })

  it("rejects 'a.Dockerfile' — lowercased suffix '.dockerfile' is not in the set", () => {
    const verdict = classifyFile('a.Dockerfile', 10)
    expect(verdict).toEqual({ ok: false, kind: 'type', message: 'Unsupported file type' })
  })

  it('rejects .exe, .docx and .png as type', () => {
    for (const name of ['notes.exe', 'notes.docx', 'image.png']) {
      expect(classifyFile(name, 10)).toEqual({
        ok: false,
        kind: 'type',
        message: 'Unsupported file type',
      })
    }
  })

  it('rejects one byte over the cap and accepts exactly the cap', () => {
    expect(classifyFile('notes.md', 10485761)).toEqual({
      ok: false,
      kind: 'size',
      message: 'Exceeds the 10485760-byte limit',
    })
    expect(classifyFile('notes.md', 10485760)).toEqual({ ok: true })
  })
})

describe('queueInvariantError', () => {
  it('returns null for legal sequences', () => {
    expect(queueInvariantError([])).toBeNull()
    expect(
      queueInvariantError([job(1, 'done'), job(2, 'ingesting'), job(3, 'queued')]),
    ).toBeNull()
    expect(queueInvariantError([job(1, 'failed'), job(2, 'queued')])).toBeNull()
  })

  it('returns non-null when two jobs are ingesting at once', () => {
    expect(queueInvariantError([job(1, 'ingesting'), job(2, 'ingesting')])).not.toBeNull()
  })

  it('returns non-null for duplicate ids', () => {
    expect(queueInvariantError([job(1, 'done'), job(1, 'queued')])).not.toBeNull()
  })

  it('returns non-null when a settled job is back in ingesting', () => {
    const settled: IngestFileResult = {
      path: 'p',
      action: 'ingested',
      reason: null,
      entry_count: 1,
      entry_ids: ['kb-1'],
      summary: null,
      chunks_processed: 1,
      chunks_skipped: 0,
      chunks_flagged: 0,
    }
    const reingesting: Job = { ...job(1, 'ingesting'), result: settled }
    expect(queueInvariantError([reingesting])).not.toBeNull()
  })
})

describe('jobErrorMessage', () => {
  it('maps 401 to the session-expired message', () => {
    expect(jobErrorMessage(new ApiError(401, 'Unauthorized'))).toBe(
      'Session expired. Sign in again.',
    )
  })

  it('passes other ApiError details through verbatim', () => {
    expect(jobErrorMessage(new ApiError(403, 'Interactive surface required.'))).toBe(
      'Interactive surface required.',
    )
    expect(
      jobErrorMessage(
        new ApiError(
          413,
          'Upload exceeds maximum file size of 10485760 bytes. Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads.',
        ),
      ),
    ).toBe(
      'Upload exceeds maximum file size of 10485760 bytes. Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads.',
    )
    expect(jobErrorMessage(new ApiError(422, 'A non-empty filename is required.'))).toBe(
      'A non-empty filename is required.',
    )
    expect(
      jobErrorMessage(
        new ApiError(503, 'Safety dependencies missing: detect-secrets, scrubadub.'),
      ),
    ).toBe('Safety dependencies missing: detect-secrets, scrubadub.')
  })

  it('maps a non-ApiError to a network error with the message', () => {
    expect(jobErrorMessage(new TypeError('fetch failed'))).toBe('Network error: fetch failed')
  })
})

// Job is re-exported for typing only; keep tsc honest about the shape.
describe('Job shape', () => {
  it('has the pinned fields', () => {
    const j: Job = {
      id: 1,
      name: 'a.md',
      target: 'file',
      state: 'queued',
      result: null,
      error: null,
    }
    expect(j.state).toBe('queued')
  })

  it('accepts all state values', () => {
    const states: Job['state'][] = ['queued', 'ingesting', 'done', 'failed']
    expect(states).toHaveLength(4)
  })

  it('is not confused with the kbTypes result', () => {
    const r: IngestFileResult = {
      path: '',
      action: 'dry_run',
      reason: null,
      entry_count: 0,
      entry_ids: [],
      summary: null,
      chunks_processed: 0,
      chunks_skipped: 0,
      chunks_flagged: 0,
    }
    const j: Job = { id: 0, name: '', target: 'url', state: 'done', result: r, error: null }
    expect(j.result?.action).toBe('dry_run')
  })
})
