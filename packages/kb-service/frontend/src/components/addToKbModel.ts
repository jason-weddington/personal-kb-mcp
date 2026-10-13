/**
 * Pure model for the Home page's Add to KB panel — no React, no network.
 *
 * The allowlists and the size cap are pinned to kb-core's
 * kb_core/ingest/ingester.py:34-91 / 94-104 and kb-service's config default
 * (KB_INGEST_MAX_FILE_SIZE, 10 MiB) so the client rejects client-side exactly
 * what the server would reject.
 */
import { ApiError } from '../api'
import type { IngestFileResult } from '../kbTypes'

export type { IngestFileResult } from '../kbTypes'

/** 56 extensions ingestable as text (ingester.py:34-91). */
export const ALLOWED_EXTENSIONS: ReadonlySet<string> = new Set([
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
])

/** Extension-less files accepted by exact name (ingester.py:94-104). */
export const ALLOWED_NAMES: ReadonlySet<string> = new Set([
  'Dockerfile',
  'Makefile',
  'Rakefile',
  'Gemfile',
  'Procfile',
  'README',
  'CHANGELOG',
  'LICENSE',
  'NOTES',
])

/** Server default for KB_INGEST_MAX_FILE_SIZE (config.py: 10 * 1024 * 1024). */
export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

export type ClassifyResult =
  | { ok: true }
  | { ok: false; kind: 'type' | 'size'; message: string }

/**
 * Client-side gate mirroring the server's rules:
 * - the type rule mirrors _is_allowed_file (ingester.py:898-902): an exact
 *   ALLOWED_NAMES match, else Python Path.suffix semantics — the text from
 *   the last '.' when that dot is not the first character and not trailing,
 *   lowercased.  So '.md' as a bare dotfile name yields '' and is rejected,
 *   exactly like the server, and 'a.Dockerfile' has suffix '.dockerfile'
 *   (not in the set) and is rejected.
 * - the size rule mirrors ingest_routes.py:216-218: strictly greater than the
 *   cap rejects; exactly the cap is accepted.
 */
export function classifyFile(name: string, size: number): ClassifyResult {
  const i = name.lastIndexOf('.')
  const ext = 0 < i && i < name.length - 1 ? name.slice(i).toLowerCase() : ''
  if (!ALLOWED_NAMES.has(name) && !ALLOWED_EXTENSIONS.has(ext)) {
    return { ok: false, kind: 'type', message: 'Unsupported file type' }
  }
  if (size > MAX_FILE_SIZE_BYTES) {
    return { ok: false, kind: 'size', message: 'Exceeds the 10485760-byte limit' }
  }
  return { ok: true }
}

export type JobTarget = 'file' | 'url'
export type JobState = 'queued' | 'ingesting' | 'done' | 'failed'

/** One file or URL submitted from the panel. */
export interface Job {
  id: number
  name: string
  target: JobTarget
  state: JobState
  result: IngestFileResult | null
  error: string | null
}

/**
 * Queue tripwire: null when the job list is legal, an error string otherwise.
 * Legal means at most one job ingesting, no settled job back in 'ingesting'
 * (a job carrying a result or error must be done or failed), and unique ids.
 * Mirrors the [graph-history] invariant helper pattern.
 */
export function queueInvariantError(jobs: Job[]): string | null {
  let ingesting = 0
  const ids = new Set<number>()
  for (const job of jobs) {
    if (ids.has(job.id)) return `duplicate job id ${job.id}`
    ids.add(job.id)
    if (job.state === 'ingesting') {
      ingesting += 1
      if (job.result !== null || job.error !== null) {
        return `job ${job.id} re-ingesting after settling`
      }
    }
  }
  if (ingesting > 1) return `${ingesting} jobs ingesting at once`
  return null
}

/**
 * Map a thrown error to the message shown on the failed job row.
 * - 401: kbRequest already removed the token and redirected; the row says the
 *   session expired rather than the hardcoded 'Unauthorized' detail.
 * - any other ApiError: the server detail verbatim (403 write policy, 413 size
 *   cap, 422 empty filename, 503 missing safety dependencies).
 * - anything else (a fetch TypeError): a network error with its message.
 */
export function jobErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return 'Session expired. Sign in again.'
    return err.detail
  }
  if (err instanceof Error) return `Network error: ${err.message}`
  return `Network error: ${String(err)}`
}
