/**
 * Home page's Add to KB panel: drop files or URLs (or pick files, or type a
 * URL), optionally into a project, and watch a strictly sequential queue of
 * ingestion jobs.  Frontend only — the three /api/kb/ingest endpoints do the
 * work.  The queue is page state; nothing is persisted.
 */
import { useEffect, useRef, useState } from 'react'
import type { DragEvent, ChangeEvent, FormEvent } from 'react'
import Alert from '@mui/material/Alert'
import Autocomplete from '@mui/material/Autocomplete'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import LinearProgress from '@mui/material/LinearProgress'
import Link from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { ApiError, ingestFile, ingestUrl, listKbProjects } from '../api'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import { ResponseCard } from './ResponseCard'
import {
  ALLOWED_EXTENSIONS,
  MAX_FILE_SIZE_BYTES,
  classifyFile,
  jobErrorMessage,
  queueInvariantError,
} from './addToKbModel'
import type { IngestFileResult, Job } from './addToKbModel'

const ACCEPT_ATTR = [...ALLOWED_EXTENSIONS].join(',')
const HTTP_URL = /^https?:\/\//i
const DROP_LABEL = 'Drop files or URLs here'
const DROP_CAPTION = 'Accepted: .md, .txt, .pdf and other text/code types · max 10 MiB'

/** One queued upload or URL plus its captured project_ref. */
interface JobPayload {
  file?: File
  url?: string
  projectRef: string | null
}

/** The done-job Alert: severity and text follow result.action. */
function ResultAlert({
  result,
  onOpenEntry,
}: {
  result: IngestFileResult
  onOpenEntry: (id: string) => void
}) {
  let severity: 'success' | 'warning' | 'error' | 'info'
  let text: string
  switch (result.action) {
    case 'ingested':
      severity = 'success'
      text = `Ingested · ${result.entry_count} entries · ${result.chunks_processed} chunks`
      break
    case 'skipped':
      severity = 'success'
      text = `Skipped · ${result.reason ?? ''}`
      break
    case 'unchanged':
      severity = 'success'
      text = `Unchanged · ${result.reason ?? ''}`
      break
    case 'flagged':
      severity = 'warning'
      text = `Flagged · ${result.reason ?? ''}`
      break
    case 'error':
      severity = 'error'
      text = `Failed · ${result.reason ?? ''}`
      break
    case 'dry_run':
      // Unreachable from this UI (dry_run is never sent), handled for the union.
      severity = 'info'
      text = `Dry run · ${result.reason ?? ''}`
      break
  }
  return (
    <Alert severity={severity}>
      <Box component="div">{text}</Box>
      {result.summary !== null && (
        <Typography variant="body2">{result.summary}</Typography>
      )}
      {result.action === 'ingested' &&
        result.entry_ids.map((id) => (
          <Link
            key={id}
            component="button"
            type="button"
            onClick={() => onOpenEntry(id)}
            sx={{ display: 'inline-block', mr: 1 }}
          >
            {id}
          </Link>
        ))}
    </Alert>
  )
}

export function AddToKbPanel({ onIngested }: { onIngested?: () => void }) {
  const { openEntry } = useEntryDrawer()

  const [jobs, setJobs] = useState<Job[]>([])
  const [urlValue, setUrlValue] = useState('')
  const [projectValue, setProjectValue] = useState('')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [rejectNotice, setRejectNotice] = useState<string | null>(null)

  const idRef = useRef(0)
  const startedRef = useRef<Set<number>>(new Set())
  const payloadRef = useRef<Map<number, JobPayload>>(new Map())
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const onIngestedRef = useRef<(() => void) | undefined>(onIngested)

  useEffect(() => {
    onIngestedRef.current = onIngested
  }, [onIngested])

  // Project suggestions: the KB project list, once. Free text always allowed.
  useEffect(() => {
    listKbProjects()
      .then((resp) => {
        setSuggestions(resp.items.map((p) => p.name))
      })
      .catch((err: unknown) => {
        console.warn('[add-to-kb] project suggestions failed', { error: String(err) })
      })
  }, [])

  const enqueue = (
    target: 'file' | 'url',
    name: string,
    size: number | null,
    payload: JobPayload,
  ) => {
    const id = ++idRef.current
    payloadRef.current.set(id, payload)
    console.info('[add-to-kb] enqueue', {
      target,
      name,
      size,
      project_ref: payload.projectRef,
    })
    setJobs((prev) => [
      ...prev,
      { id, name, target, state: 'queued', result: null, error: null },
    ])
  }

  const resolvedProject = () => {
    const trimmed = projectValue.trim()
    return trimmed ? trimmed : null
  }

  /** classifyFile path shared by the drop zone and the Browse picker. */
  const acceptFile = (file: File) => {
    const verdict = classifyFile(file.name, file.size)
    if (!verdict.ok) {
      console.info('[add-to-kb] reject', {
        name: file.name,
        size: file.size,
        kind: verdict.kind,
        message: verdict.message,
      })
      setRejectNotice(`Rejected: ${file.name} — ${verdict.message}`)
      return
    }
    enqueue('file', file.name, file.size, {
      file,
      projectRef: resolvedProject(),
    })
  }

  const handleDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault()
    setRejectNotice(null)
    const dt = e.dataTransfer

    const files = Array.from(dt.files ?? [])
    for (const file of files) acceptFile(file)

    // A link or URL dragged from a browser arrives as text/uri-list (or
    // text/plain); one job per http(s) line, after the file jobs.
    const uriList = dt.getData('text/uri-list')
    const text = uriList !== '' ? uriList : dt.getData('text/plain')
    const urls = text
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => HTTP_URL.test(line))

    if (files.length === 0 && urls.length === 0) {
      console.debug('[add-to-kb] drop ignored', {
        kinds: dt.items ? Array.from(dt.items, (item) => item.type) : [],
      })
      return
    }
    for (const url of urls) {
      enqueue('url', url, null, { url, projectRef: resolvedProject() })
    }
  }

  const handleFiles = (e: ChangeEvent<HTMLInputElement>) => {
    setRejectNotice(null)
    for (const file of Array.from(e.target.files ?? [])) acceptFile(file)
    // Reset so picking the same file again fires another change event.
    e.target.value = ''
  }

  const handleUrlSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const url = urlValue.trim()
    if (!url) return
    enqueue('url', url, null, { url, projectRef: resolvedProject() })
    setUrlValue('')
  }

  // The queue pump: strictly one job at a time, in submission order.
  useEffect(() => {
    const invariant = queueInvariantError(jobs)
    if (invariant !== null) {
      console.warn('[add-to-kb] invariant violated', {
        error: invariant,
        jobs_len: jobs.length,
        ingesting_count: jobs.filter((j) => j.state === 'ingesting').length,
      })
    }
    if (jobs.some((j) => j.state === 'ingesting')) return
    const next = jobs.find((j) => j.state === 'queued')
    if (!next || startedRef.current.has(next.id)) return
    startedRef.current.add(next.id)

    const payload = payloadRef.current.get(next.id)
    if (!payload) return
    const started = performance.now()
    setJobs((prev) =>
      prev.map((j) => (j.id === next.id ? { ...j, state: 'ingesting' } : j)),
    )

    const projectRef = payload.projectRef ?? undefined
    const run =
      next.target === 'file'
        ? ingestFile(payload.file as File, projectRef)
        : ingestUrl(payload.url as string, projectRef)

    run
      .then((result) => {
        console.info('[add-to-kb] done', {
          target: next.target,
          action: result.action,
          entry_count: result.entry_count,
          chunks_processed: result.chunks_processed,
          duration_ms: Math.round(performance.now() - started),
        })
        if (result.action === 'error') {
          console.warn('[add-to-kb] ingest error', {
            name: next.name,
            reason: result.reason,
          })
        }
        if (result.action === 'ingested') {
          console.info('[add-to-kb] refresh', {
            action: result.action,
            entry_count: result.entry_count,
            entry_ids_count: result.entry_ids.length,
          })
          onIngestedRef.current?.()
        }
        setJobs((prev) =>
          prev.map((j) => (j.id === next.id ? { ...j, state: 'done', result } : j)),
        )
      })
      .catch((err: unknown) => {
        const message = jobErrorMessage(err)
        console.warn('[add-to-kb] ingest error', { name: next.name, reason: message })
        if (next.target === 'file' && err instanceof ApiError && err.status === 413) {
          // The client size gate (classifyFile) let this file through, yet the
          // server still rejected it as too large: the caps have drifted.
          console.warn('[add-to-kb] size gate bypassed', {
            name: next.name,
            size: payload.file?.size,
            client_max: MAX_FILE_SIZE_BYTES,
            server_detail: err.detail,
          })
        }
        setJobs((prev) =>
          prev.map((j) =>
            j.id === next.id ? { ...j, state: 'failed', error: message } : j,
          ),
        )
      })
  }, [jobs])

  const clearFinished = () => {
    setJobs((prev) => prev.filter((j) => j.state !== 'done' && j.state !== 'failed'))
  }
  const hasFinished = jobs.some((j) => j.state === 'done' || j.state === 'failed')

  return (
    <ResponseCard>
      <Stack spacing={1.5} data-testid="add-to-kb-panel">
        <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
          Add to KB
        </Typography>

        <Box
          data-testid="add-to-kb-drop"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          sx={{
            border: '1px dashed',
            borderColor: 'divider',
            borderRadius: 1,
            p: 1.5,
            textAlign: 'center',
          }}
        >
          <Typography variant="body2">{DROP_LABEL}</Typography>
          <Typography variant="caption" color="text.secondary" component="p">
            {DROP_CAPTION}
          </Typography>
          <Button size="small" onClick={() => fileInputRef.current?.click()}>
            Browse
          </Button>
          <input
            ref={fileInputRef}
            data-testid="add-to-kb-file-input"
            type="file"
            multiple
            accept={ACCEPT_ATTR}
            onChange={handleFiles}
            style={{ display: 'none' }}
          />
        </Box>

        {rejectNotice !== null && (
          <Typography variant="body2" color="error" data-testid="add-to-kb-reject">
            {rejectNotice}
          </Typography>
        )}

        <Box component="form" onSubmit={handleUrlSubmit}>
          <Stack direction="row" spacing={1} alignItems="flex-start">
            <TextField
              label="URL"
              value={urlValue}
              onChange={(e) => setUrlValue(e.target.value)}
              size="small"
              fullWidth
            />
            <Button type="submit" variant="outlined" disabled={!urlValue.trim()}>
              Ingest URL
            </Button>
          </Stack>
        </Box>

        <Autocomplete
          freeSolo
          options={suggestions}
          inputValue={projectValue}
          onInputChange={(_, v) => setProjectValue(v)}
          renderInput={(params) => (
            <TextField {...params} label="Project (optional)" size="small" />
          )}
        />

        <Stack spacing={1} data-testid="add-to-kb-jobs">
          {jobs.map((job) => (
            <Box key={job.id} sx={{ wordBreak: 'break-all' }}>
              <Typography variant="body2" noWrap>
                {job.name}
              </Typography>
              {job.state === 'queued' && (
                <Typography variant="body2" color="text.secondary">
                  Queued
                </Typography>
              )}
              {job.state === 'ingesting' && (
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Ingesting…
                  </Typography>
                  <LinearProgress />
                </Box>
              )}
              {job.state === 'done' && job.result !== null && (
                <ResultAlert result={job.result} onOpenEntry={openEntry} />
              )}
              {job.state === 'failed' && (
                <Alert severity="error">Failed · {job.error}</Alert>
              )}
            </Box>
          ))}
        </Stack>

        <Button
          size="small"
          onClick={clearFinished}
          disabled={!hasFinished}
          sx={{ alignSelf: 'flex-start' }}
        >
          Clear finished
        </Button>
      </Stack>
    </ResponseCard>
  )
}
