import { useState, useEffect } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Card from '@mui/material/Card'
import CardActionArea from '@mui/material/CardActionArea'
import CardContent from '@mui/material/CardContent'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { searchKb, listKbProjects } from '../api'
import { useEntryDrawer } from '../contexts/EntryDrawerContext'
import type {
  SearchResult,
  KbProject,
  EntryType,
  MatchSource,
} from '../kbTypes'

const ENTRY_TYPES: EntryType[] = [
  'factual_reference',
  'decision',
  'pattern_convention',
  'lesson_learned',
  'mental_map',
]

const ENTRY_TYPE_COLORS: Record<EntryType, 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning'> = {
  factual_reference: 'info',
  decision: 'primary',
  pattern_convention: 'secondary',
  lesson_learned: 'warning',
  mental_map: 'success',
}

export function Search() {
  const { openEntry } = useEntryDrawer()
  const [query, setQuery] = useState('')
  const [projectRef, setProjectRef] = useState('')
  const [entryType, setEntryType] = useState('any')
  const [tagsInput, setTagsInput] = useState('')
  const [limit, setLimit] = useState(10)
  const [projects, setProjects] = useState<KbProject[]>([])
  const [results, setResults] = useState<SearchResult[] | null>(null)
  const [filteredCount, setFilteredCount] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listKbProjects()
      .then((r) => setProjects(r.items))
      .catch(() => {
        // non-fatal — projects list stays empty
      })
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResults(null)
    setFilteredCount(null)

    // Build body — omit blank filters entirely (never '' or literal 'any')
    const body: Parameters<typeof searchKb>[0] = {
      query: query.trim(),
      limit,
      include_stale: false,
      include_expired: false,
    }
    if (projectRef) body.project_ref = projectRef
    if (entryType !== 'any') body.entry_type = entryType
    const tags = tagsInput
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
    if (tags.length > 0) body.tags = tags

    try {
      const resp = await searchKb(body)
      setResults(resp.results)
      setFilteredCount(resp.filtered_count)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Search Knowledge Base
      </Typography>

      <Box component="form" onSubmit={(e) => void handleSubmit(e)} sx={{ mb: 3 }}>
        <Stack spacing={2}>
          <TextField
            label="Query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            required
            fullWidth
            autoFocus
          />

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <FormControl fullWidth>
              <InputLabel id="project-label">Project</InputLabel>
              <Select
                labelId="project-label"
                label="Project"
                value={projectRef}
                onChange={(e) => setProjectRef(e.target.value)}
              >
                <MenuItem value="">All projects</MenuItem>
                {projects.map((p) => (
                  <MenuItem key={p.name} value={p.name}>
                    {p.name} ({p.entry_count})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel id="type-label">Entry type</InputLabel>
              <Select
                labelId="type-label"
                label="Entry type"
                value={entryType}
                onChange={(e) => setEntryType(e.target.value)}
              >
                <MenuItem value="any">any</MenuItem>
                {ENTRY_TYPES.map((t) => (
                  <MenuItem key={t} value={t}>
                    {t}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Stack>

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <TextField
              label="Tags (comma-separated)"
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              fullWidth
              placeholder="tag1, tag2"
            />

            <TextField
              label="Limit"
              type="number"
              value={limit}
              onChange={(e) => {
                const v = Math.min(50, Math.max(1, Number(e.target.value)))
                setLimit(v)
              }}
              inputProps={{ min: 1, max: 50 }}
              sx={{ minWidth: 100 }}
            />
          </Stack>

          <Button
            type="submit"
            variant="contained"
            disabled={loading || !query.trim()}
            startIcon={loading ? <CircularProgress size={18} /> : null}
          >
            {loading ? 'Searching…' : 'Search'}
          </Button>
        </Stack>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {filteredCount !== null && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {filteredCount} result{filteredCount !== 1 ? 's' : ''}
        </Typography>
      )}

      {results && (
        <Stack spacing={2}>
          {results.map((r) => (
            <ResultCard key={r.entry.id} result={r} onOpenEntry={openEntry} />
          ))}
          {results.length === 0 && (
            <Typography color="text.secondary">No results found.</Typography>
          )}
        </Stack>
      )}
    </Box>
  )
}

function ResultCard({
  result,
  onOpenEntry,
}: {
  result: SearchResult
  onOpenEntry: (id: string) => void
}) {
  const { entry, effective_confidence, match_source, staleness_warning } = result
  return (
    <Card variant="outlined">
      <CardActionArea onClick={() => onOpenEntry(entry.id)}>
        <CardContent>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" sx={{ mb: 0.5 }}>
            <Typography variant="subtitle1" component="span" sx={{ fontWeight: 600 }}>
              {entry.short_title}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {entry.id}
            </Typography>
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" sx={{ mb: 1 }}>
            <Chip
              label={entry.entry_type}
              color={ENTRY_TYPE_COLORS[entry.entry_type as EntryType] ?? 'default'}
              size="small"
            />
            {entry.tags.map((tag) => (
              <Chip key={tag} label={tag} size="small" variant="outlined" />
            ))}
          </Stack>

          <Stack direction="row" spacing={2} alignItems="center">
            <Typography variant="body2">
              Confidence: {effective_confidence.toFixed(2)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {match_source as MatchSource}
            </Typography>
          </Stack>

          {staleness_warning && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              {staleness_warning}
            </Alert>
          )}
        </CardContent>
      </CardActionArea>
    </Card>
  )
}
