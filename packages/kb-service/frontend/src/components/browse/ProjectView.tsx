import { useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import FormControl from '@mui/material/FormControl'
import IconButton from '@mui/material/IconButton'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { ApiError, browseProject } from '../../api'
import type { BrowseEntry, EntryType } from '../../kbTypes'
import { formatRelative } from '../../browse/format'
import { useDebouncedValue } from '../../browse/useDebouncedValue'
import { EntryList } from './EntryList'
import { TypeSelect } from './TypeSelect'

type ProjectSort = 'newest' | 'updated' | 'confidence'

const sectionFor = (r: BrowseEntry, all: BrowseEntry[]): string | null => {
  const hasMaps = all.some((x) => x.entry_type === 'mental_map')
  const hasOther = all.some((x) => x.entry_type !== 'mental_map')
  if (!hasMaps) return null
  return r.entry_type === 'mental_map' ? 'Mental maps' : hasOther ? 'Entries' : null
}

/** Entries of one project, mental maps first. */
export function ProjectView({
  project,
  entryType,
  onTypeChange,
  onBack,
}: {
  project: string
  entryType: EntryType | undefined
  onTypeChange: (t: EntryType | undefined) => void
  onBack: () => void
}) {
  const [sort, setSort] = useState<ProjectSort>('newest')
  const [text, setText] = useState('')
  const trimmed = useDebouncedValue(text.trim(), 250)
  const q = trimmed === '' ? undefined : trimmed

  return (
    <Box>
      <Stack
        direction="row"
        spacing={2}
        useFlexGap
        sx={{ flexWrap: 'wrap', alignItems: 'center', mb: 2 }}
      >
        <IconButton aria-label="Back to projects" onClick={onBack}>
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h6" sx={{ mr: 1 }}>
          {project}
        </Typography>
        <FormControl size="small" sx={{ minWidth: 170 }}>
          <InputLabel id="project-sort-label">Sort</InputLabel>
          <Select
            labelId="project-sort-label"
            label="Sort"
            value={sort}
            SelectDisplayProps={{ 'aria-labelledby': 'project-sort-label' }}
            onChange={(e) => setSort(e.target.value as ProjectSort)}
          >
            <MenuItem value="newest">Newest</MenuItem>
            <MenuItem value="updated">Recently updated</MenuItem>
            <MenuItem value="confidence">Confidence</MenuItem>
          </Select>
        </FormControl>
        <TextField
          label="Filter in project"
          size="small"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <TypeSelect labelId="project-type-label" value={entryType} onChange={onTypeChange} />
      </Stack>
      <EntryList
        listKey={`project:${project}:${sort}:${entryType ?? ''}:${q ?? ''}`}
        fetchPage={(cursor) =>
          browseProject(project, { sort, entryType, q, limit: 25, cursor })
        }
        sectionFor={sectionFor}
        ageLabelFor={(r, now) =>
          sort === 'updated'
            ? `updated ${formatRelative(r.updated_at, now)}`
            : formatRelative(r.created_at, now)
        }
        renderError={(e) =>
          e instanceof ApiError && e.status === 404 ? (
            <Alert severity="warning">Project not found: {project}</Alert>
          ) : null
        }
        emptyText="No entries match these filters."
      />
    </Box>
  )
}
