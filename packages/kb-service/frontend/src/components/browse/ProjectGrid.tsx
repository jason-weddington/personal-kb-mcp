import { useState } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Card from '@mui/material/Card'
import CardActionArea from '@mui/material/CardActionArea'
import CardContent from '@mui/material/CardContent'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Skeleton from '@mui/material/Skeleton'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { browseProjects } from '../../api'
import type { BrowseProject, EntryType } from '../../kbTypes'
import { ENTRY_TYPE_COLORS, entryTypeLabel } from '../../entryTypes'
import { entriesLabel, formatRelative } from '../../browse/format'
import { usePagedList } from '../../browse/usePagedList'
import { useDebouncedValue } from '../../browse/useDebouncedValue'
import { LoadError } from './EntryList'

const PROJECT_PAGE_SIZE = 24
const projectKey = (p: BrowseProject) => p.project_ref

function segmentColor(type: string): string {
  const color = ENTRY_TYPE_COLORS[type as EntryType]
  return color && color !== 'default' ? `${color}.main` : 'grey.500'
}

function ProjectCard({
  project,
  now,
  onOpen,
}: {
  project: BrowseProject
  now: number
  onOpen: (ref: string) => void
}) {
  const counts = Object.entries(project.counts_by_type)
  return (
    <Card data-testid="project-card" variant="outlined" sx={{ height: '100%' }}>
      <CardActionArea
        aria-label={`Open project ${project.project_ref}`}
        onClick={() => onOpen(project.project_ref)}
        sx={{ height: '100%' }}
      >
        <CardContent>
          <Typography variant="subtitle1" fontWeight="bold" noWrap>
            {project.project_ref}
          </Typography>
          <Typography variant="caption" color="text.secondary" component="p">
            {`${entriesLabel(project.entry_count)} · active ${formatRelative(project.last_activity, now)}`}
          </Typography>
          <Box
            sx={{
              display: 'flex',
              height: '6px',
              borderRadius: 3,
              overflow: 'hidden',
              my: 1,
            }}
          >
            {counts.map(([type, n]) => (
              <Box
                key={type}
                title={`${entryTypeLabel(type)}: ${n}`}
                sx={{
                  width: `${(n / project.entry_count) * 100}%`,
                  bgcolor: segmentColor(type),
                }}
              />
            ))}
          </Box>
          <Stack direction="row" spacing={0.5} useFlexGap sx={{ flexWrap: 'wrap' }}>
            {counts.map(([type, n]) => (
              <Chip key={type} size="small" label={`${entryTypeLabel(type)} ${n}`} />
            ))}
          </Stack>
        </CardContent>
      </CardActionArea>
    </Card>
  )
}

/** Filterable, paged grid of project cards. */
export function ProjectGrid({ onOpen }: { onOpen: (ref: string) => void }) {
  const [text, setText] = useState('')
  const q = useDebouncedValue(text.trim(), 250)
  const list = usePagedList(
    `projects:${q}`,
    (cursor) =>
      browseProjects({ q: q === '' ? undefined : q, limit: PROJECT_PAGE_SIZE, cursor }),
    projectKey,
  )

  let body
  if (list.status === 'loading') {
    body = (
      <Grid container spacing={2}>
        {Array.from({ length: 6 }, (_, i) => (
          <Grid key={i} size={{ xs: 12, sm: 6, lg: 4 }}>
            <Skeleton variant="rounded" height={120} data-testid="project-skeleton" />
          </Grid>
        ))}
      </Grid>
    )
  } else if (list.status === 'error') {
    body = <LoadError message={list.errorMessage ?? ''} onRetry={list.retry} />
  } else if (list.items.length === 0) {
    body = (
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ textAlign: 'center', py: 4 }}
      >
        {q ? `No projects match "${q}".` : 'No projects yet.'}
      </Typography>
    )
  } else {
    const now = list.fetchedAt ?? 0
    body = (
      <>
        <Grid container spacing={2}>
          {list.items.map((p) => (
            <Grid key={p.project_ref} size={{ xs: 12, sm: 6, lg: 4 }}>
              <ProjectCard project={p} now={now} onOpen={onOpen} />
            </Grid>
          ))}
        </Grid>
        {list.nextCursor !== null && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Button onClick={list.loadMore} disabled={list.loadingMore}>
              Load more
            </Button>
          </Box>
        )}
      </>
    )
  }

  return (
    <Box>
      <TextField
        label="Filter projects"
        size="small"
        value={text}
        onChange={(e) => setText(e.target.value)}
        sx={{ mb: 2, minWidth: 240 }}
      />
      {body}
    </Box>
  )
}
