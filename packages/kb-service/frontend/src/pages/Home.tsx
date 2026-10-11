import { useEffect, useState } from 'react'
import Box from '@mui/material/Box'
import Grid from '@mui/material/Grid'
import Skeleton from '@mui/material/Skeleton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { ResponseCard } from '../components/ResponseCard'
import { BrowsePanel } from '../components/browse/BrowsePanel'
import {
  listKbProjects,
  listKbContributors,
  getMapsIndex,
  browseRecent,
  browseProjects,
} from '../api'

/** undefined = loading, null = failed, number = resolved value */
type StatValue = number | null | undefined

function StatRow({
  label,
  value,
}: {
  label: string
  value: StatValue
}) {
  const loading = value === undefined
  return (
    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ py: 0.5 }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      {loading ? (
        <Skeleton variant="text" width={48} data-testid={`stat-skeleton-${label.toLowerCase()}`} />
      ) : (
        <Typography variant="body2" fontWeight="medium" data-testid={`stat-value-${label.toLowerCase()}`}>
          {value !== null ? value.toLocaleString() : '—'}
        </Typography>
      )}
    </Stack>
  )
}

function StatsCard() {
  const [entries, setEntries] = useState<StatValue>(undefined)
  const [noProject, setNoProject] = useState<number>(0)
  const [projects, setProjects] = useState<StatValue>(undefined)
  const [contributors, setContributors] = useState<StatValue>(undefined)
  const [maps, setMaps] = useState<StatValue>(undefined)

  useEffect(() => {
    listKbProjects()
      .then((resp) => {
        setProjects(resp.items.length)
      })
      .catch(() => {
        setProjects(null)
      })

    // Entries = active, non-superseded entries with or without a project (the
    // unfiltered browse total). Project cards sum to Entries minus 'No project'.
    const countEntries = async () => {
      const total = (await browseRecent({ sort: 'created', limit: 1 })).total
      let inProjects = 0
      let cursor: string | null = null
      do {
        const page: Awaited<ReturnType<typeof browseProjects>> = await browseProjects({
          limit: 100,
          cursor,
        })
        inProjects += page.items.reduce((sum, p) => sum + p.entry_count, 0)
        cursor = page.next_cursor
      } while (cursor)
      return { total, none: Math.max(0, total - inProjects) }
    }
    countEntries()
      .then(({ total, none }) => {
        setEntries(total)
        setNoProject(none)
      })
      .catch(() => {
        setEntries(null)
      })

    listKbContributors()
      .then((resp) => {
        setContributors(resp.items.length)
      })
      .catch(() => {
        setContributors(null)
      })

    getMapsIndex()
      .then((resp) => {
        setMaps(resp.projects.reduce((sum, p) => sum + p.maps.length, 0))
      })
      .catch(() => {
        setMaps(null)
      })
  }, [])

  return (
    <ResponseCard>
      <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
        KB Stats
      </Typography>
      <StatRow label="Entries" value={entries} />
      {noProject > 0 && <StatRow label="No project" value={noProject} />}
      <StatRow label="Projects" value={projects} />
      <StatRow label="Contributors" value={contributors} />
      <StatRow label="Maps" value={maps} />
    </ResponseCard>
  )
}

export function Home() {
  return (
    <Box>
      <Grid container spacing={3}>
        {/* Left column: Browse panel (~2/3) */}
        <Grid size={{ xs: 12, md: 8 }} sx={{ order: { xs: 2, md: 1 } }}>
          <Typography variant="h5" gutterBottom>
            Browse the KB
          </Typography>
          <BrowsePanel />
        </Grid>

        {/* Right column: KB Stats (~1/3) */}
        <Grid size={{ xs: 12, md: 4 }} sx={{ order: { xs: 1, md: 2 } }}>
          <StatsCard />
        </Grid>
      </Grid>
    </Box>
  )
}
