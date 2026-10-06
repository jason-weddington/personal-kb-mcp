import { useEffect, useState } from 'react'
import Box from '@mui/material/Box'
import Grid from '@mui/material/Grid'
import Skeleton from '@mui/material/Skeleton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { ResponseCard } from '../components/ResponseCard'
import { AskPanel } from '../components/AskPanel'
import { listKbProjects, listKbContributors, getMapsIndex } from '../api'

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
  const [projects, setProjects] = useState<StatValue>(undefined)
  const [contributors, setContributors] = useState<StatValue>(undefined)
  const [maps, setMaps] = useState<StatValue>(undefined)

  useEffect(() => {
    // Entries + Projects come from the same endpoint
    listKbProjects()
      .then((resp) => {
        setEntries(resp.items.reduce((sum, p) => sum + p.entry_count, 0))
        setProjects(resp.items.length)
      })
      .catch(() => {
        setEntries(null)
        setProjects(null)
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
        {/* Left column: welcome heading + Ask panel (~2/3) */}
        <Grid size={{ xs: 12, md: 8 }}>
          <Typography variant="h5" gutterBottom>
            Welcome to Personal KB
          </Typography>
          <AskPanel />
        </Grid>

        {/* Right column: KB Stats (~1/3) */}
        <Grid size={{ xs: 12, md: 4 }}>
          <StatsCard />
        </Grid>
      </Grid>
    </Box>
  )
}
