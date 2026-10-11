import { useState } from 'react'
import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Typography from '@mui/material/Typography'
import FiberNewIcon from '@mui/icons-material/FiberNew'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import HourglassBottomIcon from '@mui/icons-material/HourglassBottom'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import { useSearchParams } from 'react-router-dom'
import { browseRecent, browseStale, browseTop } from '../../api'
import type { BrowseEntry, EntryType } from '../../kbTypes'
import {
  STALE_FILTER_THRESHOLD,
  connectionsLabel,
  formatConfidence,
  formatRelative,
} from '../../browse/format'
import { EntryList } from './EntryList'
import { ProjectGrid } from './ProjectGrid'
import { ProjectView } from './ProjectView'
import { TypeSelect, parseEntryType } from './TypeSelect'

type View = 'new' | 'top' | 'stale' | 'projects'
type RecentSort = 'created' | 'updated'
type TopSignal = 'degree' | 'read'
type StaleKind = 'decayed' | 'expired'

const VIEWS: View[] = ['new', 'top', 'stale', 'projects']
const PAGE_SIZE = 25
const TYPE_FILTER_EMPTY = 'No entries match this type filter.'

function parseView(raw: string | null): View {
  return VIEWS.includes(raw as View) ? (raw as View) : 'new'
}

const createdAge = (r: BrowseEntry, now: number) => formatRelative(r.created_at, now)
const updatedAge = (r: BrowseEntry, now: number) =>
  `updated ${formatRelative(r.updated_at, now)}`

/** Home's Browse panel: New / Top / Stale / Projects tabs with URL state. */
export function BrowsePanel() {
  const [params, setSearchParams] = useSearchParams()
  const view = parseView(params.get('view'))
  const entryType = parseEntryType(params.get('type'))
  const project = view === 'projects' ? params.get('project') : null

  const [recentSort, setRecentSort] = useState<RecentSort>('created')
  const [topSignal, setTopSignal] = useState<TopSignal>('degree')
  const [staleKind, setStaleKind] = useState<StaleKind>('decayed')

  const changeView = (next: View) => {
    const p = new URLSearchParams(params)
    p.delete('project')
    if (next === 'new') p.delete('view')
    else p.set('view', next)
    setSearchParams(p)
  }

  const changeType = (t: EntryType | undefined) => {
    const p = new URLSearchParams(params)
    if (t) p.set('type', t)
    else p.delete('type')
    setSearchParams(p)
  }

  const openProject = (ref: string | null) => {
    const p = new URLSearchParams(params)
    if (ref === null) p.delete('project')
    else p.set('project', ref)
    setSearchParams(p)
  }

  const pct = formatConfidence(STALE_FILTER_THRESHOLD)
  const typeKey = entryType ?? ''

  let toggle = null
  let caption: string | null = null
  let list = null

  if (view === 'new') {
    toggle = (
      <ToggleButtonGroup
        size="small"
        exclusive
        value={recentSort}
        aria-label="New entries order"
        onChange={(_, v: RecentSort | null) => v && setRecentSort(v)}
      >
        <ToggleButton value="created">Created</ToggleButton>
        <ToggleButton value="updated">Updated</ToggleButton>
      </ToggleButtonGroup>
    )
    list = (
      <EntryList
        listKey={`new:${recentSort}:${typeKey}`}
        fetchPage={(cursor) =>
          browseRecent({ sort: recentSort, entryType, limit: PAGE_SIZE, cursor })
        }
        ageLabelFor={recentSort === 'updated' ? updatedAge : createdAge}
        emptyText={
          entryType
            ? TYPE_FILTER_EMPTY
            : recentSort === 'updated'
              ? 'No entries have been updated since they were created.'
              : 'No entries yet.'
        }
      />
    )
  } else if (view === 'top') {
    toggle = (
      <ToggleButtonGroup
        size="small"
        exclusive
        value={topSignal}
        aria-label="Top signal"
        onChange={(_, v: TopSignal | null) => v && setTopSignal(v)}
      >
        <ToggleButton value="degree">Most connected</ToggleButton>
        <ToggleButton value="read">Recently read</ToggleButton>
      </ToggleButtonGroup>
    )
    caption =
      topSignal === 'degree'
        ? 'Ranked by graph connections (edges to tags, projects, tools and other entries). The KB records no per-entry retrieval counts.'
        : 'Ranked by last_accessed: the last kb_get by an agent, or an entry opened here.'
    list = (
      <EntryList
        listKey={`top:${topSignal}:${typeKey}`}
        fetchPage={(cursor) =>
          browseTop({ signal: topSignal, entryType, limit: PAGE_SIZE, cursor })
        }
        ageLabelFor={createdAge}
        metaFor={(r, now) =>
          topSignal === 'degree'
            ? connectionsLabel(r.signal_value ?? 0)
            : r.last_accessed
              ? `read ${formatRelative(r.last_accessed, now)}`
              : null
        }
        emptyText={
          entryType
            ? TYPE_FILTER_EMPTY
            : topSignal === 'degree'
              ? 'No entries have graph connections yet.'
              : 'No entry has been opened with kb_get yet.'
        }
      />
    )
  } else if (view === 'stale') {
    toggle = (
      <ToggleButtonGroup
        size="small"
        exclusive
        value={staleKind}
        aria-label="Stale kind"
        onChange={(_, v: StaleKind | null) => v && setStaleKind(v)}
      >
        <ToggleButton value="decayed">Decayed</ToggleButton>
        <ToggleButton value="expired">Expired</ToggleButton>
      </ToggleButtonGroup>
    )
    caption =
      staleKind === 'decayed'
        ? `Effective confidence below ${pct} after decay: hidden from kb_search unless include_stale is set. Opening an entry counts as a read and resets its decay clock.`
        : 'TTL has passed: hidden from kb_search unless include_expired is set.'
    list = (
      <EntryList
        listKey={`stale:${staleKind}:${typeKey}`}
        fetchPage={(cursor) =>
          browseStale({ kind: staleKind, entryType, limit: PAGE_SIZE, cursor })
        }
        ageLabelFor={createdAge}
        metaFor={(r, now) => {
          if (staleKind === 'expired') {
            return r.expires_at ? `expired ${formatRelative(r.expires_at, now)}` : null
          }
          if (r.half_life_days === null) return 'never decays'
          return `half-life ${r.half_life_days}d · untouched ${Math.round(r.anchor_age_days)}d`
        }}
        emptyText={
          entryType
            ? TYPE_FILTER_EMPTY
            : staleKind === 'decayed'
              ? `Nothing is stale: every entry is at or above ${pct} effective confidence.`
              : 'No expired entries.'
        }
      />
    )
  } else if (project) {
    list = (
      <ProjectView
        key={project}
        project={project}
        entryType={entryType}
        onTypeChange={changeType}
        onBack={() => openProject(null)}
      />
    )
  } else {
    list = <ProjectGrid onOpen={openProject} />
  }

  return (
    <Box>
      <Tabs
        value={view}
        onChange={(_, v: View) => changeView(v)}
        aria-label="Browse views"
        variant="scrollable"
        allowScrollButtonsMobile
        sx={{ mb: 2 }}
      >
        <Tab value="new" label="New" icon={<FiberNewIcon />} iconPosition="start" />
        <Tab value="top" label="Top" icon={<TrendingUpIcon />} iconPosition="start" />
        <Tab
          value="stale"
          label="Stale"
          icon={<HourglassBottomIcon />}
          iconPosition="start"
        />
        <Tab
          value="projects"
          label="Projects"
          icon={<FolderOpenIcon />}
          iconPosition="start"
        />
      </Tabs>
      {view !== 'projects' && (
        <Stack
          direction="row"
          spacing={2}
          useFlexGap
          sx={{ flexWrap: 'wrap', alignItems: 'center', mb: 1 }}
        >
          {toggle}
          <TypeSelect labelId="browse-type-label" value={entryType} onChange={changeType} />
        </Stack>
      )}
      {caption && (
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}>
          {caption}
        </Typography>
      )}
      {list}
    </Box>
  )
}
