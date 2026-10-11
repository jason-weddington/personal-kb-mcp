import { Fragment } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import List from '@mui/material/List'
import ListSubheader from '@mui/material/ListSubheader'
import Skeleton from '@mui/material/Skeleton'
import Typography from '@mui/material/Typography'
import type { BrowseEntriesResponse, BrowseEntry } from '../../kbTypes'
import { useEntryDrawer } from '../../contexts/EntryDrawerContext'
import { usePagedList } from '../../browse/usePagedList'
import { entriesLabel } from '../../browse/format'
import { BrowseRow } from './BrowseRow'

export interface EntryListProps {
  listKey: string
  fetchPage: (cursor: string | null) => Promise<BrowseEntriesResponse>
  emptyText: string
  ageLabelFor: (row: BrowseEntry, fetchedAt: number) => string
  metaFor?: (row: BrowseEntry, fetchedAt: number) => string | null
  sectionFor?: (row: BrowseEntry, loaded: BrowseEntry[]) => string | null
  renderError?: (error: unknown, retry: () => void) => ReactNode | null
}

const entryKey = (r: BrowseEntry) => r.id

export function BrowseSkeletons({ count }: { count: number }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <Skeleton
          key={i}
          data-testid="browse-skeleton"
          variant="rounded"
          height={56}
          sx={{ my: 0.5 }}
        />
      ))}
    </>
  )
}

/** Default error block: message plus a Retry button. */
export function LoadError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <Alert
      severity="error"
      action={
        <Button color="inherit" size="small" onClick={onRetry}>
          Retry
        </Button>
      }
    >
      Could not load: {message}
    </Alert>
  )
}

function moveFocus(e: KeyboardEvent<HTMLElement>) {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
  const rows = Array.from(
    e.currentTarget.querySelectorAll<HTMLElement>('[data-testid="browse-row"]'),
  )
  const index = rows.findIndex((r) => r === document.activeElement)
  if (index === -1) return
  const next = e.key === 'ArrowDown' ? index + 1 : index - 1
  if (next < 0 || next >= rows.length) return
  e.preventDefault()
  rows[next].focus()
}

/** Per row: the subheader to show before it, or null. */
function sectionHeaders(
  items: BrowseEntry[],
  sectionFor: EntryListProps['sectionFor'],
): (string | null)[] {
  const headers: (string | null)[] = []
  let previous: string | null = null
  for (const row of items) {
    const section = sectionFor ? sectionFor(row, items) : null
    headers.push(section !== null && section !== previous ? section : null)
    previous = section
  }
  return headers
}

/** A paged list of browse entries; rows open the entry drawer. */
export function EntryList({
  listKey,
  fetchPage,
  emptyText,
  ageLabelFor,
  metaFor,
  sectionFor,
  renderError,
}: EntryListProps) {
  const { openEntry } = useEntryDrawer()
  const list = usePagedList(listKey, fetchPage, entryKey)

  if (list.status === 'loading') {
    return <BrowseSkeletons count={8} />
  }

  if (list.status === 'error') {
    const custom = renderError?.(list.error, list.retry)
    if (custom !== undefined && custom !== null) return <>{custom}</>
    return <LoadError message={list.errorMessage ?? ''} onRetry={list.retry} />
  }

  if (list.items.length === 0) {
    return (
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ textAlign: 'center', py: 4 }}
      >
        {emptyText}
      </Typography>
    )
  }

  const fetchedAt = list.fetchedAt ?? 0
  const headers = sectionHeaders(list.items, sectionFor)

  return (
    <Box>
      <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}>
        {entriesLabel(list.total)}
        {list.skipped > 0 && ` · ${list.skipped} unreadable`}
      </Typography>
      <List dense disablePadding onKeyDown={moveFocus}>
        {list.items.map((row, i) => {
          const header = headers[i]
          return (
            <Fragment key={row.id}>
              {header !== null && <ListSubheader disableSticky>{header}</ListSubheader>}
              <BrowseRow
                row={row}
                ageLabel={ageLabelFor(row, fetchedAt)}
                meta={metaFor ? metaFor(row, fetchedAt) : null}
                onOpen={openEntry}
              />
            </Fragment>
          )
        })}
      </List>
      {list.loadingMore && <BrowseSkeletons count={3} />}
      {list.nextCursor !== null && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
          <Button onClick={list.loadMore} disabled={list.loadingMore}>
            Load more
          </Button>
        </Box>
      )}
    </Box>
  )
}
