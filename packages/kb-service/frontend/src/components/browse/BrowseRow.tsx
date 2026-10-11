import Chip from '@mui/material/Chip'
import ListItemButton from '@mui/material/ListItemButton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import type { BrowseEntry } from '../../kbTypes'
import { ENTRY_TYPE_COLORS, entryTypeLabel } from '../../entryTypes'
import { confidenceColor, formatConfidence } from '../../browse/format'

interface BrowseRowProps {
  row: BrowseEntry
  meta?: string | null
  ageLabel: string
  onOpen: (id: string) => void
}

/** One entry in a browse list. Click or Enter opens the entry drawer. */
export function BrowseRow({ row, meta, ageLabel, onOpen }: BrowseRowProps) {
  return (
    <ListItemButton
      data-testid="browse-row"
      data-entry-id={row.id}
      onClick={() => onOpen(row.id)}
      sx={{ alignItems: 'flex-start', gap: 2, borderRadius: 1 }}
    >
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography variant="body1" noWrap title={row.long_title}>
          {row.short_title}
        </Typography>
        <Stack
          direction="row"
          spacing={1}
          useFlexGap
          sx={{ flexWrap: 'wrap', alignItems: 'center', mt: 0.5 }}
        >
          <Chip
            size="small"
            label={entryTypeLabel(row.entry_type)}
            color={ENTRY_TYPE_COLORS[row.entry_type] ?? 'default'}
          />
          <Typography variant="caption" color="text.secondary">
            {row.project_ref ?? 'no project'}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {ageLabel}
          </Typography>
          <Typography
            variant="caption"
            color={confidenceColor(row.effective_confidence)}
            title={`Effective confidence (base ${formatConfidence(row.confidence_level)})`}
          >
            {formatConfidence(row.effective_confidence)}
          </Typography>
          {row.expired && (
            <Chip size="small" variant="outlined" color="error" label="Expired" />
          )}
        </Stack>
      </Box>
      {meta && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ flexShrink: 0, textAlign: 'right', mt: 0.5 }}
        >
          {meta}
        </Typography>
      )}
    </ListItemButton>
  )
}
