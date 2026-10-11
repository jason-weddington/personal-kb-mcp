import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import type { EntryType } from '../../kbTypes'
import { ENTRY_TYPES, entryTypeLabel } from '../../entryTypes'

/** An EntryType for a known value, else undefined (no filter). */
export function parseEntryType(raw: string | null): EntryType | undefined {
  return ENTRY_TYPES.includes(raw as EntryType) ? (raw as EntryType) : undefined
}

/** Type filter select shared by the entry views and the project view. */
export function TypeSelect({
  labelId,
  value,
  onChange,
}: {
  labelId: string
  value: EntryType | undefined
  onChange: (value: EntryType | undefined) => void
}) {
  return (
    <FormControl size="small" sx={{ minWidth: 180 }}>
      <InputLabel id={labelId} shrink>Type</InputLabel>
      <Select
        labelId={labelId}
        label="Type"
        value={value ?? ''}
        displayEmpty
        notched
        SelectDisplayProps={{ 'aria-labelledby': labelId }}
        onChange={(e) => onChange(parseEntryType(e.target.value))}
      >
        <MenuItem value="">All types</MenuItem>
        {ENTRY_TYPES.map((t) => (
          <MenuItem key={t} value={t}>
            {entryTypeLabel(t)}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  )
}
