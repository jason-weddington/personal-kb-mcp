// Client-side node search for the Graph page (old explorer search bar).
import { useMemo, useState } from 'react'
import Autocomplete from '@mui/material/Autocomplete'
import Box from '@mui/material/Box'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { SEARCH_LIMIT, nodeColor, rankSearch } from '../graph/explorerModel'
import type { ExplorerNode } from '../graph/explorerModel'

export interface GraphSearchProps {
  nodes: ExplorerNode[]
  onSelectNode: (id: string) => void
  /** Enter on free text with nothing highlighted. */
  onFreeTextSubmit?: (text: string) => void
  placeholder?: string
}

export function GraphSearch({
  nodes,
  onSelectNode,
  onFreeTextSubmit,
  placeholder = 'Search nodes...',
}: GraphSearchProps) {
  const [inputValue, setInputValue] = useState('')
  const [open, setOpen] = useState(false)
  const options = useMemo(() => rankSearch(nodes, inputValue, SEARCH_LIMIT), [nodes, inputValue])

  return (
    <Autocomplete<ExplorerNode, false, false, true>
      freeSolo
      value={null}
      inputValue={inputValue}
      open={open}
      onOpen={() => setOpen(true)}
      onClose={() => setOpen(false)}
      onInputChange={(_e, v) => setInputValue(v)}
      options={options}
      filterOptions={(o) => o}
      getOptionLabel={(o) => (typeof o === 'string' ? o : o.label)}
      onChange={(_e, value, reason) => {
        if (reason === 'selectOption' && value !== null && typeof value !== 'string') {
          onSelectNode(value.id)
        } else if (reason === 'createOption' && typeof value === 'string') {
          if (onFreeTextSubmit) onFreeTextSubmit(value.trim())
          else if (options.length > 0) onSelectNode(options[0].id)
        } else {
          return
        }
        setInputValue('')
      }}
      onKeyDown={(e) => {
        if (e.key !== 'Escape') return
        ;(e as unknown as { defaultMuiPrevented?: boolean }).defaultMuiPrevented = true
        setInputValue('')
        setOpen(false)
        ;(e.target as HTMLElement).blur()
      }}
      renderOption={(props, o) => {
        const { key, ...rest } = props
        return (
          <Box component="li" key={key} {...rest} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Box
              component="span"
              sx={{ width: 10, height: 10, borderRadius: '50%', flexShrink: 0, bgcolor: nodeColor(o.type) }}
            />
            <Typography component="span" variant="body2" sx={{ flexGrow: 1 }}>
              {o.label}
            </Typography>
            {o.id !== o.label && (
              <Typography component="span" variant="caption" color="text.secondary">
                {o.id}
              </Typography>
            )}
          </Box>
        )
      }}
      renderInput={(params) => (
        <TextField {...params} size="small" placeholder={placeholder} aria-label="Search nodes" />
      )}
      sx={{ width: { xs: '100%', sm: 320 } }}
    />
  )
}
