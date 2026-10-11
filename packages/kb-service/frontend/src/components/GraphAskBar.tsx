import { useRef, useState } from 'react'
import Box from '@mui/material/Box'
import CircularProgress from '@mui/material/CircularProgress'
import IconButton from '@mui/material/IconButton'
import InputAdornment from '@mui/material/InputAdornment'
import TextField from '@mui/material/TextField'
import SendIcon from '@mui/icons-material/Send'

export interface GraphAskBarProps {
  onAsk: (question: string) => void
  busy: boolean
}

export function GraphAskBar({ onAsk, busy }: GraphAskBarProps) {
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <Box
      component="form"
      sx={{ width: { xs: '100%', sm: 320 } }}
      onSubmit={(e: React.FormEvent) => {
        e.preventDefault()
        const trimmed = value.trim()
        if (trimmed !== '') {
          onAsk(trimmed)
          setValue('')
        }
      }}
    >
      <TextField
        size="small"
        fullWidth
        placeholder="Ask the KB a question…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            setValue('')
            inputRef.current?.blur()
          }
        }}
        inputRef={inputRef}
        slotProps={{
          htmlInput: { 'aria-label': 'Ask the knowledge base' },
          input: {
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  type="submit"
                  aria-label="Ask"
                  size="small"
                  edge="end"
                  disabled={value.trim() === ''}
                >
                  {busy ? <CircularProgress size={16} /> : <SendIcon fontSize="small" />}
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />
    </Box>
  )
}
