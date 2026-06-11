import Paper from '@mui/material/Paper'
import Box from '@mui/material/Box'
import type { SxProps, Theme } from '@mui/material/styles'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const BASE_SX: SxProps<Theme> = {
  px: 2,
  py: 1,
  bgcolor: 'action.hover',
  color: 'text.primary',
  borderRadius: 2,
}

export function ResponseCard({
  children,
  sx,
}: {
  children: React.ReactNode
  sx?: SxProps<Theme>
}) {
  return (
    <Paper
      elevation={0}
      data-testid="response-card"
      sx={[BASE_SX, ...(Array.isArray(sx) ? sx : [sx ?? false])]}
    >
      {children}
    </Paper>
  )
}

export function MarkdownBody({ children }: { children: string }) {
  return (
    <Box
      sx={{
        '& p': { mt: 0, mb: 1 },
        '& p:last-child': { mb: 0 },
        '& pre': { overflowX: 'auto', bgcolor: 'action.selected', p: 1, borderRadius: 1 },
        '& code': { fontFamily: 'monospace', fontSize: '0.875em' },
      }}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </Box>
  )
}
