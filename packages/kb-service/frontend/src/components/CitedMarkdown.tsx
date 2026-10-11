import { useMemo } from 'react'
import Box from '@mui/material/Box'
import Link from '@mui/material/Link'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { MARKDOWN_BODY_SX } from './ResponseCard'
import { CITE_HREF_PREFIX, remarkKbCitations } from '../graph/chatModel'

const PLUGINS = [remarkGfm, remarkKbCitations]

export function CitedMarkdown({
  children,
  onCitation,
}: {
  children: string
  onCitation: (id: string) => void
}) {
  const components = useMemo<Components>(
    () => ({
      a: ({ href, children }) => {
        if (href?.startsWith(CITE_HREF_PREFIX)) {
          const id = href.slice(CITE_HREF_PREFIX.length)
          return (
            <Link
              component="button"
              type="button"
              aria-label={`Show ${id} on the graph`}
              onClick={() => onCitation(id)}
              sx={{ color: 'info.main', verticalAlign: 'baseline', font: 'inherit' }}
            >
              {children}
            </Link>
          )
        }
        return (
          <Link href={href} target="_blank" rel="noopener noreferrer">
            {children}
          </Link>
        )
      },
    }),
    [onCitation],
  )
  return (
    <Box sx={MARKDOWN_BODY_SX}>
      <ReactMarkdown remarkPlugins={PLUGINS} components={components}>
        {children}
      </ReactMarkdown>
    </Box>
  )
}
