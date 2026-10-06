import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ResponseCard, MarkdownBody } from '../components/ResponseCard'

describe('ResponseCard', () => {
  it('renders children inside an element with data-testid="response-card" that has class MuiPaper-root', () => {
    render(<ResponseCard>Hello world</ResponseCard>)
    const card = screen.getByTestId('response-card')
    expect(card).toBeInTheDocument()
    expect(card.className).toContain('MuiPaper-root')
    expect(card).toHaveTextContent('Hello world')
  })

  it('merges caller sx on top of base sx', () => {
    render(<ResponseCard sx={{ maxWidth: '80%' }}>content</ResponseCard>)
    const card = screen.getByTestId('response-card')
    expect(card).toBeInTheDocument()
    expect(card).toHaveTextContent('content')
  })
})

describe('MarkdownBody', () => {
  it('renders bold and code from markdown', () => {
    const { container } = render(<MarkdownBody>{'**bold** and `code`'}</MarkdownBody>)
    const boldEl = container.querySelector('strong')
    const codeEl = container.querySelector('code')
    expect(boldEl).not.toBeNull()
    expect(boldEl?.textContent).toBe('bold')
    expect(codeEl).not.toBeNull()
    expect(codeEl?.textContent).toBe('code')
  })

  it('renders a GFM table', () => {
    const tableMarkdown = `
| Col1 | Col2 |
| ---- | ---- |
| A    | B    |
`
    render(<MarkdownBody>{tableMarkdown}</MarkdownBody>)
    const table = document.querySelector('table')
    expect(table).not.toBeNull()
  })
})
