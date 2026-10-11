import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GraphSearch } from '../components/GraphSearch'
import type { ExplorerNode } from '../graph/explorerModel'

describe('GraphSearch', () => {
  it('never wraps the id caption in a dropdown option', async () => {
    const nodes: ExplorerNode[] = [
      { id: 'kb-00001', label: 'A very long title for a node', type: 'entry', val: 1, properties: {} },
    ]
    render(<GraphSearch nodes={nodes} onSelectNode={vi.fn()} />)
    await userEvent.setup().type(screen.getByRole('combobox'), 'very')
    const caption = await screen.findByText('kb-00001')
    expect(getComputedStyle(caption).whiteSpace).toBe('nowrap')
  })
})
