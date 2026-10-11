import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TypeSelect } from '../components/browse/TypeSelect'

describe('TypeSelect', () => {
  it('shrinks the label when no type is chosen so it never overlaps "All types"', () => {
    render(<TypeSelect labelId="t-label" value={undefined} onChange={vi.fn()} />)
    const label = document.getElementById('t-label')
    expect(label).not.toBeNull()
    expect(label).toHaveAttribute('data-shrink', 'true')
    expect(screen.getByText('All types')).toBeInTheDocument()
  })
})
