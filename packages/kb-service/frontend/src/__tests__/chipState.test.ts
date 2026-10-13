import { describe, it, expect } from 'vitest'
import { toggleChipSx } from '../components/chipState'

describe('toggleChipSx', () => {
  it('returns an empty sx object for an ON chip (call sites keep their filled styles)', () => {
    expect(toggleChipSx(true)).toEqual({})
  })

  it('returns the pinned grey sx for an OFF chip', () => {
    expect(toggleChipSx(false)).toEqual({
      bgcolor: 'transparent',
      color: 'text.disabled',
      borderColor: 'text.disabled',
    })
  })
})
