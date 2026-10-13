/**
 * Shared OFF-state styles for toggle chips (the Graph page type chips and the
 * inspector facet chips): a chip that is OFF reads as grey — outlined with a
 * grey border and grey text — never as a coloured outline. The helper takes
 * no colour parameter (an ON chip keeps its own filled styles at the call
 * site; the MUI `variant` stays a Chip prop there, not an sx key).
 */
export function toggleChipSx(on: boolean) {
  if (on) return {}
  return {
    bgcolor: 'transparent',
    color: 'text.disabled',
    borderColor: 'text.disabled',
  }
}
