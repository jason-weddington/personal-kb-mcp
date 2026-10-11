import type { EntryType } from './kbTypes'

/** Entry types in kb-core EntryType declaration order. */
export const ENTRY_TYPES: EntryType[] = [
  'factual_reference',
  'decision',
  'pattern_convention',
  'lesson_learned',
  'mental_map',
]

export type ChipColor =
  | 'default'
  | 'primary'
  | 'secondary'
  | 'error'
  | 'info'
  | 'success'
  | 'warning'

export const ENTRY_TYPE_COLORS: Record<EntryType, ChipColor> = {
  factual_reference: 'info',
  decision: 'primary',
  pattern_convention: 'secondary',
  lesson_learned: 'warning',
  mental_map: 'success',
}

/** Human label for an entry type: underscores become spaces. */
export function entryTypeLabel(t: string): string {
  return t.replace(/_/g, ' ')
}
