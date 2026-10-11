import { describe, it, expect } from 'vitest'
import {
  confidenceColor,
  connectionsLabel,
  entriesLabel,
  formatConfidence,
  formatRelative,
} from '../browse/format'

const now = Date.parse('2026-07-01T00:00:00Z')

describe('formatRelative', () => {
  it.each([
    ['2026-06-30T23:59:30+00:00', 'just now'],
    ['2026-06-30T23:58:00Z', '2m ago'],
    ['2026-06-30T23:00:00+00:00', '1h ago'],
    ['2026-06-30T23:00:00', '1h ago'],
    ['2026-06-24T00:00:00+00:00', '7d ago'],
    ['2026-03-01T00:00:00+00:00', '4mo ago'],
    ['2024-07-01T00:00:00+00:00', '2y ago'],
    ['2026-07-02T00:00:00Z', 'just now'],
  ])('%s -> %s', (iso, expected) => {
    expect(formatRelative(iso, now)).toBe(expected)
  })
})

describe('formatConfidence', () => {
  it.each([
    [0.0301, '3%'],
    [0.2472, '25%'],
    [1, '100%'],
  ])('%s -> %s', (x, expected) => {
    expect(formatConfidence(x)).toBe(expected)
  })
})

describe('confidenceColor', () => {
  it.each([
    [0.29, 'error.main'],
    [0.3, 'warning.main'],
    [0.49, 'warning.main'],
    [0.5, 'text.secondary'],
  ])('%s -> %s', (x, expected) => {
    expect(confidenceColor(x)).toBe(expected)
  })
})

describe('labels', () => {
  it('connectionsLabel pluralises', () => {
    expect(connectionsLabel(1)).toBe('1 connection')
    expect(connectionsLabel(5)).toBe('5 connections')
  })

  it('entriesLabel pluralises', () => {
    expect(entriesLabel(1)).toBe('1 entry')
    expect(entriesLabel(3)).toBe('3 entries')
  })
})
