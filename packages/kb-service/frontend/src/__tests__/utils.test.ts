import { describe, it, expect } from 'vitest'
import { toSnakeCase, toCamelCase, convertKeys } from '../utils'

describe('toSnakeCase', () => {
  it('converts camelCase to snake_case', () => {
    expect(toSnakeCase('helloWorld')).toBe('hello_world')
    expect(toSnakeCase('isAdmin')).toBe('is_admin')
    expect(toSnakeCase('createdAt')).toBe('created_at')
    expect(toSnakeCase('inviteToken')).toBe('invite_token')
  })

  it('handles empty string', () => {
    expect(toSnakeCase('')).toBe('')
  })

  it('leaves already-snake strings unchanged', () => {
    expect(toSnakeCase('hello_world')).toBe('hello_world')
    expect(toSnakeCase('simple')).toBe('simple')
  })
})

describe('toCamelCase', () => {
  it('converts snake_case to camelCase', () => {
    expect(toCamelCase('hello_world')).toBe('helloWorld')
    expect(toCamelCase('is_admin')).toBe('isAdmin')
    expect(toCamelCase('created_at')).toBe('createdAt')
    expect(toCamelCase('invite_token')).toBe('inviteToken')
  })

  it('handles empty string', () => {
    expect(toCamelCase('')).toBe('')
  })
})

describe('snake->camel->snake roundtrip', () => {
  it('roundtrips correctly', () => {
    const snake = 'is_admin'
    expect(toSnakeCase(toCamelCase(snake))).toBe(snake)
  })
})

describe('convertKeys', () => {
  it('converts top-level keys', () => {
    const result = convertKeys({ helloWorld: 1 }, toSnakeCase)
    expect(result).toEqual({ hello_world: 1 })
  })

  it('converts nested keys', () => {
    const result = convertKeys(
      { outerKey: { innerKey: 'value' } },
      toSnakeCase,
    )
    expect(result).toEqual({ outer_key: { inner_key: 'value' } })
  })

  it('converts keys in arrays of objects', () => {
    const result = convertKeys([{ myKey: 1 }, { anotherKey: 2 }], toSnakeCase)
    expect(result).toEqual([{ my_key: 1 }, { another_key: 2 }])
  })

  it('passes through null unchanged', () => {
    expect(convertKeys(null, toSnakeCase)).toBeNull()
  })

  it('passes through primitives unchanged', () => {
    expect(convertKeys(42, toSnakeCase)).toBe(42)
    expect(convertKeys('hello', toSnakeCase)).toBe('hello')
    expect(convertKeys(true, toSnakeCase)).toBe(true)
  })
})
