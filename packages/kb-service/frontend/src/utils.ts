export function toSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase())
}

export function toCamelCase(str: string): string {
  return str.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())
}

export function convertKeys(
  obj: unknown,
  fn: (key: string) => string,
): unknown {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }
  if (Array.isArray(obj)) {
    return obj.map((item) => convertKeys(item, fn))
  }
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
    result[fn(key)] = convertKeys(value, fn)
  }
  return result
}
