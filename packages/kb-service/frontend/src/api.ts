import { convertKeys, toCamelCase, toSnakeCase } from './utils'
import type { AuthResponse, UserResponse } from './types'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
    this.name = 'ApiError'
  }
}

function getToken(): string | null {
  return localStorage.getItem('kb-token')
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const headers: Record<string, string> = {}
  const token = getToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(`/api${path}`, {
    method,
    headers,
    body:
      body !== undefined
        ? JSON.stringify(convertKeys(body, toSnakeCase))
        : undefined,
  })

  if (res.status === 401) {
    localStorage.removeItem('kb-token')
    localStorage.removeItem('kb-user')
    // kb-01449 guard: never redirect from /login itself
    if (window.location.pathname !== '/login') {
      window.location.href = '/login'
    }
    throw new ApiError(401, 'Unauthorized')
  }

  if (res.status === 204) {
    return undefined as T
  }

  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = (await res.json()) as { detail?: string }
      if (data.detail) detail = data.detail
    } catch {
      // use statusText fallback
    }
    throw new ApiError(res.status, detail)
  }

  return convertKeys(await res.json(), toCamelCase) as T
}

export const api = {
  auth: {
    register: (
      email: string,
      password: string,
      inviteToken: string,
    ): Promise<AuthResponse> =>
      request<AuthResponse>('POST', '/auth/register', {
        email,
        password,
        inviteToken,
      }),
    login: (email: string, password: string): Promise<AuthResponse> =>
      request<AuthResponse>('POST', '/auth/login', { email, password }),
    logout: (): Promise<void> => request<void>('POST', '/auth/logout'),
    me: (): Promise<UserResponse> => request<UserResponse>('GET', '/auth/me'),
  },
}
