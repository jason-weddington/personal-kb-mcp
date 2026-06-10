import { convertKeys, toCamelCase, toSnakeCase } from './utils'
import type {
  AuthResponse,
  UserResponse,
  ApiKeyCreated,
  ApiKeysResponse,
  Invite,
  CreatedInvite,
  PasswordResetIssued,
  Settings,
  AdminUser,
} from './types'

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
    changePassword: (
      currentPassword: string,
      newPassword: string,
    ): Promise<void> =>
      request<void>('POST', '/auth/password', { currentPassword, newPassword }),
    resetPassword: (token: string, newPassword: string): Promise<void> =>
      request<void>('POST', '/auth/password-reset', { token, newPassword }),
  },
  settings: {
    get: (): Promise<Settings> => request<Settings>('GET', '/settings'),
    put: (team: string): Promise<Settings> =>
      request<Settings>('PUT', '/settings', { team }),
  },
  apiKeys: {
    list: (): Promise<ApiKeysResponse> =>
      request<ApiKeysResponse>('GET', '/auth/api-keys'),
    create: (name: string): Promise<ApiKeyCreated> =>
      request<ApiKeyCreated>('POST', '/auth/api-keys', { name }),
    revoke: (id: string): Promise<void> =>
      request<void>('DELETE', `/auth/api-keys/${id}`),
  },
  admin: {
    invites: {
      list: (): Promise<Invite[]> =>
        request<Invite[]>('GET', '/admin/invites'),
      create: (note: string): Promise<CreatedInvite> =>
        request<CreatedInvite>('POST', '/admin/invites', { note }),
      revoke: (token: string): Promise<void> =>
        request<void>('DELETE', `/admin/invites/${token}`),
    },
    users: {
      list: (): Promise<AdminUser[]> =>
        request<AdminUser[]>('GET', '/admin/users'),
      promote: (id: string): Promise<AdminUser> =>
        request<AdminUser>('POST', `/admin/users/${id}/promote`),
      issuePasswordReset: (id: string): Promise<PasswordResetIssued> =>
        request<PasswordResetIssued>('POST', `/admin/users/${id}/password-reset`),
      deleteUser: (id: string): Promise<void> =>
        request<void>('DELETE', `/admin/users/${id}`),
    },
  },
}
