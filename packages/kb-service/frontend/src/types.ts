export interface UserResponse {
  id: string
  email: string
  isAdmin: boolean
  createdAt: string
}

export interface AuthResponse {
  token: string
  user: UserResponse
}

export interface ApiKeyInfo {
  id: string
  name: string
  hashPrefix: string
  createdAt: string
}

export interface ApiKeyCreated {
  apiKey: string
  name: string
}

export interface ApiKeysResponse {
  keys: ApiKeyInfo[]
}

export interface Invite {
  token: string
  issuedBy: string
  note: string | null
  createdAt: string
  usedAt: string | null
  usedBy: string | null
}

export interface CreatedInvite {
  token: string
  url: string
  note: string | null
  createdAt: string
}

export interface PasswordResetIssued {
  token: string
  url: string
  expiresAt: string
}

export interface Settings {
  team: string | null
}

export interface AdminUser {
  id: string
  email: string
  isAdmin: boolean
  createdAt: string
}
