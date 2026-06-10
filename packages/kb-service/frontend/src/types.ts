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
