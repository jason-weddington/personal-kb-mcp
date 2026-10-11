import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Layout } from '../components/Layout'

vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(() => ({
    isAuthenticated: true,
    user: { id: 'u1', email: 'user@example.com', isAdmin: false, createdAt: '' },
    authMode: 'jwt',
    loading: false,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  })),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))
vi.mock('../contexts/ThemeContext', () => ({
  useThemeMode: () => ({ mode: 'dark', toggleTheme: vi.fn() }),
}))
vi.mock('../contexts/EntryDrawerContext', () => ({
  useEntryDrawer: vi.fn(() => ({ openEntry: vi.fn(), closeEntry: vi.fn() })),
  EntryDrawerProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))
vi.mock('../components/Sidebar', () => ({ default: () => null }))

describe('Layout accessibility', () => {
  it('names the icon buttons and starts with a skip link to main content', () => {
    render(
      <MemoryRouter>
        <Layout />
      </MemoryRouter>,
    )
    expect(screen.getByRole('button', { name: 'Toggle light/dark theme' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Account' })).toBeInTheDocument()
    const skip = screen.getByRole('link', { name: 'Skip to content' })
    expect(skip).toHaveAttribute('href', '#main-content')
    expect(document.getElementById('main-content')).toBe(screen.getByRole('main'))
    const focusables = Array.from(
      document.querySelectorAll<HTMLElement>('a[href], button'),
    )
    expect(focusables[0]).toBe(skip)
  })

  it('makes the skip link the first Tab stop and focuses #main-content on activation', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <Layout />
      </MemoryRouter>,
    )
    await user.tab()
    const skip = screen.getByRole('link', { name: 'Skip to content' })
    expect(skip).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(document.getElementById('main-content')).toHaveFocus()
  })
})
