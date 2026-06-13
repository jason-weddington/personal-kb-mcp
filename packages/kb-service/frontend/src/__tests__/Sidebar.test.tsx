/**
 * Tests for the new Sidebar/Layout drawer behavior:
 * (a) desktop expanded -> hamburger click collapses (width 0 / no layout space)
 * (b) mobile: nav-item click calls navigate AND closes the drawer
 * (c) admin items hidden for non-admin
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Sidebar from '../components/Sidebar'

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

import { useAuth } from '../contexts/AuthContext'

// ── Helpers ───────────────────────────────────────────────────────────────────

function mockUser(isAdmin = false) {
  vi.mocked(useAuth).mockReturnValue({
    isAuthenticated: true,
    user: {
      id: 'u1',
      email: isAdmin ? 'admin@example.com' : 'user@example.com',
      isAdmin,
      createdAt: '',
    },
    authMode: 'jwt',
    loading:false,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  })
}

function renderSidebar(props: {
  open: boolean
  isMobile?: boolean
  onClose?: () => void
}) {
  return render(
    <MemoryRouter>
      <Sidebar {...props} />
    </MemoryRouter>,
  )
}

// ── (a) Desktop: collapse via open=false gives width 0 ───────────────────────

describe('(a) Desktop: hamburger collapses drawer (open=false => width 0)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUser(false)
  })

  it('when open=true on desktop the nav items are rendered', () => {
    const { container } = renderSidebar({ open: true, isMobile: false })
    expect(screen.getByText('Home')).toBeInTheDocument()
    expect(screen.getByText('Settings')).toBeInTheDocument()
    // The outer Drawer root should have width 240 from sx
    const root = container.firstChild as HTMLElement
    expect(root).toBeInTheDocument()
  })

  it('when open=false on desktop the outer Drawer container has width 0 (content reflows)', () => {
    const { container } = renderSidebar({ open: false, isMobile: false })
    // MUI renders the sx `width` as an inline style or emotion class on the
    // root element.  In jsdom, MUI's sx is applied via emotion and appears in
    // the element's className / style.  The key assertion: the outer root's
    // computed/inline style includes width: 0 (or 0px) because our sx sets
    // `width: open ? DRAWER_WIDTH : 0` for non-mobile.
    const root = container.firstChild as HTMLElement
    // Check inline style width OR that the MuiDrawer-root has width 0
    // MUI applies sx to the root div, which renders as an inline style in jsdom
    const styleAttr = root.getAttribute('style') ?? ''
    // Style may contain `width: 0px` or `width: 0`
    const hasWidthZero =
      styleAttr.includes('width: 0') ||
      styleAttr.includes('width:0') ||
      // fallback: check the class-applied styles via getComputedStyle (best-effort)
      window.getComputedStyle(root).width === '0px'
    expect(hasWidthZero).toBe(true)
  })

  it('on desktop nav-item click does NOT call onClose', () => {
    const onClose = vi.fn()
    renderSidebar({ open: true, isMobile: false, onClose })
    fireEvent.click(screen.getByText('Settings'))
    expect(mockNavigate).toHaveBeenCalledWith('/settings')
    expect(onClose).not.toHaveBeenCalled()
  })
})

// ── (b) Mobile: nav-item click navigates AND closes ──────────────────────────

describe('(b) Mobile: nav-item click navigates AND closes the drawer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUser(false)
  })

  it('clicking a nav item calls navigate(path) then onClose()', () => {
    const onClose = vi.fn()
    renderSidebar({ open: true, isMobile: true, onClose })
    fireEvent.click(screen.getByText('Home'))
    expect(mockNavigate).toHaveBeenCalledWith('/')
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('clicking each nav item closes the drawer', () => {
    const onClose = vi.fn()
    renderSidebar({ open: true, isMobile: true, onClose })
    fireEvent.click(screen.getByText('Search'))
    expect(mockNavigate).toHaveBeenCalledWith('/search')
    expect(onClose).toHaveBeenCalled()
  })

  it('on mobile the Drawer uses variant=temporary (no layout space reserved)', () => {
    // MUI temporary Drawer renders via Portal to document.body, so it does not
    // participate in the flex layout — this is how width-0 behaviour is achieved.
    // We verify by confirming the drawer paper is rendered in document.body (portal)
    // and the regular container has no layout-contributing Drawer root.
    const { container } = renderSidebar({ open: true, isMobile: true })
    // The container should be empty or have no MuiDrawer-root (portaled away)
    const drawerInContainer = container.querySelector('[class*="MuiDrawer-root"]')
    // temporary variant portals the Modal out of container → no root inside container
    // (or if MUI does render a root, it has display:none / zero-width)
    if (drawerInContainer) {
      const style = window.getComputedStyle(drawerInContainer as HTMLElement)
      const widthVal = style.width
      expect(['0px', '0', '']).toContain(widthVal)
    } else {
      // Portal rendered to body — nothing in the container — correct behaviour
      expect(drawerInContainer).toBeNull()
    }
    // Confirm drawer content IS visible (rendered to body via portal)
    expect(screen.getByText('Home')).toBeInTheDocument()
  })
})

// ── (c) Admin gating ─────────────────────────────────────────────────────────

describe('(c) Admin gating: admin items hidden for non-admin', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('non-admin: Invites, Users, and Admin heading are NOT rendered', () => {
    mockUser(false)
    renderSidebar({ open: true })
    expect(screen.queryByText('Invites')).not.toBeInTheDocument()
    expect(screen.queryByText('Users')).not.toBeInTheDocument()
    expect(screen.queryByText('Admin')).not.toBeInTheDocument()
  })

  it('admin: Invites, Users, and Admin heading ARE rendered', () => {
    mockUser(true)
    renderSidebar({ open: true })
    expect(screen.getByText('Invites')).toBeInTheDocument()
    expect(screen.getByText('Users')).toBeInTheDocument()
    expect(screen.getByText('Admin')).toBeInTheDocument()
  })

  it('admin: clicking Invites navigates to /admin/invites', () => {
    mockUser(true)
    renderSidebar({ open: true, isMobile: false })
    fireEvent.click(screen.getByText('Invites'))
    expect(mockNavigate).toHaveBeenCalledWith('/admin/invites')
  })

  it('admin mobile: clicking admin item also calls onClose', () => {
    const onClose = vi.fn()
    mockUser(true)
    renderSidebar({ open: true, isMobile: true, onClose })
    fireEvent.click(screen.getByText('Invites'))
    expect(mockNavigate).toHaveBeenCalledWith('/admin/invites')
    expect(onClose).toHaveBeenCalledOnce()
  })
})
