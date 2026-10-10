import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Settings, __testing } from '../pages/Settings'

// Mock AuthContext
vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

// Mock api module
vi.mock('../api', () => ({
  api: {
    auth: {
      changePassword: vi.fn(),
    },
    apiKeys: {
      list: vi.fn(),
      create: vi.fn(),
      revoke: vi.fn(),
    },
    settings: {
      get: vi.fn(),
      put: vi.fn(),
    },
  },
  ApiError: class ApiError extends Error {
    status: number
    detail: string
    constructor(status: number, detail: string) {
      super(detail)
      this.status = status
      this.detail = detail
      this.name = 'ApiError'
    }
  },
}))

import { useAuth } from '../contexts/AuthContext'
import { api } from '../api'

function mockAdminUser() {
  vi.mocked(useAuth).mockReturnValue({
    isAuthenticated: true,
    user: { id: 'u1', email: 'admin@example.com', isAdmin: true, createdAt: '' },
    authMode: 'jwt',
    loading:false,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  })
}

function mockNonAdminUser() {
  vi.mocked(useAuth).mockReturnValue({
    isAuthenticated: true,
    user: { id: 'u2', email: 'user@example.com', isAdmin: false, createdAt: '' },
    authMode: 'jwt',
    loading:false,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  })
}

function renderSettings() {
  return render(
    <MemoryRouter>
      <Settings />
    </MemoryRouter>,
  )
}

describe('Settings page — API Access card (create-key flow)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.apiKeys.list).mockResolvedValue({ keys: [] })
    vi.mocked(api.settings.get).mockResolvedValue({ team: null })
    mockAdminUser()
  })

  it('create-key success dialog renders plaintext key and MCP snippet', async () => {
    const user = userEvent.setup()
    const plainKey = 'sk-test-plaintext-key-12345'
    vi.mocked(api.apiKeys.create).mockResolvedValue({
      apiKey: plainKey,
      name: 'My Key',
    })
    // reload after creation
    vi.mocked(api.apiKeys.list)
      .mockResolvedValueOnce({ keys: [] })
      .mockResolvedValue({
        keys: [{ id: 'k1', name: 'My Key', hashPrefix: 'sk-test', createdAt: '' }],
      })

    renderSettings()

    // Wait for keys to load
    await waitFor(() =>
      expect(vi.mocked(api.apiKeys.list)).toHaveBeenCalled(),
    )

    // Open create dialog
    const createBtn = screen.getByRole('button', { name: /create api key/i })
    await user.click(createBtn)

    // Dialog should be open
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByLabelText(/key name/i)).toBeInTheDocument()

    // Fill in name
    await user.type(within(dialog).getByLabelText(/key name/i), 'My Key')

    // Click Create
    const confirmBtn = within(dialog).getByRole('button', { name: /^create$/i })
    await user.click(confirmBtn)

    // The "API Key Created" dialog should appear
    await waitFor(() =>
      expect(screen.getByText(/api key created/i)).toBeInTheDocument(),
    )

    const createdDialog = screen.getByRole('dialog')

    // Plaintext key should appear
    const inputs = within(createdDialog).getAllByRole('textbox')
    const keyInput = inputs.find((el) =>
      (el as HTMLInputElement).value === plainKey,
    )
    expect(keyInput).toBeDefined()

    // MCP snippet should contain the required strings
    const snippetInput = inputs.find(
      (el) =>
        (el as HTMLInputElement).value.includes('personal-kb') &&
        (el as HTMLInputElement).value.includes('PERSONAL_KB_API_KEY'),
    )
    expect(snippetInput).toBeDefined()
    const snippetValue = (snippetInput as HTMLInputElement).value
    expect(snippetValue).toContain(plainKey)
    // Uses uvx --from (the package is not on PyPI) with the canonical entry
    // point `personal-kb` — not the legacy/wrong `personal-kb-mcp`.
    expect(snippetValue).toContain('"uvx"')
    expect(snippetValue).toContain('"--from"')
    expect(snippetValue).toContain('personal-kb')
    expect(snippetValue).toContain('"personal-kb"')
    // The dead `MCP_TOOL_PREFIX` env (read by nothing) must be gone.
    expect(snippetValue).not.toContain('MCP_TOOL_PREFIX')
    // team=null default → KB_INSTANCE_ROLE must be absent (kb_* tools).
    expect(snippetValue).not.toContain('KB_INSTANCE_ROLE')

    // Primary: /mcp HTTP command + JSON, stdio labelled deprecated.
    const mcp = `${window.location.origin}/mcp`
    const values = inputs.map((el) => (el as HTMLInputElement).value)
    expect(values).toContain(
      `claude mcp add --transport http personal-kb ${mcp} --header "Authorization: Bearer ${plainKey}"`,
    )
    const jsonVal = values.find((v) => v.includes('"type": "http"'))
    expect(jsonVal).toBeDefined()
    expect(jsonVal).toContain(`"url": "${mcp}"`)
    expect(jsonVal).toContain(`"Authorization": "Bearer ${plainKey}"`)
    expect(within(createdDialog).getByText(/local stdio \(deprecated\)/i)).toBeInTheDocument()
  })

  it('team set → KB_INSTANCE_ROLE=team is emitted in the snippet', async () => {
    const user = userEvent.setup()
    const plainKey = 'sk-team-key-9999'
    vi.mocked(api.settings.get).mockResolvedValue({ team: 'Acme' })
    vi.mocked(api.apiKeys.create).mockResolvedValue({
      apiKey: plainKey,
      name: 'Team Key',
    })

    renderSettings()

    await waitFor(() =>
      expect(vi.mocked(api.settings.get)).toHaveBeenCalled(),
    )

    const createBtn = screen.getByRole('button', { name: /create api key/i })
    await user.click(createBtn)

    const dialog = screen.getByRole('dialog')
    await user.type(within(dialog).getByLabelText(/key name/i), 'Team Key')
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }))

    await waitFor(() =>
      expect(screen.getByText(/api key created/i)).toBeInTheDocument(),
    )

    const createdDialog = screen.getByRole('dialog')
    const inputs = within(createdDialog).getAllByRole('textbox')
    const snippetInput = inputs.find(
      (el) =>
        (el as HTMLInputElement).value.includes('personal-kb') &&
        (el as HTMLInputElement).value.includes('PERSONAL_KB_API_KEY'),
    )
    expect(snippetInput).toBeDefined()
    const snippetValue = (snippetInput as HTMLInputElement).value
    expect(snippetValue).toContain('"KB_INSTANCE_ROLE": "team"')
  })
})

describe('buildMcpSnippet — pure unit', () => {
  const { buildMcpSnippet, DEFAULT_MCP_CLIENT_FROM } = __testing

  it('uses uvx --from with the canonical entry point and git origin spec', () => {
    const snippet = JSON.parse(buildMcpSnippet('sk-123', null)) as {
      mcpServers: {
        'personal-kb': {
          command: string
          args: string[]
          env: Record<string, string>
        }
      }
    }
    const block = snippet.mcpServers['personal-kb']
    expect(block.command).toBe('uvx')
    expect(block.args).toEqual(['--from', DEFAULT_MCP_CLIENT_FROM, 'personal-kb'])
    expect(DEFAULT_MCP_CLIENT_FROM).toBe(
      'personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp',
    )
  })

  it('uses a server-provided install spec when given', () => {
    const spec = 'personal-kb @ git+https://git-host.example.com/x/personal_kb@v1'
    const snippet = JSON.parse(buildMcpSnippet('sk-1', null, spec)) as {
      mcpServers: { 'personal-kb': { args: string[] } }
    }
    expect(snippet.mcpServers['personal-kb'].args).toEqual([
      '--from',
      spec,
      'personal-kb',
    ])
  })

  it('team=null → env has URL + API key, no KB_INSTANCE_ROLE, no MCP_TOOL_PREFIX', () => {
    const snippet = JSON.parse(buildMcpSnippet('sk-abc', null)) as {
      mcpServers: { 'personal-kb': { env: Record<string, string> } }
    }
    const env = snippet.mcpServers['personal-kb'].env
    expect(env.PERSONAL_KB_API_KEY).toBe('sk-abc')
    expect(env.PERSONAL_KB_URL).toBeDefined()
    expect(env.KB_INSTANCE_ROLE).toBeUndefined()
    // Dead env (read by no consumer) — must NOT be emitted.
    expect(env.MCP_TOOL_PREFIX).toBeUndefined()
  })

  it('team="" (blank) → KB_INSTANCE_ROLE is omitted (default kb_* tools)', () => {
    const snippet = JSON.parse(buildMcpSnippet('sk-xyz', '')) as {
      mcpServers: { 'personal-kb': { env: Record<string, string> } }
    }
    const env = snippet.mcpServers['personal-kb'].env
    expect(env.KB_INSTANCE_ROLE).toBeUndefined()
  })

  it('team set → KB_INSTANCE_ROLE=team (team_kb_* tools)', () => {
    const snippet = JSON.parse(buildMcpSnippet('sk-def', 'Platform')) as {
      mcpServers: { 'personal-kb': { env: Record<string, string> } }
    }
    const env = snippet.mcpServers['personal-kb'].env
    expect(env.KB_INSTANCE_ROLE).toBe('team')
  })

  it('team whitespace-only → KB_INSTANCE_ROLE is omitted', () => {
    const snippet = JSON.parse(buildMcpSnippet('sk-ws', '   ')) as {
      mcpServers: { 'personal-kb': { env: Record<string, string> } }
    }
    expect(snippet.mcpServers['personal-kb'].env.KB_INSTANCE_ROLE).toBeUndefined()
  })
})

describe('Settings page — Account card (change password)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.apiKeys.list).mockResolvedValue({ keys: [] })
    vi.mocked(api.settings.get).mockResolvedValue({ team: null })
    mockAdminUser()
  })

  it('mismatched confirm makes no network call and shows inline error', async () => {
    const user = userEvent.setup()

    renderSettings()

    // Wait for mount
    await waitFor(() =>
      expect(vi.mocked(api.apiKeys.list)).toHaveBeenCalled(),
    )

    // Fill in password fields with mismatched values
    await user.type(screen.getByLabelText(/current password/i), 'oldpassword')
    await user.type(screen.getByLabelText(/^new password/i), 'newpassword1')
    await user.type(screen.getByLabelText(/confirm new password/i), 'newpassword2')

    // Submit
    await user.click(screen.getByRole('button', { name: /update password/i }))

    // Inline error appears
    await waitFor(() =>
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument(),
    )

    // No network request was made
    expect(vi.mocked(api.auth.changePassword)).not.toHaveBeenCalled()
  })
})

describe('Settings page — Team card', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.apiKeys.list).mockResolvedValue({ keys: [] })
    vi.mocked(api.settings.get).mockResolvedValue({ team: 'Test Team' })
  })

  it('non-admin: TextField is disabled with admins-only helper text and no Save button', async () => {
    mockNonAdminUser()
    renderSettings()

    await waitFor(() =>
      expect(vi.mocked(api.settings.get)).toHaveBeenCalled(),
    )

    const teamField = screen.getByLabelText(/team name/i)
    expect(teamField).toBeDisabled()
    expect(
      screen.getByText(/only admins can change the team name/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: /^save$/i }),
    ).not.toBeInTheDocument()
  })

  it('admin: TextField is editable and Save button is rendered', async () => {
    mockAdminUser()
    renderSettings()

    await waitFor(() =>
      expect(vi.mocked(api.settings.get)).toHaveBeenCalled(),
    )

    const teamField = screen.getByLabelText(/team name/i)
    expect(teamField).not.toBeDisabled()
    expect(screen.getByRole('button', { name: /^save$/i })).toBeInTheDocument()
  })
})
