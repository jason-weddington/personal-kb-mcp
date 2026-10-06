import { useState, useEffect, useCallback } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import CircularProgress from '@mui/material/CircularProgress'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import DeleteIcon from '@mui/icons-material/Delete'
import { useAuth } from '../contexts/AuthContext'
import { api, ApiError } from '../api'
import type { ApiKeyInfo, ApiKeyCreated } from '../types'

// ─── Account Card ────────────────────────────────────────────────────────────

function AccountCard() {
  const { user } = useAuth()
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [mismatch, setMismatch] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const disabled = !current || !next || !confirm || submitting

  const handleSubmit = useCallback(async () => {
    if (next !== confirm) {
      setMismatch(true)
      return
    }
    setMismatch(false)
    setError(null)
    setSuccess(false)
    setSubmitting(true)
    try {
      await api.auth.changePassword(current, next)
      setSuccess(true)
      setCurrent('')
      setNext('')
      setConfirm('')
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail)
      } else {
        setError('Unexpected error')
      }
    } finally {
      setSubmitting(false)
    }
  }, [current, next, confirm])

  return (
    <Card variant="outlined" sx={{ mb: 3 }}>
      <CardContent>
        <Typography
          variant="overline"
          display="block"
          gutterBottom
          color="text.secondary"
        >
          Account
        </Typography>

        <Typography variant="body2" sx={{ mb: 2 }}>
          Signed in as <strong>{user?.email}</strong>
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Password updated successfully.
          </Alert>
        )}

        <Box
          component="form"
          onSubmit={(e) => {
            e.preventDefault()
            void handleSubmit()
          }}
          sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 400 }}
        >
          <TextField
            label="Current password"
            type="password"
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
            size="small"
            fullWidth
          />
          <TextField
            label="New password"
            type="password"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            size="small"
            fullWidth
          />
          <TextField
            label="Confirm new password"
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            error={mismatch}
            helperText={mismatch ? 'Passwords do not match' : undefined}
            size="small"
            fullWidth
          />
          <Button
            type="submit"
            variant="contained"
            disabled={disabled}
            sx={{ alignSelf: 'flex-start' }}
          >
            Update password
          </Button>
        </Box>
      </CardContent>
    </Card>
  )
}

// ─── API Access Card ──────────────────────────────────────────────────────────

/**
 * `uvx --from` spec for the thin MCP client.
 *
 * Matches `scripts/provision.sh` KB_CLIENT_FROM for hosted mode: the package
 * is pulled from the home-lab git origin (the same repo all three Pis fetch).
 * Personal-kb is NOT a published PyPI package, so the `--from` indirection is
 * required for `uvx` to resolve the entry point.
 *
 * The HTTP thin-client (HttpBackend) never opens a database connection in
 * remote mode, so no extras are needed — base deps already cover
 * fastmcp + httpx.
 */
const DEFAULT_MCP_CLIENT_FROM =
  'personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp'

/**
 * Build the MCP-config JSON shown after minting an API key.
 *
 * The snippet pulls the thin client from the home-lab git origin via
 * `uvx --from` and points it at this instance's `/api` over HTTP.
 *
 * Tool-prefix mapping (see personal_kb server.py::_get_tool_prefix):
 *   - team set/non-blank  →  KB_INSTANCE_ROLE='team'    →  team_kb_* tools
 *   - team null/blank     →  KB_INSTANCE_ROLE omitted   →  kb_* tools (default)
 *
 * The `personal` role is reserved for the local single-user daemon and is not
 * emitted from the Settings UI (every hosted instance is either default-kb_*
 * or team-team_kb_*).
 */
function buildMcpSnippet(
  apiKey: string,
  team: string | null,
  installSpec: string = DEFAULT_MCP_CLIENT_FROM,
): string {
  const env: Record<string, string> = {
    PERSONAL_KB_URL: window.location.origin,
    PERSONAL_KB_API_KEY: apiKey,
  }
  if (team && team.trim() !== '') {
    env.KB_INSTANCE_ROLE = 'team'
  }
  return JSON.stringify(
    {
      mcpServers: {
        'personal-kb': {
          command: 'uvx',
          args: ['--from', installSpec, 'personal-kb'],
          env,
        },
      },
    },
    null,
    2,
  )
}

// Exposed for unit tests.
export const __testing = { buildMcpSnippet, DEFAULT_MCP_CLIENT_FROM }

function ApiAccessCard() {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  // Increment to trigger a data reload
  const [reloadTick, setReloadTick] = useState(0)

  // The instance's `team` setting drives the MCP snippet's KB_INSTANCE_ROLE.
  // Fetched once at mount; the snippet is only rendered after a key is
  // minted, so a brief loading window is invisible to the user.
  const [team, setTeam] = useState<string | null>(null)
  const [installSpec, setInstallSpec] = useState(DEFAULT_MCP_CLIENT_FROM)

  // Create-key dialog
  const [createOpen, setCreateOpen] = useState(false)
  const [createName, setCreateName] = useState('')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)

  // Created-key dialog
  const [createdKey, setCreatedKey] = useState<ApiKeyCreated | null>(null)

  // Delete-key dialog
  const [deleteTarget, setDeleteTarget] = useState<ApiKeyInfo | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  // Load keys — inline async IIFE avoids react-hooks/set-state-in-effect
  useEffect(() => {
    void (async () => {
      try {
        const res = await api.apiKeys.list()
        setKeys(res.keys)
        setLoadError(null)
      } catch (e) {
        if (e instanceof ApiError) {
          setLoadError(e.detail)
        } else {
          setLoadError('Failed to load API keys')
        }
      }
    })()
  }, [reloadTick])

  // Load the instance's team setting once so the MCP snippet can derive
  // KB_INSTANCE_ROLE. A fetch failure is non-fatal — we silently fall back
  // to the default (kb_* tools), and any deeper backend issue surfaces via
  // the other cards.
  useEffect(() => {
    void (async () => {
      try {
        const s = await api.settings.get()
        setTeam(s.team ?? null)
        if (s.clientInstallSpec) setInstallSpec(s.clientInstallSpec)
      } catch {
        setTeam(null)
      }
    })()
  }, [])

  const handleCreate = useCallback(async () => {
    setCreating(true)
    setCreateError(null)
    try {
      const result = await api.apiKeys.create(createName)
      setCreatedKey(result)
      setCreateOpen(false)
      setCreateName('')
      setReloadTick((t) => t + 1)
    } catch (e) {
      if (e instanceof ApiError) {
        setCreateError(e.detail)
      } else {
        setCreateError('Failed to create API key')
      }
    } finally {
      setCreating(false)
    }
  }, [createName])

  const handleDelete = useCallback(async () => {
    if (!deleteTarget) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await api.apiKeys.revoke(deleteTarget.id)
      setKeys((prev) => prev.filter((k) => k.id !== deleteTarget.id))
      setDeleteTarget(null)
    } catch (e) {
      if (e instanceof ApiError) {
        setDeleteError(e.detail)
      } else {
        setDeleteError('Failed to delete API key')
      }
    } finally {
      setDeleting(false)
    }
  }, [deleteTarget])

  const mcpSnippet = createdKey ? buildMcpSnippet(createdKey.apiKey, team, installSpec) : ''

  return (
    <Card variant="outlined" sx={{ mb: 3 }}>
      <CardContent>
        <Typography
          variant="overline"
          display="block"
          gutterBottom
          color="text.secondary"
        >
          API Access
        </Typography>

        {loadError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {loadError}
          </Alert>
        )}

        <Table size="small" sx={{ mb: 2 }}>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Prefix</TableCell>
              <TableCell>Created</TableCell>
              <TableCell />
            </TableRow>
          </TableHead>
          <TableBody>
            {keys.map((k) => (
              <TableRow key={k.id}>
                <TableCell>{k.name || 'Untitled'}</TableCell>
                <TableCell>
                  <code>{k.hashPrefix}</code>
                </TableCell>
                <TableCell>
                  {new Date(k.createdAt).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <IconButton
                    size="small"
                    aria-label="delete api key"
                    onClick={() => {
                      setDeleteTarget(k)
                      setDeleteError(null)
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            {keys.length === 0 && (
              <TableRow>
                <TableCell colSpan={4} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No API keys yet.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>

        <Button
          variant="outlined"
          onClick={() => {
            setCreateOpen(true)
            setCreateName('')
            setCreateError(null)
          }}
        >
          Create API Key
        </Button>

        {/* Create-key dialog */}
        <Dialog
          open={createOpen}
          onClose={() => setCreateOpen(false)}
          fullWidth
          maxWidth="xs"
        >
          <DialogTitle>Create API Key</DialogTitle>
          <DialogContent>
            {createError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {createError}
              </Alert>
            )}
            <TextField
              label="Key name (optional)"
              value={createName}
              onChange={(e) => setCreateName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handleCreate()
              }}
              fullWidth
              autoFocus
              sx={{ mt: 1 }}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button
              onClick={() => void handleCreate()}
              variant="contained"
              disabled={creating}
            >
              Create
            </Button>
          </DialogActions>
        </Dialog>

        {/* Created-key dialog */}
        <Dialog
          open={Boolean(createdKey)}
          onClose={() => setCreatedKey(null)}
          fullWidth
          maxWidth="sm"
        >
          <DialogTitle>API Key Created</DialogTitle>
          <DialogContent>
            <Alert severity="warning" sx={{ mb: 2 }}>
              This key will not be shown again. Copy it now.
            </Alert>

            <Typography variant="subtitle2" gutterBottom>
              API Key
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 3 }}>
              <TextField
                value={createdKey?.apiKey ?? ''}
                inputProps={{ readOnly: true, style: { fontFamily: 'monospace' } }}
                fullWidth
                size="small"
                multiline
              />
              <IconButton
                size="small"
                aria-label="copy api key"
                onClick={() => {
                  void navigator.clipboard.writeText(createdKey?.apiKey ?? '')
                }}
              >
                <ContentCopyIcon fontSize="small" />
              </IconButton>
            </Box>

            <Typography variant="subtitle2" gutterBottom>
              MCP Config Snippet
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 1 }}>
              <TextField
                value={mcpSnippet}
                inputProps={{ readOnly: true, style: { fontFamily: 'monospace' } }}
                fullWidth
                size="small"
                multiline
                rows={12}
              />
              <IconButton
                size="small"
                aria-label="copy mcp snippet"
                onClick={() => {
                  void navigator.clipboard.writeText(mcpSnippet)
                }}
              >
                <ContentCopyIcon fontSize="small" />
              </IconButton>
            </Box>
            <Typography variant="caption" color="text.secondary">
              Paste this into <code>~/.claude.json</code> under{' '}
              <code>mcpServers</code>. The thin client is pulled via{' '}
              <code>uvx --from</code> using the install spec configured on the
              server.{installSpec.includes('git+ssh://') &&
                ' SSH access to the git host is required.'}
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreatedKey(null)}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Delete confirm dialog */}
        <Dialog
          open={Boolean(deleteTarget)}
          onClose={() => setDeleteTarget(null)}
          maxWidth="xs"
          fullWidth
        >
          <DialogTitle>Delete API Key?</DialogTitle>
          <DialogContent>
            {deleteError && (
              <Alert severity="error" sx={{ mb: 1 }}>
                {deleteError}
              </Alert>
            )}
            <Typography>
              Delete key &ldquo;{deleteTarget?.name || 'Untitled'}&rdquo;? This
              cannot be undone.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteTarget(null)}>Cancel</Button>
            <Button
              color="error"
              variant="contained"
              onClick={() => void handleDelete()}
              disabled={deleting}
            >
              Delete
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  )
}

// ─── Team Card ────────────────────────────────────────────────────────────────

function TeamCard() {
  const { user } = useAuth()
  const isAdmin = user?.isAdmin === true

  const [team, setTeam] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    void (async () => {
      try {
        const s = await api.settings.get()
        setTeam(s.team ?? '')
      } catch (e) {
        if (e instanceof ApiError) {
          setError(e.detail)
        } else {
          setError('Failed to load settings')
        }
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  const handleSave = useCallback(async () => {
    setError(null)
    setSuccess(false)
    setSaving(true)
    try {
      const s = await api.settings.put(team)
      setTeam(s.team ?? '')
      setSuccess(true)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail)
      } else {
        setError('Failed to save settings')
      }
    } finally {
      setSaving(false)
    }
  }, [team])

  return (
    <Card variant="outlined" sx={{ mb: 3 }}>
      <CardContent>
        <Typography
          variant="overline"
          display="block"
          gutterBottom
          color="text.secondary"
        >
          Team
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Team name saved.
          </Alert>
        )}

        {loading ? (
          <CircularProgress size={24} />
        ) : (
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', maxWidth: 400 }}>
            <TextField
              label="Team name"
              value={team}
              onChange={(e) => setTeam(e.target.value)}
              disabled={!isAdmin}
              size="small"
              fullWidth
              // The UI hiding (no Save button for non-admins) is convenience only —
              // the server enforces admin via require_admin (403 'Admin only').
              helperText={
                !isAdmin
                  ? 'Only admins can change the team name'
                  : undefined
              }
            />
            {isAdmin && (
              <Button
                variant="contained"
                onClick={() => void handleSave()}
                disabled={saving}
                sx={{ mt: 0.5 }}
              >
                Save
              </Button>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

// ─── Settings Page ────────────────────────────────────────────────────────────

export function Settings() {
  const { authMode } = useAuth()
  // In no-auth (local) mode there is no session, no API keys, and no team —
  // AccountCard, ApiAccessCard (and its MCP snippet) and TeamCard are all
  // hosted-only surfaces, so render them only in jwt mode.
  const isJwt = authMode === 'jwt'
  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3 }}>
        Settings
      </Typography>
      {isJwt && <AccountCard />}
      {isJwt && <ApiAccessCard />}
      {isJwt && <TeamCard />}
    </Box>
  )
}
