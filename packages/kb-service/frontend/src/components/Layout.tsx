import { useState } from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import AppBar from '@mui/material/AppBar'
import Box from '@mui/material/Box'
import CssBaseline from '@mui/material/CssBaseline'
import IconButton from '@mui/material/IconButton'
import Menu from '@mui/material/Menu'
import MenuItem from '@mui/material/MenuItem'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import MenuIcon from '@mui/icons-material/Menu'
import Brightness4Icon from '@mui/icons-material/Brightness4'
import Brightness7Icon from '@mui/icons-material/Brightness7'
import AccountCircleIcon from '@mui/icons-material/AccountCircle'
import useMediaQuery from '@mui/material/useMediaQuery'
import { useTheme } from '@mui/material/styles'
import { useAuth } from '../contexts/AuthContext'
import { useThemeMode } from '../contexts/ThemeContext'
import Sidebar from './Sidebar'

export function Layout() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const { mode, toggleTheme } = useThemeMode()
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)

  const handleUserMenuOpen = (e: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(e.currentTarget)
  }

  const handleUserMenuClose = () => {
    setAnchorEl(null)
  }

  const handleLogout = () => {
    handleUserMenuClose()
    logout()
    void navigate('/login')
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: (t) => t.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            sx={{ mr: 1 }}
            aria-label="toggle sidebar"
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Personal KB
          </Typography>
          <IconButton color="inherit" onClick={toggleTheme}>
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          <IconButton color="inherit" onClick={handleUserMenuOpen}>
            <AccountCircleIcon />
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleUserMenuClose}
          >
            <MenuItem disabled>
              <Typography variant="body2">{user?.email}</Typography>
            </MenuItem>
            <MenuItem onClick={handleLogout}>Logout</MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Sidebar
        open={sidebarOpen}
        isMobile={isMobile}
        onClose={() => setSidebarOpen(false)}
      />

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          mt: '64px',
          p: 3,
          minHeight: 'calc(100vh - 64px)',
          transition: 'margin-left 225ms cubic-bezier(0, 0, 0.2, 1)',
          overflowX: 'hidden',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  )
}
