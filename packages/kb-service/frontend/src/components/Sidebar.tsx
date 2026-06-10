import { useNavigate, useLocation } from 'react-router-dom'
import Box from '@mui/material/Box'
import Drawer from '@mui/material/Drawer'
import Divider from '@mui/material/Divider'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import Typography from '@mui/material/Typography'
import MailIcon from '@mui/icons-material/Mail'
import PeopleIcon from '@mui/icons-material/People'
import { useAuth } from '../contexts/AuthContext'
import { appPages } from '../pages/registry'

const DRAWER_WIDTH = 240

interface SidebarProps {
  open: boolean
  isMobile?: boolean
  onClose?: () => void
}

export default function Sidebar({ open, isMobile = false, onClose }: SidebarProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const { user } = useAuth()

  const isSelected = (path: string) =>
    path === '/' ? location.pathname === '/' : location.pathname.startsWith(path)

  const handleNavClick = (path: string) => {
    void navigate(path)
    if (isMobile) onClose?.()
  }

  const adminItems = [
    { path: '/admin/invites', label: 'Invites', icon: <MailIcon /> },
    { path: '/admin/users', label: 'Users', icon: <PeopleIcon /> },
  ]

  return (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      anchor="left"
      open={open}
      onClose={onClose}
      sx={{
        width: isMobile ? 0 : open ? DRAWER_WIDTH : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          top: '64px',
          height: 'calc(100% - 64px)',
        },
      }}
    >
      <Box sx={{ overflow: 'auto' }}>
        <List dense>
          {appPages.map((page) => (
            <ListItem key={page.path} disablePadding>
              <ListItemButton
                selected={isSelected(page.path)}
                onClick={() => handleNavClick(page.path)}
              >
                <ListItemIcon>{page.icon}</ListItemIcon>
                <ListItemText primary={page.label} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        {user?.isAdmin && (
          <>
            <Divider />
            <Typography
              variant="overline"
              sx={{ px: 2, pt: 1, display: 'block', color: 'text.secondary' }}
            >
              Admin
            </Typography>
            <List dense>
              {adminItems.map((item) => (
                <ListItem key={item.path} disablePadding>
                  <ListItemButton
                    selected={isSelected(item.path)}
                    onClick={() => handleNavClick(item.path)}
                  >
                    <ListItemIcon>{item.icon}</ListItemIcon>
                    <ListItemText primary={item.label} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </>
        )}
      </Box>
    </Drawer>
  )
}
