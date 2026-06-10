import type { ReactElement } from 'react'
import HomeIcon from '@mui/icons-material/Home'
import SettingsIcon from '@mui/icons-material/Settings'
import { Home } from './Home'
import { Settings } from './Settings'

export interface AppPage {
  path: string
  label: string
  icon: ReactElement
  element: ReactElement
}

export const appPages: AppPage[] = [
  {
    path: '/',
    label: 'Home',
    icon: <HomeIcon />,
    element: <Home />,
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    element: <Settings />,
  },
]
