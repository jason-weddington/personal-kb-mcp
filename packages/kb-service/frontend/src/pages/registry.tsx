import type { ReactElement } from 'react'
import HomeIcon from '@mui/icons-material/Home'
import { Home } from './Home'

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
]
