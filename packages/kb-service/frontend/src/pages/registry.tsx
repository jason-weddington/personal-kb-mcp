import type { ReactElement } from 'react'
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer'
import SettingsIcon from '@mui/icons-material/Settings'
import SearchIcon from '@mui/icons-material/Search'
import BubbleChartIcon from '@mui/icons-material/BubbleChart'
import ChatIcon from '@mui/icons-material/Chat'
import { Home } from './Home'
import { Settings } from './Settings'
import { Search } from './Search'
import { Graph } from './Graph'
import { Chat } from './Chat'

export interface AppPage {
  path: string
  label: string
  icon: ReactElement
  element: ReactElement
  /**
   * Marks the page as hosted-only (jwt auth required). In no-auth (local)
   * mode such pages are hidden from the sidebar AND their route redirects to
   * "/". Currently used for /chat, whose history endpoints hit the auth DB
   * (get_db) — unreachable when KB_AUTH_MODE=none. Chat persistence in local
   * mode is explicitly out-of-scope for v1, so hiding is the consistent fix.
   */
  requiresAuth?: boolean
}

export const appPages: AppPage[] = [
  {
    path: '/',
    label: 'Home',
    icon: <QuestionAnswerIcon />,
    element: <Home />,
  },
  {
    path: '/search',
    label: 'Search',
    icon: <SearchIcon />,
    element: <Search />,
  },
  {
    path: '/chat',
    label: 'Chat',
    icon: <ChatIcon />,
    element: <Chat />,
    requiresAuth: true,
  },
  {
    path: '/graph',
    label: 'Graph',
    icon: <BubbleChartIcon />,
    element: <Graph />,
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    element: <Settings />,
  },
]
