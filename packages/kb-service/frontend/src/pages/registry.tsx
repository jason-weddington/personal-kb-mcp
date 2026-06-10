import type { ReactElement } from 'react'
import HomeIcon from '@mui/icons-material/Home'
import SettingsIcon from '@mui/icons-material/Settings'
import SearchIcon from '@mui/icons-material/Search'
import BubbleChartIcon from '@mui/icons-material/BubbleChart'
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer'
import ChatIcon from '@mui/icons-material/Chat'
import { Home } from './Home'
import { Settings } from './Settings'
import { Search } from './Search'
import { Graph } from './Graph'
import { Ask } from './Ask'
import { Chat } from './Chat'

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
  {
    path: '/search',
    label: 'Search',
    icon: <SearchIcon />,
    element: <Search />,
  },
  {
    path: '/graph',
    label: 'Graph',
    icon: <BubbleChartIcon />,
    element: <Graph />,
  },
  {
    path: '/ask',
    label: 'Ask',
    icon: <QuestionAnswerIcon />,
    element: <Ask />,
  },
  {
    path: '/chat',
    label: 'Chat',
    icon: <ChatIcon />,
    element: <Chat />,
  },
]
