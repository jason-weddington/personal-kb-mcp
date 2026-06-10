import { createTheme } from '@mui/material/styles'

const commonOptions = {
  typography: {
    fontFamily: 'Roboto, sans-serif',
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none' as const,
        },
      },
    },
  },
}

export const darkTheme = createTheme({
  ...commonOptions,
  palette: {
    mode: 'dark',
    background: {
      default: '#1a1a1a',
      paper: '#242424',
    },
    primary: {
      main: '#4a9eff',
    },
  },
})

export const lightTheme = createTheme({
  ...commonOptions,
  palette: {
    mode: 'light',
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    primary: {
      main: '#3d4f5f',
    },
  },
})
