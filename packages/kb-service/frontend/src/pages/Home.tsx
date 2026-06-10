import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'

export function Home() {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Welcome to Personal KB
      </Typography>
      <Typography variant="body1" color="text.secondary">
        The knowledge explorer and workbench will arrive in a later phase.
      </Typography>
    </Box>
  )
}
