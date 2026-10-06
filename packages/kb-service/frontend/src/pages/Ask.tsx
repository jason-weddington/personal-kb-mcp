import { Navigate } from 'react-router-dom'

/** /ask is now merged into the Home page — redirect deep links. */
export function Ask() {
  return <Navigate to="/" replace />
}
