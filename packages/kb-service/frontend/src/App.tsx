import { Routes, Route, Navigate } from 'react-router-dom'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { ResetPassword } from './pages/ResetPassword'
import { AdminInvites } from './pages/AdminInvites'
import { AdminUsers } from './pages/AdminUsers'
import { Layout } from './components/Layout'
import { ProtectedRoute } from './components/ProtectedRoute'
import { AdminRoute } from './components/AdminRoute'
import { appPages } from './pages/registry'
import { EntryDetail } from './pages/EntryDetail'

export function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/reset-password" element={<ResetPassword />} />

      {/* Authenticated routes (home, settings, search, graph, ask, chat, …) */}
      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        {appPages.map((page) => (
          <Route key={page.path} path={page.path} element={page.element} />
        ))}
        {/* /ask is merged into Home — redirect deep links */}
        <Route path="/ask" element={<Navigate to="/" replace />} />
        {/* Entry detail — navigation-only, no nav link */}
        <Route path="/entries/:id" element={<EntryDetail />} />
      </Route>

      {/* Admin-only routes */}
      <Route
        element={
          <AdminRoute>
            <Layout />
          </AdminRoute>
        }
      >
        <Route path="/admin/invites" element={<AdminInvites />} />
        <Route path="/admin/users" element={<AdminUsers />} />
      </Route>
    </Routes>
  )
}
