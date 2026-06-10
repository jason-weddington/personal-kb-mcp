import { Routes, Route } from 'react-router-dom'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { Layout } from './components/Layout'
import { ProtectedRoute } from './components/ProtectedRoute'
import { appPages } from './pages/registry'

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
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
      </Route>
    </Routes>
  )
}
