import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'

const ArchitecturePage = lazy(() => import('./pages/ArchitecturePage').then((module) => ({ default: module.ArchitecturePage })))
const MonitoringPage = lazy(() => import('./pages/MonitoringPage').then((module) => ({ default: module.MonitoringPage })))
const RegistryPage = lazy(() => import('./pages/RegistryPage').then((module) => ({ default: module.RegistryPage })))
const ResultsPage = lazy(() => import('./pages/ResultsPage').then((module) => ({ default: module.ResultsPage })))

function RouteFallback() {
  return <div className="grid min-h-72 place-items-center text-sm font-medium text-stone-500">Opening workspace…</div>
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="models" element={<Suspense fallback={<RouteFallback />}><RegistryPage /></Suspense>} />
          <Route path="monitoring" element={<Suspense fallback={<RouteFallback />}><MonitoringPage /></Suspense>} />
          <Route path="results" element={<Suspense fallback={<RouteFallback />}><ResultsPage /></Suspense>} />
          <Route path="architecture" element={<Suspense fallback={<RouteFallback />}><ArchitecturePage /></Suspense>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
