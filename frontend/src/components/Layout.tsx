import { useEffect, useState } from 'react'
import {
  Activity,
  Boxes,
  FlaskConical,
  GitBranch,
  Menu,
  PanelLeftClose,
  ScanSearch,
  X,
} from 'lucide-react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { errorMessage, getHealth, type HealthResponse } from '../api'

const navigation = [
  { to: '/', label: 'Inference lab', icon: ScanSearch },
  { to: '/models', label: 'Model registry', icon: Boxes },
  { to: '/monitoring', label: 'Live monitoring', icon: Activity },
  { to: '/results', label: 'Research results', icon: FlaskConical },
  { to: '/architecture', label: 'Architecture', icon: GitBranch },
]

const pageNames: Record<string, string> = {
  '/': 'Inference lab',
  '/models': 'Model registry',
  '/monitoring': 'Live monitoring',
  '/results': 'Research results',
  '/architecture': 'System architecture',
}

export function Layout() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState('')
  const location = useLocation()

  useEffect(() => {
    let active = true
    const check = async () => {
      try {
        const next = await getHealth()
        if (active) {
          setHealth(next)
          setHealthError('')
        }
      } catch (error) {
        if (active) {
          setHealth(null)
          setHealthError(errorMessage(error))
        }
      }
    }
    void check()
    const timer = window.setInterval(check, 15_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => setMenuOpen(false), [location.pathname])

  return (
    <div className="min-h-screen bg-canvas text-ink lg:grid lg:grid-cols-[272px_minmax(0,1fr)]">
      {menuOpen && (
        <button
          className="fixed inset-0 z-30 bg-ink/45 lg:hidden"
          aria-label="Close navigation overlay"
          onClick={() => setMenuOpen(false)}
        />
      )}
      <aside className={`fixed inset-y-0 left-0 z-40 flex w-[272px] flex-col bg-ink text-stone-100 transition-transform duration-200 lg:sticky lg:top-0 lg:h-screen lg:translate-x-0 ${menuOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex h-24 items-center justify-between border-b border-white/10 px-7">
          <NavLink to="/" className="flex items-center gap-3" aria-label="AdaptiveServe home">
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-emerald-500/15 ring-1 ring-inset ring-emerald-300/20">
              <span className="logo-mark" aria-hidden="true"><i /><i /><i /></span>
            </span>
            <span>
              <span className="block text-[15px] font-semibold tracking-tight">AdaptiveServe</span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.22em] text-stone-400">Research runtime</span>
            </span>
          </NavLink>
          <button className="icon-button lg:hidden" onClick={() => setMenuOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <nav className="flex-1 space-y-1 px-4 py-7" aria-label="Primary navigation">
          <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">Workspace</p>
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${isActive ? 'bg-white/10 text-white' : 'text-stone-400 hover:bg-white/[0.06] hover:text-stone-100'}`}
            >
              {({ isActive }) => (
                <>
                  <Icon size={18} strokeWidth={isActive ? 2.2 : 1.7} className={isActive ? 'text-emerald-400' : 'text-stone-500 group-hover:text-stone-300'} />
                  <span>{label}</span>
                  {isActive && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-emerald-400" />}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="m-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4">
          <div className="flex items-center gap-2">
            <span className={`relative flex h-2 w-2 rounded-full ${health?.ready ? 'bg-emerald-400' : healthError ? 'bg-rose-400' : 'bg-amber-300'}`}>
              {health?.ready && <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400 opacity-50" />}
            </span>
            <span className="text-xs font-semibold text-stone-200">{health?.ready ? 'Runtime ready' : healthError ? 'Runtime offline' : 'Checking runtime'}</span>
          </div>
          <p className="mt-2 truncate text-[11px] text-stone-500" title={healthError || health?.detail || undefined}>
            {health ? `${health.backend} backend · ${health.status}` : healthError || 'Connecting to API…'}
          </p>
        </div>
      </aside>

      <div className="min-w-0">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-stone-200/80 bg-canvas/90 px-5 backdrop-blur-xl md:px-8 lg:px-10">
          <div className="flex items-center gap-3">
            <button className="rounded-lg p-2 text-stone-600 hover:bg-stone-200/70 lg:hidden" onClick={() => setMenuOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Adaptive inference</p>
              <p className="text-sm font-semibold text-stone-700">{pageNames[location.pathname] ?? 'Research console'}</p>
            </div>
          </div>
          <div className="hidden items-center gap-2 text-xs text-stone-500 sm:flex">
            <PanelLeftClose size={15} />
            <span>Batch one · 1,000 classes</span>
          </div>
        </header>
        <main className="mx-auto max-w-[1480px] px-5 py-8 md:px-8 lg:px-10 lg:py-10">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
