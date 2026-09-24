import type { ReactNode } from 'react'
import { AlertCircle, LoaderCircle, RotateCcw } from 'lucide-react'

export function PageHeading({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: ReactNode }) {
  return (
    <div className="mb-8 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
      <div className="max-w-3xl">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="mt-2 font-display text-3xl font-semibold tracking-[-0.035em] text-ink sm:text-4xl">{title}</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-stone-600">{description}</p>
      </div>
      {actions}
    </div>
  )
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
      <AlertCircle className="mt-0.5 shrink-0" size={17} />
      <div className="min-w-0 flex-1"><p className="font-semibold">Request could not be completed</p><p className="mt-0.5 text-rose-800">{message}</p></div>
      {onRetry && <button className="inline-flex items-center gap-1.5 font-semibold hover:text-rose-700" onClick={onRetry}><RotateCcw size={14} /> Retry</button>}
    </div>
  )
}

export function LoadingState({ label = 'Loading data' }: { label?: string }) {
  return <div className="grid min-h-56 place-items-center rounded-2xl border border-stone-200 bg-white"><div className="flex items-center gap-2 text-sm font-medium text-stone-500"><LoaderCircle className="animate-spin" size={18} />{label}</div></div>
}

export function Stat({ label, value, detail }: { label: string; value: ReactNode; detail?: string }) {
  return (
    <div className="rounded-2xl border border-stone-200/90 bg-white p-5 shadow-panel">
      <p className="text-xs font-semibold uppercase tracking-[0.13em] text-stone-500">{label}</p>
      <div className="mt-3 font-display text-2xl font-semibold tracking-tight text-ink">{value}</div>
      {detail && <p className="mt-1 text-xs text-stone-500">{detail}</p>}
    </div>
  )
}

export const formatMs = (value: number | null | undefined) => value == null ? 'Awaiting profile' : `${value.toFixed(1)} ms`
export const formatMb = (value: number | null | undefined) => value == null ? 'Awaiting profile' : `${value.toFixed(1)} MB`
export const formatPercent = (value: number | null | undefined, fraction = false) => value == null ? 'Awaiting profile' : `${(fraction ? value * 100 : value).toFixed(1)}%`
export const modelLabel = (name: string, displayNames?: Record<string, string>) => displayNames?.[name] || name.split('_').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ')
