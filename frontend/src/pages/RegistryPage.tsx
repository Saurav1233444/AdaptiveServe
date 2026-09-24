import { useCallback, useEffect, useState } from 'react'
import { Box, CheckCircle2, CircleDashed, Cpu, Database, RefreshCw, ServerOff } from 'lucide-react'
import { errorMessage, getModels, type ModelInfo } from '../api'
import { ErrorBanner, LoadingState, PageHeading, formatMb, formatMs, formatPercent } from '../components/Ui'

export function RegistryPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try { setModels((await getModels()).models) }
    catch (nextError) { setError(errorMessage(nextError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { void load() }, [load])

  return (
    <>
      <PageHeading eyebrow="Runtime inventory" title="Model registry" description="Availability, residency, and calibrated profiles for every inference candidate. Blank measurements remain blank until the profiling pipeline records them." actions={<button className="secondary-button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? 'animate-spin' : ''} /> Refresh</button>} />
      {error && <ErrorBanner message={error} onRetry={() => void load()} />}
      {loading ? <LoadingState label="Reading model registry" /> : models.length === 0 ? (
        <div className="empty-state"><ServerOff size={28} /><h2>No models registered</h2><p>Add model definitions to the registry, then refresh this view.</p></div>
      ) : (
        <div className="grid gap-5 xl:grid-cols-3">
          {models.map((model, index) => <ModelCard key={model.name} model={model} index={index} />)}
        </div>
      )}
    </>
  )
}

function ModelCard({ model, index }: { model: ModelInfo; index: number }) {
  const profiled = model.accuracy != null || model.latency_ms != null || model.memory_mb != null
  return (
    <article className="panel overflow-hidden">
      <div className="border-b border-stone-200 p-5">
        <div className="flex items-start justify-between gap-3">
          <span className="grid h-11 w-11 place-items-center rounded-xl bg-stone-100 text-stone-700"><Box size={20} /></span>
          <span className={`status-pill ${model.available ? 'status-good' : 'status-muted'}`}>{model.available ? <CheckCircle2 size={13} /> : <CircleDashed size={13} />}{model.available ? 'Available' : 'Unavailable'}</span>
        </div>
        <p className="mt-5 text-[10px] font-bold uppercase tracking-[0.16em] text-stone-400">Candidate {String(index + 1).padStart(2, '0')}</p>
        <h2 className="mt-1 font-display text-xl font-semibold text-ink">{model.display_name}</h2>
        <p className="mt-1 break-all font-mono text-[11px] text-stone-400">{model.name}</p>
      </div>
      <div className="grid grid-cols-3 divide-x divide-stone-200 border-b border-stone-200">
        <RegistryDatum label="Accuracy" value={formatPercent(model.accuracy, true)} />
        <RegistryDatum label="Latency" value={formatMs(model.latency_ms)} />
        <RegistryDatum label="Memory" value={formatMb(model.memory_mb)} />
      </div>
      <div className="space-y-3 p-5 text-xs">
        <div className="flex items-center justify-between"><span className="flex items-center gap-2 text-stone-500"><Cpu size={14} />Runtime state</span><span className="font-semibold text-stone-800">{model.loaded ? 'Loaded' : 'Not loaded'}</span></div>
        <div className="flex items-center justify-between"><span className="flex items-center gap-2 text-stone-500"><Database size={14} />Input type</span><span className="font-semibold text-stone-800">{model.input_type}</span></div>
        <div className="flex items-center justify-between"><span className="text-stone-500">Profile status</span><span className={`font-semibold ${profiled ? 'text-accent' : 'text-amber-700'}`}>{profiled ? 'Calibrated' : 'Awaiting profile'}</span></div>
      </div>
    </article>
  )
}

function RegistryDatum({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0 px-3 py-4 text-center"><p className="text-[9px] font-bold uppercase tracking-wider text-stone-400">{label}</p><p className={`mt-1 truncate text-xs font-semibold ${value === 'Awaiting profile' ? 'text-stone-400' : 'text-stone-800'}`} title={value}>{value}</p></div>
}
