import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useState } from 'react'
import { Check, ChevronRight, FileImage, Gauge, LoaderCircle, MemoryStick, RotateCcw, ShieldCheck, Sparkles, UploadCloud, X } from 'lucide-react'
import { errorMessage, getModels, runPrediction, type ModelInfo, type PredictionResponse } from '../api'
import { ErrorBanner, PageHeading, formatMb, formatMs, formatPercent, modelLabel } from '../components/Ui'

const MAX_FILE_SIZE = 10 * 1024 * 1024

export function HomePage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [policy, setPolicy] = useState('learned')
  const [latencyBudget, setLatencyBudget] = useState(50)
  const [memoryBudget, setMemoryBudget] = useState(512)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    let active = true
    getModels().then((response) => active && setModels(response.models)).catch(() => undefined)
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview('')
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const names = useMemo(() => Object.fromEntries(models.map((item) => [item.name, item.display_name])), [models])

  function acceptFile(next: File | undefined) {
    setError('')
    setResult(null)
    if (!next) return
    if (!next.type.startsWith('image/')) {
      setError('Choose an image file such as JPEG, PNG, or WebP.')
      return
    }
    if (next.size > MAX_FILE_SIZE) {
      setError('The image is larger than 10 MiB. Choose a smaller file.')
      return
    }
    setFile(next)
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!file) {
      setError('Choose an image before running inference.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    const data = new FormData()
    data.append('file', file)
    data.append('policy', policy)
    data.append('latency_budget_ms', String(latencyBudget))
    data.append('memory_budget_mb', String(memoryBudget))
    try {
      setResult(await runPrediction(data))
    } catch (nextError) {
      setError(errorMessage(nextError))
    } finally {
      setLoading(false)
    }
  }

  const selectedModel = result ? modelLabel(result.selected_model, names) : ''

  return (
    <>
      <PageHeading eyebrow="Interactive experiment" title="Route one image, inspect every decision." description="Upload an image and test how the adaptive router responds to explicit latency and memory constraints. Measurements below come directly from this inference run." />
      <form onSubmit={submit} className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(340px,.65fr)]">
        <section className="panel overflow-hidden">
          <div className="section-heading"><div><p className="section-kicker">01 · Input</p><h2>Image specimen</h2></div><FileImage size={20} /></div>
          <div className="p-5 sm:p-6">
            {!file ? (
              <label
                className={`group grid min-h-[370px] cursor-pointer place-items-center rounded-2xl border border-dashed p-8 text-center transition focus-within:ring-2 focus-within:ring-accent/30 ${dragging ? 'border-accent bg-emerald-50/70' : 'border-stone-300 bg-stone-50 hover:border-accent/60 hover:bg-emerald-50/30'}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]) }}
              >
                <input className="sr-only" type="file" accept="image/*" aria-label="Upload image" onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFile(event.target.files?.[0])} />
                <span>
                  <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-ink text-white shadow-lg shadow-stone-900/10 transition group-hover:-translate-y-0.5"><UploadCloud size={24} /></span>
                  <span className="mt-5 block font-display text-xl font-semibold text-ink">Drop an image here</span>
                  <span className="mt-2 block text-sm text-stone-500">or browse your device · JPEG, PNG, WebP · up to 10 MiB</span>
                </span>
              </label>
            ) : (
              <div className="relative min-h-[370px] overflow-hidden rounded-2xl bg-stone-100">
                <img src={preview} alt="Selected upload preview" className="absolute inset-0 h-full w-full object-contain" />
                <div className="absolute inset-x-3 bottom-3 flex items-center gap-3 rounded-xl border border-white/40 bg-ink/85 p-3 text-white backdrop-blur-md">
                  <FileImage className="shrink-0 text-emerald-300" size={20} />
                  <div className="min-w-0 flex-1"><p className="truncate text-sm font-semibold">{file.name}</p><p className="text-[11px] text-stone-300">{(file.size / 1024).toFixed(1)} KB</p></div>
                  <button type="button" onClick={() => { setFile(null); setResult(null) }} className="rounded-lg p-2 hover:bg-white/10" aria-label="Remove image"><X size={17} /></button>
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="panel flex flex-col">
          <div className="section-heading"><div><p className="section-kicker">02 · Constraints</p><h2>Routing policy</h2></div><Gauge size={20} /></div>
          <div className="flex flex-1 flex-col p-5 sm:p-6">
            <label className="field-label" htmlFor="policy">Routing policy</label>
            <select id="policy" value={policy} onChange={(event) => setPolicy(event.target.value)} className="field-control">
              <option value="learned">Learned router</option>
              <option value="rule">Rule-based router</option>
              {models.filter((item) => item.available).map((item) => <option key={item.name} value={item.name}>Static · {item.display_name}</option>)}
            </select>
            <p className="field-hint">Adaptive policies include analysis and routing overhead.</p>

            <div className="mt-7 flex items-center justify-between"><label className="field-label mb-0" htmlFor="latency">Latency budget</label><output className="budget-value">{latencyBudget} ms</output></div>
            <input id="latency" aria-label="Latency budget" type="range" min="5" max="500" step="5" value={latencyBudget} onChange={(event) => setLatencyBudget(Number(event.target.value))} className="range-control" />
            <div className="flex justify-between text-[10px] font-medium text-stone-400"><span>5 ms</span><span>500 ms</span></div>

            <div className="mt-7 flex items-center justify-between"><label className="field-label mb-0" htmlFor="memory">Memory budget</label><output className="budget-value">{memoryBudget} MB</output></div>
            <input id="memory" aria-label="Memory budget" type="range" min="64" max="4096" step="64" value={memoryBudget} onChange={(event) => setMemoryBudget(Number(event.target.value))} className="range-control" />
            <div className="flex justify-between text-[10px] font-medium text-stone-400"><span>64 MB</span><span>4,096 MB</span></div>

            <div className="mt-7 rounded-xl border border-stone-200 bg-stone-50 p-4 text-xs leading-5 text-stone-600">
              <div className="flex items-center gap-2 font-semibold text-stone-800"><ShieldCheck size={15} className="text-accent" /> Constraint visibility</div>
              <p className="mt-1.5">If no model fits the budgets, the runtime reports the fallback and marks the constraint violation.</p>
            </div>

            <button type="submit" disabled={loading || !file} className="primary-button mt-auto pt-3">
              {loading ? <><LoaderCircle className="animate-spin" size={18} /> Running analysis…</> : <>Run inference <ChevronRight size={18} /></>}
            </button>
          </div>
        </section>
      </form>

      {error && <div className="mt-6"><ErrorBanner message={error} /></div>}

      {result && (
        <section className="mt-6 overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-panel" aria-live="polite">
          <div className="flex flex-col gap-3 border-b border-stone-200 bg-ink px-5 py-5 text-white sm:flex-row sm:items-center sm:justify-between sm:px-7">
            <div className="flex items-center gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl bg-emerald-400/15 text-emerald-300"><Sparkles size={18} /></span><div><p className="text-[10px] font-bold uppercase tracking-[0.18em] text-emerald-300">Inference complete</p><h2 className="font-display text-xl font-semibold">{result.prediction}</h2></div></div>
            <span className="font-mono text-[11px] text-stone-400">{result.request_id}</span>
          </div>
          <div className="grid md:grid-cols-[1fr_1.25fr]">
            <div className="border-b border-stone-200 p-6 md:border-b-0 md:border-r md:p-7">
              <p className="section-kicker">Prediction</p>
              <p className="mt-3 font-display text-3xl font-semibold tracking-tight text-ink">{result.prediction}</p>
              <p className="mt-1 text-sm text-stone-500">ImageNet class {result.class_index}</p>
              <div className="mt-6 grid grid-cols-2 gap-3">
                <ResultDatum label="Confidence" value={formatPercent(result.confidence, true)} />
                <ResultDatum label="Complexity" value={result.complexity.toFixed(3)} />
                <ResultDatum label="Total latency" value={formatMs(result.latency_ms)} />
                <ResultDatum label="Peak memory" value={formatMb(result.memory_mb)} />
              </div>
            </div>
            <div className="p-6 md:p-7">
              <div className="flex items-center justify-between gap-3"><div><p className="section-kicker">Routing decision</p><p className="mt-2 text-xl font-semibold text-ink">{selectedModel}</p></div><span className={`status-pill ${result.constraint_satisfied ? 'status-good' : 'status-warn'}`}>{result.constraint_satisfied ? <Check size={13} /> : <X size={13} />}{result.constraint_satisfied ? 'Within constraints' : 'Fallback used'}</span></div>
              <p className="mt-5 rounded-xl bg-stone-50 p-4 text-sm leading-6 text-stone-600">{result.reason}</p>
              <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-4">
                <Detail label="Inference" value={formatMs(result.inference_ms)} />
                <Detail label="Analyzer" value={formatMs(result.analyzer_ms)} />
                <Detail label="CPU" value={formatPercent(result.cpu_percent)} />
                <Detail label="Backend" value={result.backend} />
              </dl>
              <button type="button" onClick={() => { setResult(null); setFile(null) }} className="secondary-button mt-6"><RotateCcw size={15} /> New specimen</button>
            </div>
          </div>
        </section>
      )}
    </>
  )
}

function ResultDatum({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-stone-200 px-4 py-3"><dt className="text-[10px] font-bold uppercase tracking-[0.12em] text-stone-400">{label}</dt><dd className="mt-1 text-sm font-semibold text-stone-800">{value}</dd></div>
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div><dt className="text-xs text-stone-400">{label}</dt><dd className="mt-1 font-semibold text-stone-800">{value}</dd></div>
}
