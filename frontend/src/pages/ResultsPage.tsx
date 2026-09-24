import { useCallback, useEffect, useState } from 'react'
import { BarChart3, FileQuestion, RefreshCw } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { errorMessage, getResults, type ResultsResponse } from '../api'
import { ErrorBanner, LoadingState, PageHeading, formatMb, formatMs, formatPercent, modelLabel } from '../components/Ui'

export function ResultsPage() {
  const [results, setResults] = useState<ResultsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const load = useCallback(async () => { setLoading(true); setError(''); try { setResults(await getResults()) } catch (nextError) { setError(errorMessage(nextError)) } finally { setLoading(false) } }, [])
  useEffect(() => { void load() }, [load])

  return (
    <>
      <PageHeading eyebrow="Held-out evaluation" title="Research results" description="Compare adaptive and static policies using benchmark artifacts produced by the experiment pipeline. Values and figures on this page are served from recorded runs." actions={<button className="secondary-button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? 'animate-spin' : ''} /> Refresh</button>} />
      {error && <ErrorBanner message={error} onRetry={() => void load()} />}
      {loading ? <LoadingState label="Loading benchmark artifacts" /> : !results?.available || results.summary.length === 0 ? (
        <div className="empty-state"><FileQuestion size={30} /><h2>No benchmark results yet</h2><p>Run the held-out experiment pipeline to publish summary rows and plots. This view does not substitute demonstration values.</p></div>
      ) : (
        <>
          <div className="grid gap-6 xl:grid-cols-2">
            <ResultChart title="Accuracy by policy" data={results.summary} dataKey="accuracy" formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} fraction />
            <ResultChart title="End-to-end latency" data={results.summary} dataKey="latency_ms" formatter={(value) => `${Number(value).toFixed(1)} ms`} />
          </div>
          <section className="panel mt-6 overflow-hidden"><div className="section-heading"><div><p className="section-kicker">Comparison matrix</p><h2>Policy measurements</h2></div><BarChart3 size={19} /></div><div className="overflow-x-auto"><table className="data-table"><thead><tr><th>Policy</th><th>Samples</th><th>Accuracy</th><th>Mean latency</th><th>P95 latency</th><th>Throughput</th><th>Memory</th><th>CPU</th><th>SLA violations</th></tr></thead><tbody>{results.summary.map((row) => <tr key={row.policy}><td className="font-semibold">{modelLabel(row.policy)}</td><td>{row.n.toLocaleString()}</td><td>{formatPercent(row.accuracy, true)}</td><td>{formatMs(row.latency_ms)}</td><td>{formatMs(row.p95_latency_ms)}</td><td>{row.throughput_rps.toFixed(2)} rps</td><td>{formatMb(row.memory_mb)}</td><td>{formatPercent(row.cpu_percent)}</td><td>{formatPercent(row.sla_violation_rate, true)}</td></tr>)}</tbody></table></div></section>
          {results.plots.length > 0 && <section className="mt-6"><div className="mb-4"><p className="section-kicker">Generated artifacts</p><h2 className="mt-1 text-lg font-semibold">Experiment figures</h2></div><div className="grid gap-6 xl:grid-cols-2">{results.plots.map((plot) => <figure key={`${plot.name}-${plot.url}`} className="panel overflow-hidden"><img src={plot.url} alt={plot.name} className="w-full bg-white object-contain" loading="lazy" /><figcaption className="border-t border-stone-200 px-5 py-3 text-sm font-medium text-stone-700">{plot.name}</figcaption></figure>)}</div></section>}
          {Object.keys(results.metadata).length > 0 && <details className="panel mt-6 p-5"><summary className="cursor-pointer text-sm font-semibold text-stone-700">Run metadata</summary><dl className="mt-4 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-3">{Object.entries(results.metadata).map(([key, value]) => <div key={key}><dt className="text-stone-400">{key.replace(/_/g, ' ')}</dt><dd className="mt-1 break-words font-medium text-stone-700">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd></div>)}</dl></details>}
        </>
      )}
    </>
  )
}

function ResultChart({ title, data, dataKey, formatter, fraction = false }: { title: string; data: ResultsResponse['summary']; dataKey: 'accuracy' | 'latency_ms'; formatter: (value: unknown) => string; fraction?: boolean }) {
  const chartData = data.map((row) => ({ ...row, policyLabel: modelLabel(row.policy), [dataKey]: fraction ? row[dataKey] * 100 : row[dataKey] }))
  return <section className="panel p-5 sm:p-6"><p className="section-kicker">Policy comparison</p><h2 className="mt-1 text-lg font-semibold">{title}</h2><div className="mt-6 h-72 min-w-0"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 4, right: 10, bottom: 18, left: -14 }}><CartesianGrid vertical={false} stroke="#e7e5e4" strokeDasharray="3 4" /><XAxis dataKey="policyLabel" tick={{ fill: '#78716c', fontSize: 10 }} axisLine={false} tickLine={false} interval={0} angle={-8} textAnchor="end" /><YAxis tick={{ fill: '#78716c', fontSize: 10 }} axisLine={false} tickLine={false} domain={fraction ? [0, 100] : undefined} /><Tooltip formatter={fraction ? (value) => `${Number(value).toFixed(1)}%` : formatter} contentStyle={{ borderRadius: 12, border: '1px solid #e7e5e4', fontSize: 12 }} /><Legend wrapperStyle={{ fontSize: 11 }} /><Bar dataKey={dataKey} name={fraction ? 'Accuracy (%)' : 'Latency (ms)'} fill="#167d69" radius={[5, 5, 0, 0]} maxBarSize={56} /></BarChart></ResponsiveContainer></div></section>
}
