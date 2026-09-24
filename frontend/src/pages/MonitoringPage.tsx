import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, Clock3, Cpu, Database, Radio, RefreshCw } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { errorMessage, getMetrics, type MetricsResponse } from '../api'
import { ErrorBanner, LoadingState, PageHeading, Stat, formatMb, formatMs, formatPercent, modelLabel } from '../components/Ui'

export function MonitoringPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      setMetrics(await getMetrics())
      setUpdatedAt(new Date())
      setError('')
    } catch (nextError) { setError(errorMessage(nextError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => void load(true), 4_000)
    return () => window.clearInterval(timer)
  }, [load])

  const chartData = useMemo(() => metrics?.recent.map((item, index) => ({
    sequence: index + 1,
    latency: typeof item.latency_ms === 'number' ? item.latency_ms : null,
    complexity: typeof item.complexity === 'number' ? item.complexity : null,
  })) ?? [], [metrics])

  return (
    <>
      <PageHeading eyebrow="Observed runtime" title="Live monitoring" description="A direct view of aggregated runtime metrics and recent requests. This page polls the backend every four seconds while it is open." actions={<div className="flex items-center gap-3"><span className="hidden text-xs text-stone-500 sm:block">{updatedAt ? `Updated ${updatedAt.toLocaleTimeString()}` : 'Not updated'}</span><button className="secondary-button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? 'animate-spin' : ''} /> Refresh</button></div>} />
      {error && <div className="mb-5"><ErrorBanner message={error} onRetry={() => void load()} /></div>}
      {loading && !metrics ? <LoadingState label="Reading runtime telemetry" /> : metrics && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
            <Stat label="Requests" value={metrics.requests.toLocaleString()} detail="Observed by this process" />
            <Stat label="Average latency" value={formatMs(metrics.average_latency_ms)} detail="End-to-end mean" />
            <Stat label="Current model" value={metrics.current_model ? modelLabel(metrics.current_model) : 'No selection'} detail="Latest routing choice" />
            <Stat label="Process CPU" value={formatPercent(metrics.cpu_percent)} detail="One core equals 100%" />
            <Stat label="Resident memory" value={formatMb(metrics.memory_mb)} detail="Current process RSS" />
          </div>

          {metrics.requests === 0 && metrics.recent.length === 0 ? (
            <div className="empty-state mt-6"><Radio size={28} /><h2>Waiting for inference traffic</h2><p>Submit an image in the inference lab. Runtime observations will appear here automatically.</p></div>
          ) : (
            <div className="mt-6 grid gap-6 xl:grid-cols-[1.5fr_1fr]">
              <section className="panel p-5 sm:p-6">
                <div className="mb-6 flex items-center justify-between"><div><p className="section-kicker">Recent series</p><h2 className="mt-1 text-lg font-semibold">Latency and complexity</h2></div><Activity className="text-accent" size={19} /></div>
                {chartData.length > 0 ? <div className="h-72 min-w-0" aria-label="Recent request chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}><CartesianGrid stroke="#e7e5e4" strokeDasharray="3 4" vertical={false} /><XAxis dataKey="sequence" tick={{ fill: '#78716c', fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis yAxisId="latency" tick={{ fill: '#78716c', fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis yAxisId="complexity" orientation="right" domain={[0, 1]} hide /><Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #e7e5e4', fontSize: 12 }} /><Legend wrapperStyle={{ fontSize: 11 }} /><Line yAxisId="latency" type="monotone" dataKey="latency" name="Latency (ms)" stroke="#167d69" strokeWidth={2.5} dot={false} connectNulls /><Line yAxisId="complexity" type="monotone" dataKey="complexity" name="Complexity" stroke="#9a6b2f" strokeWidth={2} dot={false} connectNulls /></LineChart></ResponsiveContainer></div> : <p className="py-24 text-center text-sm text-stone-500">Recent records do not include chartable values.</p>}
              </section>
              <section className="panel p-5 sm:p-6">
                <p className="section-kicker">Routing mix</p><h2 className="mt-1 text-lg font-semibold">Model selections</h2>
                <div className="mt-6 space-y-4">{Object.entries(metrics.model_counts).length === 0 ? <p className="py-16 text-center text-sm text-stone-500">No selection counts reported.</p> : Object.entries(metrics.model_counts).sort((a, b) => b[1] - a[1]).map(([name, count]) => { const ratio = metrics.requests > 0 ? (count / metrics.requests) * 100 : 0; return <div key={name}><div className="flex justify-between gap-3 text-xs"><span className="truncate font-medium text-stone-700">{modelLabel(name)}</span><span className="font-mono text-stone-500">{count}</span></div><div className="mt-2 h-2 overflow-hidden rounded-full bg-stone-100"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, ratio)}%` }} /></div></div> })}</div>
              </section>
            </div>
          )}

          {metrics.recent.length > 0 && <RecentTable metrics={metrics} />}
        </>
      )}
    </>
  )
}

function RecentTable({ metrics }: { metrics: MetricsResponse }) {
  return <section className="panel mt-6 overflow-hidden"><div className="section-heading"><div><p className="section-kicker">Request log</p><h2>Recent inference</h2></div><Clock3 size={18} /></div><div className="overflow-x-auto"><table className="data-table"><thead><tr><th>Request</th><th>Model</th><th>Latency</th><th>Complexity</th><th>Confidence</th><th>Constraint</th></tr></thead><tbody>{metrics.recent.map((item, index) => <tr key={item.request_id || index}><td className="font-mono text-xs">{item.request_id || `#${index + 1}`}</td><td>{modelLabel(String(item.selected_model || item.model || 'Unknown'))}</td><td>{formatMs(typeof item.latency_ms === 'number' ? item.latency_ms : null)}</td><td>{typeof item.complexity === 'number' ? item.complexity.toFixed(3) : '—'}</td><td>{typeof item.confidence === 'number' ? formatPercent(item.confidence, true) : '—'}</td><td>{item.constraint_satisfied == null ? '—' : item.constraint_satisfied ? 'Satisfied' : 'Violated'}</td></tr>)}</tbody></table></div></section>
}
