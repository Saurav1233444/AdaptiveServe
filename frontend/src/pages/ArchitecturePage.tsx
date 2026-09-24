import { useState } from 'react'
import { ArrowDown, BarChart3, BrainCircuit, Boxes, Gauge, Image, Network, ServerCog } from 'lucide-react'
import { PageHeading } from '../components/Ui'

const stages = [
  { id: 'analyzer', label: 'Analyzer', caption: 'Image difficulty', icon: BrainCircuit, description: 'Extracts six image statistics and a learned embedding, then returns a normalized complexity score. Its measured analyzer time is included in adaptive-policy latency.', inputs: ['PIL image'], outputs: ['Complexity', 'Features', 'Analyzer time'] },
  { id: 'context', label: 'Context', caption: 'Resource envelope', icon: Gauge, description: 'Combines image complexity with the requested latency budget, current CPU availability, and memory budget to form the routing context.', inputs: ['Complexity', 'Latency budget', 'Memory budget'], outputs: ['Routing context'] },
  { id: 'router', label: 'Router', caption: 'Feasible selection', icon: Network, description: 'Selects the feasible model with the best estimated utility. Profile masks remove infeasible candidates, while any fallback constraint violation remains visible in the response.', inputs: ['Routing context', 'Model profiles'], outputs: ['Selected model', 'Reason', 'Constraint status'] },
  { id: 'registry', label: 'Registry', caption: 'Measured profiles', icon: Boxes, description: 'Supplies ordered model metadata and calibrated measurements. Accuracy, latency, and memory remain unset until real profiling observations are recorded.', inputs: ['Model configuration', 'Profile observations'], outputs: ['Availability', 'Runtime profile'] },
  { id: 'runtime', label: 'Runtime', caption: 'Native inference', icon: ServerCog, description: 'Applies each pretrained weight’s documented preprocessing and executes the full 1,000-class model through the selected backend.', inputs: ['Image', 'Selected model'], outputs: ['Class logits', 'Timing', 'Process resources'] },
  { id: 'metrics', label: 'Metrics', caption: 'Observed evidence', icon: BarChart3, description: 'Aggregates recent requests, model-selection counts, end-to-end latency, and process resource measurements for monitoring and held-out evaluation.', inputs: ['Prediction response'], outputs: ['Live metrics', 'Research artifacts'] },
]

export function ArchitecturePage() {
  const [selected, setSelected] = useState('router')
  const active = stages.find((stage) => stage.id === selected)!
  const ActiveIcon = active.icon
  return (
    <>
      <PageHeading eyebrow="Execution model" title="System architecture" description="Explore the path from an uploaded image to a measured prediction. Select any stage to inspect its contract and role in the decision." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(330px,.55fr)]">
        <section className="panel overflow-hidden">
          <div className="section-heading"><div><p className="section-kicker">Request path</p><h2>Adaptive serving flow</h2></div><Image size={19} /></div>
          <div className="p-5 sm:p-7">
            <div className="grid gap-3 md:grid-cols-3">
              {stages.slice(0, 3).map((stage, index) => <StageButton key={stage.id} stage={stage} selected={selected} onSelect={setSelected} index={index} />)}
            </div>
            <div className="flex justify-center py-3 text-stone-300"><ArrowDown size={20} /></div>
            <div className="grid gap-3 md:grid-cols-3">
              {stages.slice(3).map((stage, index) => <StageButton key={stage.id} stage={stage} selected={selected} onSelect={setSelected} index={index + 3} />)}
            </div>
            <div className="mt-6 rounded-xl border border-dashed border-stone-300 bg-stone-50 p-4 text-center text-xs leading-5 text-stone-500">Adaptive timing spans analyzer → router → native inference. Static policies execute the selected model directly.</div>
          </div>
        </section>

        <aside className="overflow-hidden rounded-2xl bg-ink text-white shadow-panel" aria-live="polite">
          <div className="border-b border-white/10 p-6"><span className="grid h-12 w-12 place-items-center rounded-2xl bg-emerald-400/15 text-emerald-300"><ActiveIcon size={23} /></span><p className="mt-6 text-[10px] font-bold uppercase tracking-[0.18em] text-emerald-300">Selected stage</p><h2 className="mt-1 font-display text-2xl font-semibold">{active.label}</h2><p className="mt-4 text-sm leading-6 text-stone-300">{active.description}</p></div>
          <div className="grid gap-6 p-6 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2"><Contract title="Consumes" values={active.inputs} /><Contract title="Produces" values={active.outputs} /></div>
        </aside>
      </div>
    </>
  )
}

function StageButton({ stage, selected, onSelect, index }: { stage: typeof stages[number]; selected: string; onSelect: (id: string) => void; index: number }) {
  const Icon = stage.icon
  const active = selected === stage.id
  return <button type="button" aria-label={`${stage.label} stage`} aria-pressed={active} onClick={() => onSelect(stage.id)} className={`group rounded-2xl border p-4 text-left transition focus:outline-none focus:ring-2 focus:ring-accent/35 ${active ? 'border-accent bg-emerald-50 shadow-sm' : 'border-stone-200 bg-white hover:border-stone-300 hover:bg-stone-50'}`}><div className="flex items-center justify-between"><span className={`grid h-9 w-9 place-items-center rounded-xl ${active ? 'bg-accent text-white' : 'bg-stone-100 text-stone-500 group-hover:text-stone-700'}`}><Icon size={18} /></span><span className="font-mono text-[10px] text-stone-400">0{index + 1}</span></div><p className="mt-4 text-sm font-semibold text-stone-800">{stage.label}</p><p className="mt-0.5 text-xs text-stone-500">{stage.caption}</p></button>
}

function Contract({ title, values }: { title: string; values: string[] }) {
  return <div><p className="text-[10px] font-bold uppercase tracking-[0.16em] text-stone-500">{title}</p><ul className="mt-3 space-y-2">{values.map((value) => <li key={value} className="flex items-center gap-2 text-xs text-stone-300"><span className="h-1 w-1 rounded-full bg-emerald-400" />{value}</li>)}</ul></div>
}
