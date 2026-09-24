import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const models = {
  models: [
    {
      name: 'mobilenet_v3_small',
      display_name: 'MobileNetV3 Small',
      accuracy: null,
      latency_ms: null,
      memory_mb: null,
      input_type: 'image',
      endpoint: '/workers/mobilenet',
      available: true,
      loaded: false,
    },
  ],
}

const prediction = {
  request_id: 'req-42',
  complexity: 0.63,
  selected_model: 'mobilenet_v3_small',
  reason: 'Selected within the active latency and memory constraints.',
  prediction: 'tabby cat',
  class_index: 281,
  confidence: 0.873,
  latency_ms: 19.4,
  inference_ms: 12.1,
  analyzer_ms: 4.2,
  cpu_percent: 33.1,
  memory_mb: 148.2,
  constraint_satisfied: true,
  backend: 'cpp',
}

function json(data: unknown, ok = true, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  }))
}

function baseFetch(input: RequestInfo | URL) {
  const url = String(input)
  if (url.endsWith('/api/health')) return json({ status: 'ok', backend: 'cpp', ready: true, detail: null })
  if (url.endsWith('/api/models')) return json(models)
  if (url.endsWith('/api/metrics')) return json({ requests: 0, average_latency_ms: null, current_model: null, cpu_percent: null, memory_mb: null, recent: [], model_counts: {} })
  if (url.endsWith('/api/results')) return json({ available: false, summary: [], plots: [], metadata: {} })
  return json({}, false, 404)
}

beforeEach(() => {
  window.history.pushState({}, '', '/')
  vi.stubGlobal('fetch', vi.fn(baseFetch))
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('AdaptiveServe console', () => {
  it('rejects images above the API 10 MiB limit before submitting', async () => {
    const user = userEvent.setup()
    render(<App />)
    const file = new File(['pixel'], 'oversized.png', { type: 'image/png' })
    Object.defineProperty(file, 'size', { value: 10 * 1024 * 1024 + 1 })
    await user.upload(screen.getByLabelText(/upload image/i), file)
    expect(await screen.findByRole('alert')).toHaveTextContent('10 MiB')
    expect(screen.queryByText('oversized.png')).not.toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/api/predict'))).toBe(false)
  })

  it('submits an uploaded image with routing settings and renders the genuine response', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch).mockImplementation((input, init) => {
      if (String(input).endsWith('/api/predict')) {
        expect(init?.method).toBe('POST')
        const body = init?.body as FormData
        expect(body.get('policy')).toBe('rule')
        expect(body.get('latency_budget_ms')).toBe('50')
        expect(body.get('memory_budget_mb')).toBe('512')
        return json(prediction)
      }
      return baseFetch(input)
    })

    render(<App />)
    const file = new File(['pixel'], 'sample.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText(/upload image/i), file)
    expect(screen.getByText('sample.png')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText(/routing policy/i), 'rule')
    await user.click(screen.getByRole('button', { name: /run inference/i }))

    expect(await screen.findByRole('heading', { name: 'tabby cat' })).toBeInTheDocument()
    expect(screen.getByText('MobileNetV3 Small')).toBeInTheDocument()
    expect(screen.getByText('87.3%')).toBeInTheDocument()
    expect(screen.getByText('19.4 ms')).toBeInTheDocument()
  })

  it('shows an actionable API error after a failed prediction', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch).mockImplementation((input) => {
      if (String(input).endsWith('/api/predict')) return json({ detail: 'Analyzer checkpoint is not trained.' }, false, 503)
      return baseFetch(input)
    })

    render(<App />)
    await user.upload(screen.getByLabelText(/upload image/i), new File(['x'], 'input.jpg', { type: 'image/jpeg' }))
    await user.click(screen.getByRole('button', { name: /run inference/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Analyzer checkpoint is not trained.')
  })

  it('navigates with the sidebar and preserves honest empty profiling states', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('link', { name: /model registry/i }))
    expect(await screen.findByRole('heading', { name: 'Model registry' })).toBeInTheDocument()
    expect(screen.getAllByText('Awaiting profile').length).toBeGreaterThan(0)

    await user.click(screen.getByRole('link', { name: /research results/i }))
    expect(await screen.findByRole('heading', { name: 'Research results' })).toBeInTheDocument()
    expect(screen.getByText(/no benchmark results yet/i)).toBeInTheDocument()

    await user.click(screen.getByRole('link', { name: /architecture/i }))
    expect(await screen.findByRole('heading', { name: 'System architecture' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /router stage/i }))
    expect(screen.getByText(/selects the feasible model/i)).toBeInTheDocument()
  })

  it('shows a clear empty state when live monitoring has no requests', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('link', { name: /live monitoring/i }))
    expect(await screen.findByRole('heading', { name: 'Live monitoring' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(/waiting for inference traffic/i)).toBeInTheDocument())
  })
})
