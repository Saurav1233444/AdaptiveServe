export type NullableNumber = number | null

export interface HealthResponse {
  status: string
  backend: string
  ready: boolean
  detail: string | null
}

export interface ModelInfo {
  name: string
  display_name: string
  accuracy: NullableNumber
  latency_ms: NullableNumber
  memory_mb: NullableNumber
  input_type: string
  endpoint: string
  available: boolean
  loaded: boolean
}

export interface ModelsResponse {
  models: ModelInfo[]
}

export interface PredictionResponse {
  request_id: string
  complexity: number
  selected_model: string
  reason: string
  prediction: string
  class_index: number
  confidence: number
  latency_ms: number
  inference_ms: number
  analyzer_ms: number
  cpu_percent: number
  memory_mb: number
  constraint_satisfied: boolean
  backend: string
}

export interface RecentMetric {
  request_id?: string
  timestamp?: string
  model?: string
  selected_model?: string
  latency_ms?: number
  complexity?: number
  confidence?: number
  constraint_satisfied?: boolean
  [key: string]: unknown
}

export interface MetricsResponse {
  requests: number
  average_latency_ms: NullableNumber
  current_model: string | null
  cpu_percent: NullableNumber
  memory_mb: NullableNumber
  recent: RecentMetric[]
  model_counts: Record<string, number>
}

export interface ResultSummary {
  policy: string
  n: number
  accuracy: number
  latency_ms: number
  p95_latency_ms: number
  throughput_rps: number
  memory_mb: number
  cpu_percent: number
  sla_violation_rate: number
}

export interface ResultsResponse {
  available: boolean
  summary: ResultSummary[]
  plots: Array<{ name: string; url: string }>
  metadata: Record<string, unknown>
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseError(response: Response) {
  try {
    const body = await response.json() as { detail?: string; message?: string }
    return body.detail || body.message || `Request failed (${response.status}).`
  } catch {
    return `Request failed (${response.status}).`
  }
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(backendUrl(path), init)
  } catch {
    throw new ApiError('Could not reach the AdaptiveServe API. Check that the backend is running.', 0)
  }
  if (!response.ok) throw new ApiError(await parseError(response), response.status)
  try {
    return await response.json() as T
  } catch {
    throw new ApiError('The API returned an unreadable response.', response.status)
  }
}

export const getHealth = () => apiRequest<HealthResponse>('/api/health')
export const getModels = () => apiRequest<ModelsResponse>('/api/models')
export const getMetrics = () => apiRequest<MetricsResponse>('/api/metrics')
export async function getResults() {
  const report = await apiRequest<ResultsResponse>('/api/results')
  return { ...report, plots: report.plots.map(plot => ({ ...plot, url: backendUrl(plot.url) })) }
}

function backendUrl(path: string) {
  const origin = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
  return /^https?:\/\//.test(path) ? path : `${origin}${path}`
}

export function runPrediction(data: FormData) {
  return apiRequest<PredictionResponse>('/api/predict', { method: 'POST', body: data })
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'An unexpected error occurred.'
}
