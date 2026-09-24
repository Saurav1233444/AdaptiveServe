import { afterEach, expect, it, vi } from 'vitest'
import { getHealth, getResults } from './api'

afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals() })

it('uses the configured backend for API calls and research images', async () => {
  vi.stubEnv('VITE_API_URL', 'https://backend.example/')
  const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ ready: true })))
    .mockResolvedValueOnce(new Response(JSON.stringify({ available: true, plots: [{ name: 'CPU', url: '/results/cpu_usage.png' }] })))
  vi.stubGlobal('fetch', fetchMock)
  await getHealth()
  expect(fetchMock.mock.calls[0][0]).toBe('https://backend.example/api/health')
  const report = await getResults()
  expect(report.plots[0].url).toBe('https://backend.example/results/cpu_usage.png')
})
