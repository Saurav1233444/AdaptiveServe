import '@testing-library/jest-dom/vitest'

Object.defineProperty(URL, 'createObjectURL', {
  writable: true,
  value: () => 'blob:test-preview',
})

Object.defineProperty(URL, 'revokeObjectURL', {
  writable: true,
  value: () => undefined,
})

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverStub
