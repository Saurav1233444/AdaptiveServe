# AdaptiveServe research console

React, TypeScript, Tailwind CSS, and Recharts frontend for the AdaptiveServe API.

```bash
npm install
npm run dev
```

The Vite development server runs on `http://localhost:5173` and proxies same-origin `/api` requests to `http://localhost:8000`. Start the FastAPI service separately on port 8000.

```bash
npm test
npm run build
```

The interface does not bundle example benchmark results or generated plots. Registry measurements and research views stay in an explicit empty state until the API reports real profiling and experiment artifacts.
