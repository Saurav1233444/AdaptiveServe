# AdaptiveServe deployment

## Status

Current deployment target: Vercel frontend and Railway Docker backend.

- Frontend: https://frontend-snowy-alpha-ccjmtcpoyz.vercel.app
- Backend: https://backend-production-7762.up.railway.app
- Railway project: `2e6f26ee-24aa-436e-a91e-41e85a6312a0`
- Railway service: `backend`, production environment.

Both deployments were verified on 2026-09-23. Railway reports `SUCCESS`; the
public API reports `ready: true` and `backend: cpp`. Live checks passed for all
five inference policies, three available models, metrics, eight result charts,
CORS, invalid-image rejection, five frontend routes, and the frontend's API
origin. `deployment-status.json` contains the live smoke-test results.

## Redeploy Railway

Authenticate with `npx --yes @railway/cli login --browserless`. Reuse the existing
package staging function so ignored model files are included while training data,
credentials, and local environments are excluded:

```bash
# Run from the repository root.
package=$(mktemp -d /tmp/adaptiveserve-railway-XXXXXX)
.venv/bin/python -c 'from pathlib import Path; import sys; from scripts.deploy_space import stage; stage(Path(sys.argv[1]))' "$package"
cd "$package"
npx --yes @railway/cli link --project 2e6f26ee-24aa-436e-a91e-41e85a6312a0 --environment production --service backend
npx --yes @railway/cli up --detach
npx --yes @railway/cli service status --json
npx --yes @railway/cli logs --build --lines 100
```

Railway detects the root Dockerfile. The public domain targets port `8000`;
the start script honors Railway's `PORT`. Keep the ONNX models and trained
checkpoints in the package. The Vercel project's production `VITE_API_URL` is
already set to the backend origin above. Redeploy the frontend from `frontend/`
with `npx --yes vercel deploy --prod --yes`.

The following Hugging Face instructions are an alternative deployment path,
not the current backend host.

## Platforms

- Frontend: Vercel Hobby, React/Vite production build.
- Backend: Hugging Face Docker Space on `cpu-basic`. Its documented free CPU
  allocation accommodates the existing PyTorch analyzer and C++ ONNX engine.
  Render Free's 512 MB is below the observed serving peak, so it is not the
  deployment target. No paid hardware is requested.
- Models: all three existing ONNX files and both trained checkpoints are uploaded
  to the Space's model storage and copied into the image. Total deployment source
  package is about 130 MiB. No training or weight download runs at startup.

References: [Space hardware](https://huggingface.co/docs/hub/spaces-overview),
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[Render pricing](https://render.com/pricing),
[Vercel CLI](https://vercel.com/docs/cli).

## Deploy backend

Run from the repository root. `uv`, Node.js/npm, and accounts on both platforms
are required. Authenticate locally; do not commit credentials.

```bash
uvx --from huggingface_hub hf auth login
# Replace YOUR_ACCOUNT with your Hugging Face account name.
uv run --no-project --python 3.12 --with huggingface_hub scripts/deploy_space.py --repo YOUR_ACCOUNT/adaptiveserve
```

The script uploads only app sources, the three ONNX models, two trained
checkpoints, labels, measured profiles/results, and frontend assets. It excludes
training images, SQLite history, local environments and credentials. Large model
uploads are handled by the Hub client. Source and prepared model files become
public. Existing Space files are not automatically deleted.

Open the printed Space URL and wait for its Docker build to reach `Running`.
Copy its actual `https://...hf.space` direct URL as BACKEND_URL below. The returned
host is also saved in `deployment-status.json` when supplied by the provider.
The Space also serves a complete fallback dashboard at its root.

## Deploy Vercel

```bash
# Set this to the actual running Space origin, with no trailing path.
export BACKEND_URL='https://YOUR-ACTUAL-SPACE.hf.space'
curl --fail "$BACKEND_URL/health"
npx vercel login
cd frontend
# Follow prompts to create/link a Vercel Hobby project.
printf '%s\n' "$BACKEND_URL" | npx vercel env add VITE_API_URL production
npx vercel --prod --build-env "VITE_API_URL=$BACKEND_URL"
```

If the project has not yet been linked, run `npx vercel link` before `env add`.
If `VITE_API_URL` already exists, use `vercel env update VITE_API_URL production`.
Record the actual production URL printed by Vercel. For an anyone-accessible
public demo, ensure deployment protection is disabled for the production domain.
`frontend/vercel.json` provides SPA routes and refuses to deploy with a missing,
local, or non-HTTPS backend origin. `frontend/.env.production` stays empty for
same-origin Docker serving; the Vercel environment overrides it at build time.

## Environment

| Variable | Platform | Value |
|---|---|---|
| `VITE_API_URL` | Vercel production/build | Actual backend HTTPS origin, no `/api` suffix |
| `ADAPTIVESERVE_CORS_ORIGINS` | Backend | `*` for public, cookieless demo; or comma-separated exact frontend origins |
| `PORT` | Backend | `8000` (matches Space `app_port`) |
| `ADAPTIVESERVE_BACKEND` | Backend | `cpp` |
| `ADAPTIVESERVE_THREADS` | Backend | `1` |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | Backend | `1` |
| `HF_TOKEN` | Local deployment only, optional | Write token, or use `hf auth login` |
| `VERCEL_TOKEN` | Local CI only, optional | CLI token, or use `vercel login` |

No token is needed by the deployed inference application. Settings are in
`.env.example`; the start script runs one Uvicorn worker and honors `PORT`.

## Test production

```bash
export BACKEND_URL='https://YOUR-ACTUAL-SPACE.hf.space'
export FRONTEND_URL='https://YOUR-ACTUAL-PROJECT.vercel.app'
curl --fail "$BACKEND_URL/health"
curl --fail "$BACKEND_URL/api/models"
curl --fail -F 'file=@/absolute/path/to/image.jpg' -F 'policy=learned' "$BACKEND_URL/api/predict"
curl --fail "$BACKEND_URL/api/metrics"
curl --fail "$BACKEND_URL/api/results"
curl --fail -X OPTIONS -H "Origin: $FRONTEND_URL" -H 'Access-Control-Request-Method: POST' "$BACKEND_URL/api/predict"
```

Health must include `ready: true` and `backend: cpp`. Prediction must include
`complexity`, `selected_model`, `prediction`, `confidence`, `latency_ms`, and
`backend: cpp`. Test `policy=rule`, `mobilenet_v3_small`, `resnet50`, and
`efficientnet_b0` as well. Interactive API documentation is at `/docs`.

Open the Vercel URL on desktop/mobile, upload an image, verify loading/error
states, check the prediction and monitoring count, and refresh each dashboard
page. The eight result charts must load from the backend origin.

## Local validation

```bash
.venv/bin/python scripts/deploy_space.py --dry-run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
npm --prefix frontend test -- --run
npm --prefix frontend run build
# Explicit external-origin build; this does not publish anything.
VITE_API_URL=https://backend.example npm --prefix frontend run build
# Requires working Docker socket access:
docker build -t adaptiveserve-demo .
docker run --rm -p 8000:8000 adaptiveserve-demo
```

The Docker image includes model artifacts and validates readiness during build.
The Dockerfile, not `requirements.txt`, is the supported native deployment path.
Native tests run during the image build. Docker execution on the preparation
machine was blocked by OS socket permissions and sudo required a password.

Free hosting can sleep and cold-start slowly. SQLite metrics are ephemeral and
reset on restart. Uploaded image bytes are not persisted. Retained research
charts are measurements from the original local benchmark, not cloud benchmarks.

Preparation checks passed: 40 Python tests, 6 frontend tests, production build
with an external HTTPS origin, missing-origin deployment rejection, Python lint,
start-script syntax, and 129.4 MiB upload-package validation. Railway built the
Docker image, passed all five native tests and the build-time readiness check,
and successfully started the service. Public production smoke tests passed as
listed above. Browser interaction was not tested because Firefox was not
available to the browser automation tool.
