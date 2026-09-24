#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export ADAPTIVESERVE_BACKEND=cpp
export ADAPTIVESERVE_THREADS=${ADAPTIVESERVE_THREADS:-1}
exec .venv/bin/python -m uvicorn adaptiveserve.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
