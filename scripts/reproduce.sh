#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --frozen --python 3.12 --extra dev
bash scripts/setup_native.sh
.venv/bin/python scripts/prepare_data.py --train-count 300 --calibration-count 100 --test-count 100 --seed 42
.venv/bin/python scripts/export_models.py
.venv/bin/python -m experiments.profile --threads 1 --seed 42
.venv/bin/python scripts/train.py --complexity-epochs 30 --router-epochs 200 --seed 42
.venv/bin/python -m experiments.evaluate --threads 1 --repeats 3 --seed 42
cd frontend
npm ci
npm run build
