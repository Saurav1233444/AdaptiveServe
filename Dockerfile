FROM node:22-bookworm-slim AS dashboard
WORKDIR /src/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.13
RUN useradd --create-home --uid 1000 app && mkdir /app && chown app:app /app
WORKDIR /app
COPY --chown=app:app pyproject.toml uv.lock requirements.txt ./
COPY --chown=app:app adaptiveserve/ adaptiveserve/
COPY --chown=app:app experiments/ experiments/
USER app
RUN uv sync --frozen --no-dev
COPY --chown=app:app CMakeLists.txt ./
COPY --chown=app:app core/ core/
COPY --chown=app:app workers/ workers/
COPY --chown=app:app tests/ tests/
COPY --chown=app:app scripts/ scripts/
RUN bash scripts/setup_native.sh && ctest --test-dir build --output-on-failure
COPY --chown=app:app configs/ configs/
COPY --chown=app:app models/ models/
COPY --chown=app:app results/demo/ results/latest/
COPY --chown=app:app frontend/public/results/ frontend/public/results/
COPY --chown=app:app --from=dashboard /src/frontend/dist frontend/dist/
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app/build" ADAPTIVESERVE_BACKEND="cpp" \
    ADAPTIVESERVE_THREADS="1" OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" \
    PORT="8000" ADAPTIVESERVE_CORS_ORIGINS="*" MPLCONFIGDIR="/tmp/matplotlib"
RUN mkdir -p history && python -c 'from adaptiveserve.service import InferenceService; from pathlib import Path; state = InferenceService(Path("/app")).health(); assert state["ready"], state'
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c 'import json,os,urllib.request; result=json.load(urllib.request.urlopen("http://127.0.0.1:"+os.getenv("PORT","8000")+"/health")); assert result["ready"]'
CMD ["bash", "scripts/start.sh"]
