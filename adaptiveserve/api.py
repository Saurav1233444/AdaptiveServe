"""Local research API; missing artifacts are explicit setup states."""
import io
import json
import os
import threading
import warnings
from pathlib import Path

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from adaptiveserve.feedback import FeedbackStore
from adaptiveserve.runtime import ROOT
from adaptiveserve.service import InferenceService


def create_app(root=ROOT, service=None):
    root = Path(root)
    app = FastAPI(title="AdaptiveServe", version="0.1.0")
    origins = [origin.strip() for origin in os.getenv('ADAPTIVESERVE_CORS_ORIGINS', '').split(',') if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=['GET', 'POST'], allow_headers=['Content-Type'],
                           allow_credentials=False)
    service = service or InferenceService(root, os.getenv("ADAPTIVESERVE_BACKEND", "cpp"), int(os.getenv("ADAPTIVESERVE_THREADS", "1")))
    feedback = FeedbackStore(root / "history/feedback.sqlite")
    admission = threading.BoundedSemaphore(4)

    @app.get("/health")
    @app.get("/api/health")
    def health():
        return service.health()

    @app.get("/api/models")
    def models():
        return {"models": service.models()}

    @app.get("/api/metrics")
    def metrics():
        return dict(feedback.metrics(), cpu_percent=psutil.cpu_percent(), memory_mb=psutil.Process().memory_info().rss / 1024**2)

    @app.get("/api/results")
    def results():
        summary_path = root / "results/latest/summary.json"
        if not summary_path.exists():
            return {"available": False, "summary": [], "plots": [], "metadata": {}}
        report = json.loads(summary_path.read_text())
        return dict(report, available=True, plots=[{"name": path.stem.replace("_", " "), "url": f"/results/{path.name}"}
                    for path in sorted((root / "frontend/public/results").glob("*.png"))])

    @app.post("/api/predict")
    def predict(file: UploadFile = File(...), policy: str = Form("learned"),
                latency_budget_ms: float = Form(50, gt=0, le=60000),
                memory_budget_mb: float = Form(2048, gt=0, le=1048576)):
        if not admission.acquire(blocking=False):
            raise HTTPException(429, "Request queue is full; retry shortly")
        try:
            payload = file.file.read(10 * 1024 * 1024 + 1)
            if len(payload) > 10 * 1024 * 1024:
                raise HTTPException(413, "Images must be at most 10 MiB")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(payload)) as source:
                        if source.width * source.height > 20_000_000:
                            raise ValueError("Image exceeds 20 megapixels")
                        image = ImageOps.exif_transpose(source).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise HTTPException(400, f"Invalid image: {exc}") from exc
            try:
                result = service.predict(image, policy, latency_budget_ms, memory_budget_mb)
            except (FileNotFoundError, RuntimeError) as exc:
                raise HTTPException(503, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            feedback.record(result)
            return result
        finally:
            admission.release()

    plot_dir = root / "frontend/public/results"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dist = root / "frontend/dist"
    if dist.exists():
        # Register client routes before the /results asset mount so direct
        # navigation and refresh render React, including the results page.
        def dashboard():
            return FileResponse(dist / "index.html")

        for route in ["/models", "/monitoring", "/results", "/architecture"]:
            app.add_api_route(route, dashboard, methods=["GET"], include_in_schema=False)
    app.mount("/results", StaticFiles(directory=plot_dir), name="results")
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return app


app = create_app()
