"""Shared preprocessing and explicit native/Python ONNX execution adapters."""
from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def read_registry(path):
    models = json.loads(Path(path).read_text())["models"]
    names = [model["name"] for model in models]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names in registry")
    return models


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Model logits must be nonempty and finite")
    exps = np.exp(values - values.max())
    return exps / exps.sum()


class ModelRuntime:
    def __init__(self, config_path=ROOT / "configs/models.json", threads=1, backend="cpp"):
        if backend not in {"cpp", "python"}:
            raise ValueError("backend must be cpp or python")
        if threads < 1:
            raise ValueError("threads must be positive")
        self.config_path = Path(config_path).resolve()
        self.root = self.config_path.parent.parent
        self.models = {item["name"]: item for item in read_registry(self.config_path)}
        self.backend, self.threads = backend, threads
        self._engine = None
        self._sessions = {}
        self._lock = threading.RLock()
        labels = self.root / "models/labels.json"
        self.labels = json.loads(labels.read_text()) if labels.exists() else None

    def path(self, name):
        return self.root / self.models[name]["onnx_path"]

    def preprocess(self, name, image):
        # Match torchvision's weight-specific resize, center crop and normalization.
        profile = self.models[name]
        image = ImageOps.exif_transpose(image).convert("RGB")
        size, resize = profile["input_size"], profile["resize_size"]
        width, height = image.size
        if width < height:
            target = (resize, int(resize * height / width))
        else:
            target = (int(resize * width / height), resize)
        interpolation = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}
        image = image.resize(target, interpolation[profile["interpolation"]])
        left, top = int(round((image.width - size) / 2)), int(round((image.height - size) / 2))
        pixels = np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.float32) / 255
        pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])

    def _native(self):
        with self._lock:
            if self._engine is None:
                sys.path.insert(0, str(ROOT / "build"))
                try:
                    native = importlib.import_module("adaptive_core")
                except ImportError as exc:
                    raise RuntimeError("Build the C++ engine with scripts/setup_native.sh before inference") from exc
                self._engine = native.Engine(str(self.config_path), self.threads)
            return self._engine

    def _python_predict(self, name, tensor):
        import onnxruntime as ort
        load_ms = 0.0
        with self._lock:
            if name not in self._sessions:
                start = time.perf_counter()
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                self._sessions[name] = ort.InferenceSession(str(self.path(name)), options, providers=["CPUExecutionProvider"])
                load_ms = (time.perf_counter() - start) * 1000
            session = self._sessions[name]
        process = psutil.Process()
        cpu_start, start = time.process_time(), time.perf_counter()
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        elapsed = time.perf_counter() - start
        return {"logits": output.reshape(-1), "inference_ms": elapsed * 1000,
                "load_ms": load_ms, "memory_mb": process.memory_info().rss / 1024**2,
                "cpu_percent": (time.process_time() - cpu_start) / max(elapsed, 1e-9) * 100}

    def predict(self, name, image, label=None):
        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")
        if not self.path(name).is_file():
            raise FileNotFoundError(f"Missing {self.path(name)}; run python scripts/export_models.py")
        start = time.perf_counter()
        tensor = self.preprocess(name, image)
        result = self._native().predict(name, tensor) if self.backend == "cpp" else self._python_predict(name, tensor)
        probs = probabilities(result.pop("logits"))
        index = int(probs.argmax())
        result.update(class_index=index, prediction=self.labels[index] if self.labels else f"ImageNet class {index}",
                      confidence=float(probs[index]), latency_ms=(time.perf_counter() - start) * 1000,
                      backend=self.backend, selected_model=name)
        if label is not None:
            result["true_probability"] = float(probs[label])
        return result

    def loaded_models(self):
        if self.backend == "cpp":
            return list(self._engine.loaded_models()) if self._engine else []
        return list(self._sessions)

    def unload(self, name):
        with self._lock:
            if self.backend == "cpp" and self._engine:
                self._engine.unload(name)
            self._sessions.pop(name, None)
