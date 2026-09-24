"""The request path shared by the HTTP API and held-out evaluation."""
import json
import time
import uuid
from pathlib import Path

import psutil

from adaptiveserve.runtime import ModelRuntime


class InferenceService:
    def __init__(self, root, backend="cpp", threads=1):
        self.root, self.backend, self.threads = Path(root), backend, threads
        self.runtime = None
        self.analyzer = None
        self.routers = {}
        self.profiles = []

    def _runtime(self):
        if self.runtime is None:
            self.runtime = ModelRuntime(self.root / "configs/models.json", self.threads, self.backend)
        return self.runtime

    def health(self):
        try:
            runtime = self._runtime()
            missing = [str(runtime.path(name)) for name in runtime.models if not runtime.path(name).exists()]
            if missing:
                raise FileNotFoundError("Export pretrained models with scripts/export_models.py")
            if self.backend == "cpp":
                runtime._native()
            for artifact in ["complexity_predictor.pt", "router.pt", "profiles.json"]:
                if not (self.root / "models" / artifact).exists():
                    raise FileNotFoundError(f"Missing models/{artifact}; run profiling and scripts/train.py")
            return {"status": "ok", "ready": True, "backend": self.backend, "detail": "Artifacts available"}
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            return {"status": "setup_required", "ready": False, "backend": self.backend, "detail": str(exc)}

    def models(self):
        try:
            runtime = self._runtime()
        except FileNotFoundError:
            return []
        profiles_path = self.root / "models/profiles.json"
        measured = {p["name"]: p for p in json.loads(profiles_path.read_text())["models"]} if profiles_path.exists() else {}
        return [dict(item, **{k: v for k, v in measured.get(name, {}).items() if k != "name"},
                     available=runtime.path(name).exists(), loaded=name in runtime.loaded_models())
                for name, item in runtime.models.items()]

    def _adaptive(self, policy):
        from adaptiveserve.complexity import ComplexityAnalyzer
        from adaptiveserve.routing import LearnedRouter, RuleRouter
        if self.analyzer is None:
            self.analyzer = ComplexityAnalyzer(self.root / "models/complexity_predictor.pt")
        if policy not in self.routers:
            self.profiles = json.loads((self.root / "models/profiles.json").read_text())["models"]
            self.routers[policy] = (RuleRouter(self.profiles) if policy == "rule" else
                                    LearnedRouter(self.root / "models/router.pt", self.profiles))
        return self.analyzer, self.routers[policy]

    def predict(self, image, policy="learned", latency_budget_ms=50.0, memory_budget_mb=2048.0, cpu_available=None):
        start, cpu_start = time.perf_counter(), time.process_time()
        runtime = self._runtime()
        analysis = {"score": None, "analyzer_ms": 0.0}
        if policy in {"rule", "learned"}:
            analyzer, router = self._adaptive(policy)
            analysis = analyzer.analyze(image)
            # Observed host load is a context signal; the request never changes host limits.
            available = max(.01, 1 - psutil.cpu_percent() / 100) if cpu_available is None else cpu_available
            context = {"complexity": analysis["score"], "latency_budget_ms": latency_budget_ms,
                       "cpu_available": available, "memory_budget_mb": memory_budget_mb}
            selection = router.select(context)
        elif policy in runtime.models:
            selection = {"model": policy, "reason": "Static baseline: every image uses this model", "constraint_satisfied": True}
        else:
            raise ValueError(f"Unknown policy: {policy}")
        result = runtime.predict(selection["model"], image)
        elapsed = (time.perf_counter() - start) * 1000
        result.update(request_id=uuid.uuid4().hex, complexity=analysis["score"], reason=selection["reason"],
                      analyzer_ms=analysis["analyzer_ms"], latency_ms=elapsed, policy=policy,
                      cpu_percent=(time.process_time() - cpu_start) / max(elapsed / 1000, 1e-9) * 100,
                      memory_mb=psutil.Process().memory_info().rss / 1024**2,
                      latency_budget_ms=latency_budget_ms, memory_budget_mb=memory_budget_mb,
                      profile_constraint_satisfied=bool(selection["constraint_satisfied"]))
        result["constraint_satisfied"] = bool(selection["constraint_satisfied"] and elapsed <= latency_budget_ms and result["memory_mb"] <= memory_budget_mb)
        return result
