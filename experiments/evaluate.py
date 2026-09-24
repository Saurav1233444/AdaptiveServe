"""Evaluate static and adaptive policies on held-out images with real inference."""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from adaptiveserve.feedback import FeedbackStore
from adaptiveserve.runtime import ROOT
from adaptiveserve.service import InferenceService
from experiments.common import manifest_rows, provenance


def validate_evaluation_ids(test_ids, train_ids):
    overlap = set(test_ids) & set(train_ids)
    if overlap:
        raise ValueError(f"Train/calibration and test IDs overlap: {sorted(overlap)[:3]}")


def summarize(rows):
    if rows.empty:
        raise ValueError("Cannot summarize an empty experiment")
    summaries = []
    for name, group in rows.groupby("policy", sort=False):
        accuracy, n = float(group.correct.mean()), len(group)
        # Repeated timings do not create additional independent accuracy samples.
        unique = group.drop_duplicates("sample_id") if "sample_id" in group else group
        n_accuracy = len(unique)
        z, denominator = 1.96, 1 + 1.96**2 / n_accuracy
        center = (accuracy + z**2 / (2 * n_accuracy)) / denominator
        radius = z * math.sqrt(accuracy * (1 - accuracy) / n_accuracy + z**2 / (4 * n_accuracy**2)) / denominator
        summaries.append({"policy": name, "n": n, "unique_images": n_accuracy, "accuracy": accuracy,
                          "accuracy_ci95": [max(0, center - radius), min(1, center + radius)],
                          "latency_ms": float(group.latency_ms.mean()), "p95_latency_ms": float(group.latency_ms.quantile(.95)),
                          "throughput_rps": float(1000 / group.latency_ms.mean()),
                          "memory_mb": float(group.memory_mb.mean()), "cpu_percent": float(group.cpu_percent.mean()),
                          "sla_violation_rate": float((group.latency_ms > group.latency_budget_ms).mean())})
    return summaries


def worker(args):
    service = InferenceService(args.root, args.backend, args.threads)
    samples = manifest_rows(args.manifest, ["test"])
    # Warm caches before measuring; adaptive resident memory includes all candidates.
    runtime = service._runtime()
    names = list(runtime.models) if args.policy in {"rule", "learned"} else [args.policy]
    with Image.open(samples.iloc[0].path) as source:
        image = source.convert("RGB")
    for name in names:
        for _ in range(args.warmup):
            runtime.predict(name, image)
    for _ in range(args.warmup):
        service.predict(image, args.policy, args.latency_budget_ms, args.memory_budget_mb, args.cpu_available)
    rows = []
    store = FeedbackStore(args.root / "history/benchmark.sqlite")
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    for repetition in range(args.repeats):
        for item in samples.iloc[rng.permutation(len(samples))].itertuples():
            with Image.open(item.path) as source:
                image = source.convert("RGB")
            result = service.predict(image, args.policy, args.latency_budget_ms, args.memory_budget_mb, args.cpu_available)
            store.record(result)
            result.update(sample_id=item.sample_id, repetition=repetition, label=int(item.label),
                          correct=int(result["class_index"] == int(item.label)))
            rows.append(result)
    wall = time.perf_counter() - start
    pd.DataFrame(rows).to_csv(args.output / f"{args.policy}.csv", index=False)
    (args.output / f"{args.policy}.timing.json").write_text(json.dumps({"wall_seconds": wall, "requests": len(rows)}))
    print(f"{args.policy}: {len(rows)} requests, accuracy={np.mean([r['correct'] for r in rows]):.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/latest")
    parser.add_argument("--backend", choices=["cpp", "python"], default="cpp")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-budget-ms", type=float, default=50)
    parser.add_argument("--memory-budget-mb", type=float, default=2048)
    parser.add_argument("--cpu-available", type=float, default=1.0, help="Controlled context; does not throttle CPU")
    parser.add_argument("--policy", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.warmup, args.repeats, args.latency_budget_ms, args.memory_budget_mb) <= 0 or not 0 < args.cpu_available <= 1:
        parser.error("Counts/budgets must be positive and CPU availability in (0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.policy:
        worker(args)
        return
    # Fail before any test inference when checkpoints were trained on test IDs.
    import torch
    samples = manifest_rows(args.manifest, ["test"])
    for artifact in ["complexity_predictor.pt", "router.pt"]:
        checkpoint = torch.load(args.root / "models" / artifact, map_location="cpu", weights_only=True)
        ids = checkpoint.get("train_sample_ids", checkpoint.get("metadata", {}).get("train_sample_ids"))
        if ids is None:
            raise ValueError(f"{artifact} lacks training provenance")
        validate_evaluation_ids(samples.sample_id, ids)
    profile = json.loads((args.root / "models/profiles.json").read_text())
    validate_evaluation_ids(samples.sample_id, profile["metadata"]["calibration_sample_ids"])
    policies = [m["name"] for m in json.loads((args.root / "configs/models.json").read_text())["models"]] + ["rule", "learned"]
    # Randomize policy order; each policy runs in a fresh process for honest RSS.
    order = np.random.default_rng(args.seed).permutation(policies)
    for policy in order:
        subprocess.run([sys.executable, "-m", "experiments.evaluate", "--policy", policy,
                        "--manifest", str(args.manifest), "--root", str(args.root), "--output", str(args.output),
                        "--backend", args.backend, "--threads", str(args.threads), "--warmup", str(args.warmup),
                        "--repeats", str(args.repeats), "--seed", str(args.seed), "--latency-budget-ms", str(args.latency_budget_ms),
                        "--memory-budget-mb", str(args.memory_budget_mb), "--cpu-available", str(args.cpu_available)], check=True)
    rows = pd.concat([pd.read_csv(args.output / f"{p}.csv") for p in policies], ignore_index=True)
    rows.to_csv(args.output / "observations.csv", index=False)
    summary = summarize(rows)
    for item in summary:
        timing = json.loads((args.output / f"{item['policy']}.timing.json").read_text())
        item["service_throughput_rps"] = item["throughput_rps"]
        item["throughput_rps"] = timing["requests"] / timing["wall_seconds"]
    metadata = provenance(args.manifest, args.root, args.backend, args.threads, args.seed)
    metadata.update(created_at=datetime.now(timezone.utc).isoformat(), warmup=args.warmup, repeats=args.repeats,
                    test_images=len(samples), latency_budget_ms=args.latency_budget_ms, memory_budget_mb=args.memory_budget_mb,
                    cpu_available=args.cpu_available, policy_order=order.tolist(),
                    context_definition="CPU availability and budgets are controlled routing inputs; no hardware throttling",
                    memory_definition="Full process RSS; each policy isolated, adaptive caches all candidate sessions",
                    throughput_definition="Sequential batch-one wall throughput including image decode and SQLite feedback",
                    latency_definition="Warm request: analysis, routing, preprocessing, ONNX, softmax; excludes file decode and feedback",
                    limitation="Small ImageNet subset; observational laptop benchmark, no guarantee of adaptive superiority")
    adaptive = next(row for row in summary if row["policy"] == "learned")
    improvements = [{"baseline": row["policy"], "latency_improvement_pct": 100 * (row["latency_ms"] - adaptive["latency_ms"]) / row["latency_ms"],
                     "accuracy_difference_pp": 100 * (adaptive["accuracy"] - row["accuracy"])} for row in summary if row["policy"] not in {"learned", "rule"}]
    report = {"summary": summary, "metadata": metadata, "improvements": improvements,
              "selection_counts": rows.groupby(["policy", "selected_model"]).size().unstack(fill_value=0).to_dict(orient="index")}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    columns = {"latency": ["policy", "sample_id", "latency_ms", "inference_ms", "analyzer_ms"],
               "accuracy": ["policy", "sample_id", "label", "class_index", "correct"],
               "memory": ["policy", "sample_id", "memory_mb", "cpu_percent"],
               "model_selection": ["policy", "sample_id", "selected_model", "complexity", "reason"]}
    for name, fields in columns.items():
        rows[fields].to_csv(args.output / f"{name}.csv", index=False)
        rows[fields].to_csv(args.root / "results" / f"{name}.csv", index=False)
    from experiments.plot import generate_plots
    generate_plots(args.output, args.root / "frontend/public/results")
    print(json.dumps({"summary": summary, "improvements": improvements}, indent=2))


if __name__ == "__main__":
    main()
