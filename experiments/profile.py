"""Measure teachers on training and calibration images in isolated processes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from adaptiveserve.runtime import ModelRuntime, ROOT
from experiments.common import manifest_rows, provenance


def run_model(args):
    runtime = ModelRuntime(args.root / "configs/models.json", args.threads, args.backend)
    samples = manifest_rows(args.manifest, ["train", "calibration"])
    rows = []
    with Image.open(samples.iloc[0].path) as source:
        for _ in range(args.warmup):
            runtime.predict(args.model, source.convert("RGB"))
    for index, item in enumerate(samples.itertuples()):
        with Image.open(item.path) as source:
            result = runtime.predict(args.model, source.convert("RGB"), label=int(item.label))
        rows.append(dict(sample_id=item.sample_id, split=item.split, model=args.model,
                         label=int(item.label), prediction=result["class_index"],
                         **{key: result[key] for key in ["confidence", "true_probability", "inference_ms", "latency_ms", "memory_mb", "cpu_percent"]}))
        if (index + 1) % 50 == 0:
            print(f"{args.model}: measured {index + 1}/{len(samples)}", flush=True)
    pd.DataFrame(rows).to_csv(args.output / f"{args.model}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/profile")
    parser.add_argument("--backend", choices=["cpp", "python"], default="cpp")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 1:
        parser.error("threads and warmup must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.model:
        run_model(args)
        return
    registry = json.loads((args.root / "configs/models.json").read_text())["models"]
    for model in registry:
        subprocess.run([sys.executable, "-m", "experiments.profile", "--model", model["name"],
                        "--manifest", str(args.manifest), "--root", str(args.root), "--output", str(args.output),
                        "--backend", args.backend, "--threads", str(args.threads), "--warmup", str(args.warmup)], check=True)
    observations = pd.concat([pd.read_csv(args.output / f"{model['name']}.csv") for model in registry], ignore_index=True)
    observations.to_csv(args.output / "observations.csv", index=False)
    # Profiles use calibration images only; test labels never inform selection.
    profiles = []
    for name, group in observations[observations.split == "calibration"].groupby("model", sort=False):
        profiles.append({"name": name, "accuracy": float(np.mean(group.label == group.prediction)),
                         "latency_ms": float(group.latency_ms.median()), "memory_mb": float(group.memory_mb.max())})
    if len(profiles) != len(registry):
        raise ValueError("Calibration split must contain measurements for every model")
    metadata = provenance(args.manifest, args.root, args.backend, args.threads, args.seed)
    metadata.update(memory_definition="Per-model isolated process RSS, including Python and native runtime", warmup=args.warmup,
                    calibration_sample_ids=sorted(observations.loc[observations.split == "calibration", "sample_id"].unique().tolist()))
    (args.root / "models/profiles.json").write_text(json.dumps({"models": profiles, "metadata": metadata}, indent=2))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(observations)} actual observations and calibration profiles", flush=True)


if __name__ == "__main__":
    main()
