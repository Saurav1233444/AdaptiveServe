"""Experiment inputs and provenance; no artificial observations are supported."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import psutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(path, splits):
    path = Path(path).resolve()
    data = pd.read_csv(path, dtype={"sample_id": str})
    if not {"sample_id", "path", "label", "split"} <= set(data.columns):
        raise ValueError("Manifest requires sample_id,path,label,split")
    if data.sample_id.duplicated().any() or data.path.duplicated().any():
        raise ValueError("Manifest contains duplicate IDs or image paths")
    if not data.label.between(0, 999).all() or not (data.label == data.label.astype(int)).all():
        raise ValueError("ImageNet labels must be integers in 0..999")
    rows = data[data.split.isin(splits)].copy()
    if rows.empty:
        raise ValueError(f"No dataset samples for splits {splits}")
    rows["path"] = rows.path.map(lambda value: str((path.parent / value).resolve()))
    return rows


def provenance(manifest, root, backend, threads, seed):
    root = Path(root)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except subprocess.SubprocessError:
        revision = "unavailable"
    versions = subprocess.check_output([sys.executable, "-c", "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ['torch','torchvision','onnxruntime','numpy']}))"], text=True)
    sources = sorted([*root.glob("adaptiveserve/*.py"), *root.glob("experiments/*.py"),
                      *root.glob("core/**/*.cpp"), *root.glob("core/**/*.hpp"), root / "configs/models.json"])
    return {"manifest_sha256": sha256(manifest), "seed": seed, "backend": backend,
            "threads": threads, "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(), "python": sys.version, "versions": json.loads(versions),
            "git_revision": revision, "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sources},
            "model_sha256": {p.name: sha256(p) for p in sorted((root / "models").glob("*.onnx"))}}
