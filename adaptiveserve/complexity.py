"""Learned image difficulty model and checkpoint interface."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn

from .data import load_manifest, manifest_checksum
from .features import FEATURE_NAMES, extract_image_features


TEACHER_MODELS = ("mobilenet_v3_small", "resnet50", "efficientnet_b0")
ANALYZER_IMAGE_SIZE = 64


class ComplexityPredictor(nn.Module):
    """A compact CNN fused with six deterministic image statistics."""

    def __init__(self, *, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.stat_encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 8), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(24, embedding_dim), nn.ReLU())
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = self.image_encoder(images)
        stat_embedding = self.stat_encoder(features)
        embedding = self.fusion(torch.cat((image_embedding, stat_embedding), dim=1))
        score = torch.sigmoid(self.output(embedding)).squeeze(1)
        return score, embedding


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB").resize(
        (ANALYZER_IMAGE_SIZE, ANALYZER_IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    array = (array - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
        (0.229, 0.224, 0.225), dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def build_difficulty_targets(
    manifest_path: str | Path,
    observations_path: str | Path,
    *,
    teacher_models: Iterable[str] = TEACHER_MODELS,
) -> dict[str, float]:
    """Build train-only targets as mean true-class probability deficits."""
    train_ids = {row.sample_id for row in load_manifest(manifest_path, check_files=False) if row.split == "train"}
    required_models = tuple(teacher_models)
    values: dict[str, dict[str, float]] = {sample_id: {} for sample_id in train_ids}
    with Path(observations_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"sample_id", "model", "true_probability"}
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Observations are missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row["sample_id"] or "").strip()
            model = (row["model"] or "").strip()
            if sample_id not in train_ids or model not in required_models:
                continue
            if model in values[sample_id]:
                raise ValueError(f"Duplicate observation for sample {sample_id}, model {model}")
            try:
                probability = float(row["true_probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid true_probability on observations line {line_number}") from exc
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"true_probability must be in 0..1 on observations line {line_number}")
            values[sample_id][model] = probability
    targets: dict[str, float] = {}
    for sample_id in sorted(train_ids):
        missing_models = set(required_models).difference(values[sample_id])
        if missing_models:
            raise ValueError(
                f"Sample {sample_id} is missing teacher observations: {', '.join(sorted(missing_models))}"
            )
        targets[sample_id] = float(
            np.mean([1.0 - values[sample_id][model] for model in required_models], dtype=np.float64)
        )
    return targets


def save_complexity_checkpoint(
    model: ComplexityPredictor,
    path: str | Path,
    *,
    train_sample_ids: Iterable[str],
    metadata: dict[str, object] | None = None,
) -> Path:
    destination = Path(path)
    ids = sorted(set(train_sample_ids))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": "small_cnn_plus_six_statistics",
        "embedding_dim": model.embedding_dim,
        "feature_names": list(FEATURE_NAMES),
        "state_dict": model.state_dict(),
        "train_sample_ids": ids,
        "train_sample_ids_checksum": manifest_checksum(ids),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, destination)
    return destination


class ComplexityAnalyzer:
    def __init__(self, checkpoint: str | Path, *, threads: int = 1) -> None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained complexity checkpoint not found at {path}. Run scripts/train.py first."
            )
        if threads < 1:
            raise ValueError("threads must be at least 1")
        torch.set_num_threads(threads)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Complexity checkpoint feature schema does not match this package")
        self.model = ComplexityPredictor(embedding_dim=int(payload["embedding_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.metadata = payload

    def analyze(self, image: Image.Image) -> dict[str, object]:
        started = time.perf_counter()
        feature_values = extract_image_features(image)
        image_tensor = image_to_tensor(image).unsqueeze(0)
        feature_tensor = torch.from_numpy(feature_values).unsqueeze(0)
        with torch.inference_mode():
            score, embedding = self.model(image_tensor, feature_tensor)
        return {
            "score": float(score.item()),
            "features": [float(value) for value in feature_values],
            "embedding": [float(value) for value in embedding[0].tolist()],
            "analyzer_ms": (time.perf_counter() - started) * 1000.0,
            "trained": True,
        }
