"""Training pipelines for the complexity predictor and learned router."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .complexity import (
    ComplexityAnalyzer,
    ComplexityPredictor,
    build_difficulty_targets,
    image_to_tensor,
    save_complexity_checkpoint,
)
from .data import ManifestSample, load_manifest
from .features import extract_image_features
from .routing import RouterNetwork, load_profiles, normalize_context, save_router_checkpoint


OBSERVATION_FIELDS = (
    "sample_id",
    "model",
    "label",
    "prediction",
    "confidence",
    "true_probability",
    "inference_ms",
    "latency_ms",
    "memory_mb",
    "cpu_percent",
)


@dataclass(frozen=True)
class Observation:
    sample_id: str
    model: str
    label: int
    prediction: int
    confidence: float
    true_probability: float
    inference_ms: float
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    split: str | None = None

    @property
    def correct(self) -> bool:
        return self.label == self.prediction


@dataclass(frozen=True)
class RouterExample:
    sample_id: str
    context: tuple[float, float, float, float]
    target_model_index: int


def load_observations(
    path: str | Path,
    *,
    allowed_sample_ids: set[str] | None = None,
) -> list[Observation]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Profile observations not found: {source}")
    observations: list[Observation] = []
    seen: set[tuple[str, str]] = set()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(OBSERVATION_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Observations are missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row["sample_id"] or "").strip()
            if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                continue
            model = (row["model"] or "").strip()
            key = (sample_id, model)
            if not sample_id or not model:
                raise ValueError(f"Empty sample_id or model on observations line {line_number}")
            if key in seen:
                raise ValueError(f"Duplicate observation for sample {sample_id}, model {model}")
            try:
                integer_values = {field: int(row[field]) for field in ("label", "prediction")}
                float_values = {
                    field: float(row[field])
                    for field in (
                        "confidence",
                        "true_probability",
                        "inference_ms",
                        "latency_ms",
                        "memory_mb",
                        "cpu_percent",
                    )
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on observations line {line_number}") from exc
            if not all(math.isfinite(value) for value in float_values.values()):
                raise ValueError(f"Non-finite value on observations line {line_number}")
            if not 0 <= integer_values["label"] <= 999 or not 0 <= integer_values["prediction"] <= 999:
                raise ValueError(f"Labels and predictions must be in 0..999 on observations line {line_number}")
            if not 0.0 <= float_values["confidence"] <= 1.0 or not 0.0 <= float_values["true_probability"] <= 1.0:
                raise ValueError(f"Probabilities must be in 0..1 on observations line {line_number}")
            if any(float_values[field] < 0.0 for field in ("inference_ms", "latency_ms", "memory_mb", "cpu_percent")):
                raise ValueError(f"Measurements must be non-negative on observations line {line_number}")
            observations.append(
                Observation(
                    sample_id=sample_id,
                    model=model,
                    **integer_values,
                    **float_values,
                    split=(row.get("split") or "").strip() or None,
                )
            )
            seen.add(key)
    if not observations:
        raise ValueError("No observations remain after filtering")
    return observations


def build_router_examples(
    observations: Sequence[Observation],
    difficulties: Mapping[str, float],
    profiles_source: str | Path | Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    contexts_per_sample: int = 16,
    seed: int = 0,
) -> list[RouterExample]:
    if contexts_per_sample < 1:
        raise ValueError("contexts_per_sample must be at least 1")
    profiles = load_profiles(profiles_source)
    model_names = [str(profile["name"]) for profile in profiles]
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in difficulties.values()):
        raise ValueError("Difficulty scores must be finite values in 0..1")
    grouped: dict[str, dict[str, Observation]] = {}
    for row in observations:
        grouped.setdefault(row.sample_id, {})[row.model] = row
    rng = random.Random(seed)
    examples: list[RouterExample] = []
    for sample_id in sorted(difficulties):
        if sample_id not in grouped:
            raise ValueError(f"No router observations for training sample {sample_id}")
        missing = set(model_names).difference(grouped[sample_id])
        if missing:
            raise ValueError(f"Sample {sample_id} is missing router observations: {', '.join(sorted(missing))}")
        by_model = [grouped[sample_id][name] for name in model_names]
        latencies = [row.latency_ms for row in by_model]
        memories = [row.memory_mb for row in by_model]
        for _ in range(contexts_per_sample):
            context = {
                "complexity": float(difficulties[sample_id]),
                "latency_budget_ms": rng.uniform(min(latencies) * 0.75, max(latencies) * 1.25),
                "cpu_available": rng.uniform(0.25, 1.0),
                "memory_budget_mb": rng.uniform(min(memories) * 0.75, max(memories) * 1.25),
            }
            feasible = [
                index
                for index, row in enumerate(by_model)
                if row.latency_ms <= context["latency_budget_ms"]
                and row.memory_mb <= context["memory_budget_mb"]
                and row.cpu_percent <= context["cpu_available"] * 100.0
            ]
            if feasible:
                correct = [index for index in feasible if by_model[index].correct]
                candidates = correct or feasible
                target = min(
                    candidates,
                    key=lambda index: (
                        by_model[index].latency_ms,
                        by_model[index].memory_mb,
                        -by_model[index].true_probability,
                    ),
                )
            else:
                target = min(
                    range(len(by_model)),
                    key=lambda index: (by_model[index].latency_ms, by_model[index].memory_mb),
                )
            normalized = normalize_context(context, profiles)
            examples.append(RouterExample(sample_id, tuple(normalized), target))
    return examples


class _DifficultyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: Sequence[ManifestSample], targets: Mapping[str, float]) -> None:
        self.rows = list(rows)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row.absolute_path) as image:
            image.load()
            tensor = image_to_tensor(image)
            features = torch.from_numpy(extract_image_features(image))
        return tensor, features, torch.tensor(self.targets[row.sample_id], dtype=torch.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train_complexity_predictor(
    manifest_path: str | Path,
    observations_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> Path:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    _seed_everything(seed)
    train_rows = [row for row in load_manifest(manifest_path) if row.split == "train"]
    if not train_rows:
        raise ValueError("Manifest contains no training samples")
    targets = build_difficulty_targets(manifest_path, observations_path)
    dataset = _DifficultyDataset(train_rows, targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    model = ComplexityPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        loss_sum = 0.0
        item_count = 0
        for images, features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(images, features)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            item_count += len(target)
        final_loss = loss_sum / item_count
    return save_complexity_checkpoint(
        model,
        output_path,
        train_sample_ids=[row.sample_id for row in train_rows],
        metadata={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed, "final_mse": final_loss},
    )


def train_learned_router(
    observations_path: str | Path,
    difficulties: Mapping[str, float],
    profiles_source: str | Path | Sequence[Mapping[str, object]] | Mapping[str, object],
    output_path: str | Path,
    *,
    train_sample_ids: Iterable[str],
    epochs: int = 10,
    contexts_per_sample: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> Path:
    if epochs < 1 or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive")
    _seed_everything(seed)
    ids = set(train_sample_ids)
    profiles = load_profiles(profiles_source)
    observations = load_observations(observations_path, allowed_sample_ids=ids)
    examples = build_router_examples(
        observations,
        {sample_id: difficulties[sample_id] for sample_id in sorted(ids)},
        profiles,
        contexts_per_sample=contexts_per_sample,
        seed=seed,
    )
    inputs = torch.tensor([example.context for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.target_model_index for example in examples], dtype=torch.long)
    network = RouterNetwork(len(profiles))
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0
    network.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = network(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return save_router_checkpoint(
        network,
        output_path,
        profiles,
        train_sample_ids=ids,
        metadata={
            "epochs": epochs,
            "contexts_per_sample": contexts_per_sample,
            "learning_rate": learning_rate,
            "seed": seed,
            "training_examples": len(examples),
            "final_cross_entropy": final_loss,
            "target_policy": "correct_then_observed_latency_memory",
            "resource_contexts": (
                "Latency, memory, and CPU budgets are seeded simulations; targets use per-image "
                "observed correctness, latency, RSS memory, and CPU percent."
            ),
        },
    )


def train_artifacts(
    manifest_path: str | Path,
    observations_path: str | Path,
    profiles_path: str | Path,
    *,
    complexity_output: str | Path = "models/complexity_predictor.pt",
    router_output: str | Path = "models/router.pt",
    epochs: int = 10,
    complexity_epochs: int | None = None,
    router_epochs: int | None = None,
    batch_size: int = 32,
    contexts_per_sample: int = 16,
    seed: int = 0,
) -> tuple[Path, Path]:
    complexity_path = train_complexity_predictor(
        manifest_path,
        observations_path,
        complexity_output,
        epochs=complexity_epochs or epochs,
        batch_size=batch_size,
        seed=seed,
    )
    train_rows = [row for row in load_manifest(manifest_path) if row.split == "train"]
    analyzer = ComplexityAnalyzer(complexity_path)
    difficulties: dict[str, float] = {}
    for row in train_rows:
        with Image.open(row.absolute_path) as image:
            image.load()
            difficulties[row.sample_id] = float(analyzer.analyze(image)["score"])
    router_path = train_learned_router(
        observations_path,
        difficulties,
        profiles_path,
        router_output,
        train_sample_ids=[row.sample_id for row in train_rows],
        epochs=router_epochs or epochs,
        contexts_per_sample=contexts_per_sample,
        seed=seed,
    )
    return complexity_path, router_path
