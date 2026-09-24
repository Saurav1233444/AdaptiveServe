import csv
from pathlib import Path

import torch
from PIL import Image

from adaptiveserve.data import manifest_checksum
from adaptiveserve.training import (
    OBSERVATION_FIELDS,
    build_router_examples,
    load_observations,
    train_artifacts,
)


MODELS = ("mobilenet_v3_small", "resnet50", "efficientnet_b0")


def _observation_file(path: Path) -> None:
    fields = [
        "sample_id", "model", "label", "prediction", "confidence", "true_probability",
        "inference_ms", "latency_ms", "memory_mb", "cpu_percent", "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id, split in (("train-a", "train"), ("test-a", "test")):
            for index, model in enumerate(MODELS):
                writer.writerow({
                    "sample_id": sample_id,
                    "model": model,
                    "label": 4,
                    "prediction": 4 if index > 0 else 3,
                    "confidence": 0.8,
                    "true_probability": 0.1 + 0.3 * index,
                    "inference_ms": 5 + 5 * index,
                    "latency_ms": 6 + 5 * index,
                    "memory_mb": 50 + 50 * index,
                    "cpu_percent": 50 + 10 * index,
                    "split": split,
                })


def test_observation_loader_filters_manifest_train_ids_not_csv_split_hint(tmp_path: Path) -> None:
    path = tmp_path / "observations.csv"
    _observation_file(path)
    observations = load_observations(path, allowed_sample_ids={"train-a"})
    assert {row.sample_id for row in observations} == {"train-a"}


def test_router_examples_are_reproducible_and_derive_labels_from_observations(tmp_path: Path) -> None:
    path = tmp_path / "observations.csv"
    _observation_file(path)
    rows = load_observations(path, allowed_sample_ids={"train-a"})
    profiles = [
        {"name": name, "accuracy": 0.5, "latency_ms": 20.0, "memory_mb": 200.0}
        for name in MODELS
    ]
    first = build_router_examples(rows, {"train-a": 0.6}, profiles, contexts_per_sample=5, seed=7)
    second = build_router_examples(rows, {"train-a": 0.6}, profiles, contexts_per_sample=5, seed=7)
    assert first == second
    assert len(first) == 5
    assert all(example.sample_id == "train-a" for example in first)
    assert all(0 <= example.target_model_index < 3 for example in first)


def test_training_metadata_checksum_records_exact_ids() -> None:
    assert manifest_checksum(["train-b", "train-a"]) == manifest_checksum(["train-a", "train-b"])


def test_train_artifacts_records_only_manifest_train_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest_fields = ["sample_id", "path", "label", "split"]
    samples = [("train-a", "train.jpg", "train"), ("held-out", "test.jpg", "test")]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for sample_id, image_path, split in samples:
            Image.new("RGB", (20, 20), "purple").save(tmp_path / image_path)
            writer.writerow({"sample_id": sample_id, "path": image_path, "label": 0, "split": split})
    observations = tmp_path / "observations.csv"
    fields = [*OBSERVATION_FIELDS, "split"]
    with observations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id, _, split in samples:
            for index, model in enumerate(MODELS):
                writer.writerow({
                    "sample_id": sample_id, "model": model, "label": 0, "prediction": index,
                    "confidence": 0.5, "true_probability": 0.4, "inference_ms": 2 + index,
                    "latency_ms": 3 + index, "memory_mb": 40 + index, "cpu_percent": 50,
                    "split": split,
                })
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        '{"models": ['
        '{"name":"mobilenet_v3_small","accuracy":0.5,"latency_ms":3,"memory_mb":40},'
        '{"name":"resnet50","accuracy":0.6,"latency_ms":4,"memory_mb":41},'
        '{"name":"efficientnet_b0","accuracy":0.7,"latency_ms":5,"memory_mb":42}'
        ']}'
    )
    complexity = tmp_path / "complexity.pt"
    router = tmp_path / "router.pt"

    train_artifacts(
        manifest, observations, profiles, complexity_output=complexity, router_output=router,
        epochs=1, batch_size=1, contexts_per_sample=2, seed=3,
    )

    for path in (complexity, router):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["train_sample_ids"] == ["train-a"]
        assert payload["train_sample_ids_checksum"] == manifest_checksum(["train-a"])
