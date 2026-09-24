import json
from pathlib import Path

import torch

from adaptiveserve.export import MODEL_SPECS, export_models


class _TinyClassifier(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        pooled = tensor.mean(dim=(2, 3))
        return pooled.mean(dim=1, keepdim=True).repeat(1, 1000)


def test_model_specs_use_imagenet_v1_documented_transforms() -> None:
    assert [(spec.name, spec.resize_size, spec.interpolation, spec.weights) for spec in MODEL_SPECS] == [
        ("mobilenet_v3_small", 256, "bilinear", "IMAGENET1K_V1"),
        ("resnet50", 256, "bilinear", "IMAGENET1K_V1"),
        ("efficientnet_b0", 256, "bicubic", "IMAGENET1K_V1"),
    ]


def test_export_writes_ordered_registry_and_1000_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "adaptiveserve.export.load_pretrained_model",
        lambda spec: (_TinyClassifier().eval(), [f"class-{i}" for i in range(1000)]),
    )
    monkeypatch.setattr("adaptiveserve.export._export_onnx", lambda model, path: path.write_bytes(b"onnx"))
    registry = tmp_path / "configs" / "models.json"
    labels = tmp_path / "models" / "labels.json"
    reports = export_models(tmp_path / "models", registry, labels, validate=False)

    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert [model["name"] for model in payload["models"]] == [spec.name for spec in MODEL_SPECS]
    assert all(model["accuracy"] is None and model["latency_ms"] is None for model in payload["models"])
    assert all(model["memory_mb"] is None and model["input_size"] == 224 for model in payload["models"])
    assert all(model["endpoint"] == "/api/predict" for model in payload["models"])
    assert all(model["framework"] == "ONNX Runtime" for model in payload["models"])
    assert len(json.loads(labels.read_text(encoding="utf-8"))) == 1000
    assert len(reports) == 3
