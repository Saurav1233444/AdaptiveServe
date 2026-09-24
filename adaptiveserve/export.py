"""Export supported torchvision classifiers to ONNX with parity validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    input_size: int
    resize_size: int
    interpolation: str
    weights: str


MODEL_SPECS = (
    ModelSpec("mobilenet_v3_small", "MobileNetV3 Small", 224, 256, "bilinear", "IMAGENET1K_V1"),
    ModelSpec("resnet50", "ResNet-50", 224, 256, "bilinear", "IMAGENET1K_V1"),
    ModelSpec("efficientnet_b0", "EfficientNet-B0", 224, 256, "bicubic", "IMAGENET1K_V1"),
)


def load_pretrained_model(spec: ModelSpec) -> tuple[nn.Module, list[str]]:
    from torchvision.models import (
        EfficientNet_B0_Weights, MobileNet_V3_Small_Weights, ResNet50_Weights,
        efficientnet_b0, mobilenet_v3_small, resnet50,
    )
    factories = {
        "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
        "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    }
    factory, weights = factories[spec.name]
    transform = weights.transforms()
    resize_size = int(transform.resize_size[0])
    crop_size = int(transform.crop_size[0])
    interpolation = transform.interpolation.value
    if (crop_size, resize_size, interpolation) != (
        spec.input_size,
        spec.resize_size,
        spec.interpolation,
    ):
        raise RuntimeError(
            f"{spec.name} preprocessing differs from registry spec: "
            f"crop={crop_size}, resize={resize_size}, interpolation={interpolation}"
        )
    model = factory(weights=weights).cpu().eval()
    categories = list(weights.meta["categories"])
    if len(categories) != 1000:
        raise RuntimeError(f"{spec.name} weights did not provide 1,000 ImageNet labels")
    return model, categories


def _export_onnx(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros((1, 3, 224, 224), dtype=torch.float32), path,
        input_names=["input"], output_names=["logits"], opset_version=17,
        dynamo=False, do_constant_folding=True,
    )


def validate_onnx_parity(model: nn.Module, path: str | Path) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to validate exported model parity") from exc
    input_array = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(input_array)).detach().cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: input_array})[0]
    if expected.shape != (1, 1000) or actual.shape != (1, 1000):
        raise RuntimeError(f"Expected 1,000-class logits, got PyTorch {expected.shape} and ONNX {actual.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    difference = np.abs(actual - expected)
    return {"max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean())}


def export_models(
    output_dir: str | Path = "models",
    registry_path: str | Path = "configs/models.json",
    labels_path: str | Path = "models/labels.json",
    *, validate: bool = True,
) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    output = Path(output_dir).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    labels = Path(labels_path).expanduser().resolve()
    repository_root = registry.parent.parent
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    registry_models: list[dict[str, object]] = []
    expected_labels: list[str] | None = None
    for spec in MODEL_SPECS:
        model, categories = load_pretrained_model(spec)
        if expected_labels is None:
            expected_labels = categories
        elif categories != expected_labels:
            raise RuntimeError(f"ImageNet label order differs for {spec.name}")
        onnx_path = output / f"{spec.name}.onnx"
        _export_onnx(model, onnx_path)
        parity = validate_onnx_parity(model, onnx_path) if validate else {}
        reports.append({"name": spec.name, "onnx_path": str(onnx_path), **parity})
        registry_models.append({
            "name": spec.name, "display_name": spec.display_name,
            "onnx_path": Path(os.path.relpath(onnx_path, repository_root)).as_posix(),
            "input_size": spec.input_size, "resize_size": spec.resize_size,
            "interpolation": spec.interpolation, "accuracy": None,
            "latency_ms": None, "memory_mb": None, "input_type": "float32_nchw",
            "endpoint": "/api/predict", "framework": "ONNX Runtime", "weights": spec.weights,
        })
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps(expected_labels, indent=2) + "\n", encoding="utf-8")
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({"models": registry_models}, indent=2) + "\n", encoding="utf-8")
    return reports
