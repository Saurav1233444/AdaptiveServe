import csv
from pathlib import Path

import pytest
from PIL import Image

from adaptiveserve.complexity import (
    ComplexityAnalyzer,
    ComplexityPredictor,
    build_difficulty_targets,
    save_complexity_checkpoint,
)


MODELS = ("mobilenet_v3_small", "resnet50", "efficientnet_b0")


def test_difficulty_is_mean_true_probability_deficit_for_all_teachers(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,path,label,split\na,img.jpg,7,train\n", encoding="utf-8")
    observations = tmp_path / "observations.csv"
    with observations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "model", "true_probability"])
        writer.writeheader()
        for model, probability in zip(MODELS, (0.2, 0.5, 0.8), strict=True):
            writer.writerow({"sample_id": "a", "model": model, "true_probability": probability})

    assert build_difficulty_targets(manifest, observations) == {"a": pytest.approx(0.5)}


def test_difficulty_requires_every_teacher_observation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,path,label,split\na,img.jpg,7,train\n", encoding="utf-8")
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "sample_id,model,true_probability\na,mobilenet_v3_small,0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing teacher observations"):
        build_difficulty_targets(manifest, observations)


def test_analyzer_loads_trained_checkpoint_and_returns_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "complexity.pt"
    model = ComplexityPredictor()
    save_complexity_checkpoint(model, checkpoint, train_sample_ids=["sample-b", "sample-a"])
    analyzer = ComplexityAnalyzer(checkpoint)

    result = analyzer.analyze(Image.new("RGB", (48, 32), "navy"))
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["features"]) == 6
    assert len(result["embedding"]) == model.embedding_dim
    assert result["analyzer_ms"] >= 0.0
    assert result["trained"] is True


def test_analyzer_missing_checkpoint_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        ComplexityAnalyzer(tmp_path / "missing.pt")
