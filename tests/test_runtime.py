import json

import numpy as np
import pytest
from PIL import Image

from adaptiveserve.runtime import ModelRuntime, probabilities, read_registry


def test_stable_probabilities_reject_invalid_logits():
    np.testing.assert_allclose(probabilities([1000, 1000]), [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        probabilities([float("nan")])


def test_registry_rejects_duplicate_models(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"models": [{"name": "x"}, {"name": "x"}]}))
    with pytest.raises(ValueError, match="Duplicate"):
        read_registry(config)


def test_preprocessing_and_missing_artifact(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "models.json"
    config.write_text(json.dumps({"models": [{"name": "tiny", "onnx_path": "models/missing.onnx",
        "input_size": 224, "resize_size": 256, "interpolation": "bilinear"}]}))
    runtime = ModelRuntime(config, backend="python")
    tensor = runtime.preprocess("tiny", Image.new("RGB", (400, 300)))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    with pytest.raises(FileNotFoundError, match="export"):
        runtime.predict("tiny", Image.new("RGB", (20, 20)))


def test_runtime_never_silently_changes_backend(tmp_path):
    config = tmp_path / "models.json"
    config.write_text('{"models": []}')
    with pytest.raises(ValueError, match="backend"):
        ModelRuntime(config, backend="invented")
