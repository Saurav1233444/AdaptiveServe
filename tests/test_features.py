import numpy as np
from PIL import Image

from adaptiveserve.features import FEATURE_NAMES, extract_image_features


def test_features_have_six_named_finite_normalized_values() -> None:
    pixels = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
    image = Image.fromarray(np.stack([pixels, pixels, pixels], axis=-1), "RGB")
    features = extract_image_features(image)

    assert FEATURE_NAMES == (
        "resolution",
        "aspect_ratio",
        "entropy",
        "edge_density",
        "texture",
        "noise",
    )
    assert features.shape == (6,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert ((features >= 0.0) & (features <= 1.0)).all()


def test_edges_and_entropy_distinguish_checkerboard_from_flat_image() -> None:
    flat = Image.new("RGB", (64, 64), "gray")
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    checker_image = Image.fromarray(np.stack([checker] * 3, axis=-1), "RGB")

    flat_features = extract_image_features(flat)
    checker_features = extract_image_features(checker_image)
    assert checker_features[2] > flat_features[2]
    assert checker_features[3] > flat_features[3]

