"""Deterministic, bounded image statistics for complexity prediction."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


FEATURE_NAMES = ("resolution", "aspect_ratio", "entropy", "edge_density", "texture", "noise")


def extract_image_features(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    gray = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    probabilities = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities /= probabilities.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / 8.0)
    dx = np.abs(np.diff(gray, axis=1)).mean() / 255.0
    dy = np.abs(np.diff(gray, axis=0)).mean() / 255.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    values = np.asarray(
        [
            min(1.0, math.log2(max(1, width * height)) / 24.0),
            min(width, height) / max(width, height),
            entropy,
            min(1.0, float((dx + dy) / 2.0)),
            min(1.0, float(gray.std() / 127.5)),
            min(1.0, float(np.mean(np.abs(laplacian)) / 510.0)),
        ],
        dtype=np.float32,
    )
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
