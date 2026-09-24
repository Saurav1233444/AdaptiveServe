import csv
from pathlib import Path

import pytest
from PIL import Image

from adaptiveserve.data import (
    IMAGENETTE_WNID_TO_LABEL,
    load_manifest,
    manifest_checksum,
    write_imagenette_manifest,
)


def _image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (color, color, color)).save(path)


def test_imagenette_mapping_uses_imagenet_1000_class_indices() -> None:
    assert IMAGENETTE_WNID_TO_LABEL == {
        "n01440764": 0,
        "n02102040": 217,
        "n02979186": 482,
        "n03000684": 491,
        "n03028079": 497,
        "n03394916": 566,
        "n03417042": 569,
        "n03425413": 571,
        "n03445777": 574,
        "n03888257": 701,
    }


def test_manifest_splits_are_deterministic_disjoint_and_relative(tmp_path: Path) -> None:
    root = tmp_path / "imagenette2-160"
    for split in ("train", "val"):
        for wnid in IMAGENETTE_WNID_TO_LABEL:
            for index in range(6):
                _image(root / split / wnid / f"{index}.jpg", index * 10)

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_imagenette_manifest(root, first, train_count=8, calibration_count=4, test_count=4, seed=19)
    write_imagenette_manifest(root, second, train_count=8, calibration_count=4, test_count=4, seed=19)

    rows = load_manifest(first)
    assert [(r.sample_id, r.path, r.label, r.split) for r in rows] == [
        (r.sample_id, r.path, r.label, r.split) for r in load_manifest(second)
    ]
    assert {r.split for r in rows} == {"train", "calibration", "test"}
    assert {s: sum(r.split == s for r in rows) for s in ("train", "calibration", "test")} == {
        "train": 8,
        "calibration": 4,
        "test": 4,
    }
    assert len({r.sample_id for r in rows}) == len(rows)
    assert all(not Path(r.path).is_absolute() and r.absolute_path.exists() for r in rows)


def test_manifest_rejects_duplicate_image_across_splits(tmp_path: Path) -> None:
    image = tmp_path / "same.jpg"
    _image(image, 20)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "path", "label", "split"])
        writer.writerow(["a", "same.jpg", 0, "train"])
        writer.writerow(["b", "same.jpg", 0, "test"])

    with pytest.raises(ValueError, match="multiple splits"):
        load_manifest(manifest)


def test_manifest_checksum_is_order_independent(tmp_path: Path) -> None:
    ids = ["three", "one", "two"]
    assert manifest_checksum(ids) == manifest_checksum(reversed(ids))
