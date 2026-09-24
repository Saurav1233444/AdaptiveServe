"""Dataset download and manifest utilities."""

from __future__ import annotations

import csv
import hashlib
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_WNID_TO_LABEL = {
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
VALID_SPLITS = frozenset({"train", "calibration", "test"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    path: str
    label: int
    split: str
    absolute_path: Path


def manifest_checksum(sample_ids: Iterable[str]) -> str:
    """Return a stable checksum for a set of sample IDs."""
    payload = "\n".join(sorted(str(value) for value in sample_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | Path, *, check_files: bool = True) -> list[ManifestSample]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    rows: list[ManifestSample] = []
    ids: set[str] = set()
    image_splits: dict[Path, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "path", "label", "split"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        for line_number, raw in enumerate(reader, start=2):
            sample_id = (raw["sample_id"] or "").strip()
            relative = (raw["path"] or "").strip()
            split = (raw["split"] or "").strip()
            if not sample_id or not relative:
                raise ValueError(f"Manifest line {line_number} has an empty sample_id or path")
            if sample_id in ids:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            if Path(relative).is_absolute():
                raise ValueError(f"Manifest paths must be relative: {relative}")
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid split {split!r} on manifest line {line_number}")
            try:
                label = int(raw["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ImageNet label on manifest line {line_number}") from exc
            if not 0 <= label <= 999:
                raise ValueError(f"ImageNet label must be in 0..999 on manifest line {line_number}")
            absolute = (manifest_path.parent / relative).resolve()
            previous_split = image_splits.get(absolute)
            if previous_split is not None and previous_split != split:
                raise ValueError(f"Image {relative} occurs in multiple splits")
            if check_files and not absolute.is_file():
                raise FileNotFoundError(f"Manifest image not found: {absolute}")
            ids.add(sample_id)
            image_splits[absolute] = split
            rows.append(ManifestSample(sample_id, relative, label, split, absolute))
    if not rows:
        raise ValueError("Dataset manifest is empty")
    return rows


def _images_under(root: Path, source_split: str) -> list[tuple[Path, int]]:
    images: list[tuple[Path, int]] = []
    split_root = root / source_split
    for wnid, label in IMAGENETTE_WNID_TO_LABEL.items():
        class_root = split_root / wnid
        if not class_root.is_dir():
            raise FileNotFoundError(f"Imagenette class directory not found: {class_root}")
        images.extend(
            (path.resolve(), label)
            for path in sorted(class_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return images


def _shuffle(items: Sequence[tuple[Path, int]], seed: int) -> list[tuple[Path, int]]:
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


def write_imagenette_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    *,
    train_count: int | None = None,
    calibration_count: int | None = None,
    test_count: int | None = None,
    seed: int = 0,
) -> Path:
    """Create deterministic disjoint splits while preserving Imagenette's held-out val set."""
    root = Path(dataset_root).expanduser().resolve()
    output = Path(manifest_path).expanduser().resolve()
    training_pool = _shuffle(_images_under(root, "train"), seed)
    test_pool = _shuffle(_images_under(root, "val"), seed + 1)
    if calibration_count is None:
        calibration_count = max(1, round(len(training_pool) * 0.2))
    if train_count is None:
        train_count = len(training_pool) - calibration_count
    if test_count is None:
        test_count = len(test_pool)
    counts = (train_count, calibration_count, test_count)
    if any(count < 0 for count in counts):
        raise ValueError("Split counts must be non-negative")
    if train_count + calibration_count > len(training_pool):
        raise ValueError("Requested train and calibration counts exceed Imagenette train images")
    if test_count > len(test_pool):
        raise ValueError("Requested test count exceeds Imagenette validation images")
    selected = [
        *((path, label, "train") for path, label in training_pool[:train_count]),
        *((path, label, "calibration") for path, label in training_pool[train_count : train_count + calibration_count]),
        *((path, label, "test") for path, label in test_pool[:test_count]),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "path", "label", "split"])
        writer.writeheader()
        for path, label, split in selected:
            identity = path.relative_to(root).as_posix()
            writer.writerow(
                {
                    "sample_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                    "path": Path(os.path.relpath(path, output.parent)).as_posix(),
                    "label": label,
                    "split": split,
                }
            )
    load_manifest(output)
    return output


def download_imagenette(destination: str | Path, *, url: str = IMAGENETTE_URL) -> Path:
    destination_path = Path(destination).expanduser().resolve()
    dataset_root = destination_path / "imagenette2-160"
    if dataset_root.is_dir():
        return dataset_root
    destination_path.mkdir(parents=True, exist_ok=True)
    archive = destination_path / "imagenette2-160.tgz"
    if not archive.is_file():
        urllib.request.urlretrieve(url, archive)  # noqa: S310 - fixed HTTPS default URL
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination_path / member.name).resolve()
            if destination_path not in target.parents and target != destination_path:
                raise ValueError(f"Unsafe path in Imagenette archive: {member.name}")
        handle.extractall(destination_path, members=members, filter="data")
    if not dataset_root.is_dir():
        raise RuntimeError(f"Downloaded archive did not contain {dataset_root.name}")
    return dataset_root


def prepare_imagenette(
    destination: str | Path,
    manifest_path: str | Path,
    **split_options: int | None,
) -> Path:
    root = download_imagenette(destination)
    return write_imagenette_manifest(root, manifest_path, **split_options)
