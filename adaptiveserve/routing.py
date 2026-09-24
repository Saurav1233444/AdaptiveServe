"""Constraint-aware threshold and learned routing policies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from .data import manifest_checksum


PROFILE_FIELDS = ("name", "accuracy", "latency_ms", "memory_mb")
CONTEXT_FIELDS = ("complexity", "latency_budget_ms", "cpu_available", "memory_budget_mb")


def load_profiles(source: str | Path | Sequence[Mapping[str, object]] | Mapping[str, object]) -> list[dict[str, object]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Calibrated model profiles not found: {path}")
        source = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(source, Mapping):
        source = source.get("models")  # type: ignore[assignment]
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)) or not source:
        raise ValueError("Profiles must contain a non-empty models list")
    profiles: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Profile {index} must be an object")
        missing = set(PROFILE_FIELDS).difference(raw)
        if missing:
            raise ValueError(f"Profile {index} is missing fields: {', '.join(sorted(missing))}")
        name = str(raw["name"])
        if not name or name in names:
            raise ValueError(f"Profile model names must be unique and non-empty: {name!r}")
        numeric: dict[str, float] = {}
        for field in PROFILE_FIELDS[1:]:
            value = raw[field]
            if value is None:
                raise ValueError(f"Profile {name} is not calibrated: {field} is null")
            try:
                numeric[field] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Profile {name} has invalid {field}") from exc
            if not math.isfinite(numeric[field]):
                raise ValueError(f"Profile {name} has non-finite {field}")
            if numeric[field] < 0.0:
                raise ValueError(f"Profile {name} has negative {field}")
        if not 0.0 <= numeric["accuracy"] <= 1.0:
            raise ValueError(f"Profile {name} accuracy must be in 0..1")
        profiles.append({"name": name, **numeric})
        names.add(name)
    return profiles


def validate_context(context: Mapping[str, object]) -> dict[str, float]:
    missing = set(CONTEXT_FIELDS).difference(context)
    if missing:
        raise ValueError(f"Routing context is missing: {', '.join(sorted(missing))}")
    try:
        values = {field: float(context[field]) for field in CONTEXT_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError("Routing context values must be numeric") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Routing context values must be finite")
    if not 0.0 <= values["complexity"] <= 1.0:
        raise ValueError("complexity must be in 0..1")
    if not 0.0 <= values["cpu_available"] <= 1.0:
        raise ValueError("cpu_available must be in 0..1")
    if values["latency_budget_ms"] < 0.0 or values["memory_budget_mb"] < 0.0:
        raise ValueError("latency and memory budgets must be non-negative")
    return values


def feasible_indices(profiles: Sequence[Mapping[str, object]], context: Mapping[str, float]) -> list[int]:
    if context["cpu_available"] <= 0.0:
        return []
    return [
        index
        for index, profile in enumerate(profiles)
        if float(profile["latency_ms"]) <= context["latency_budget_ms"]
        and float(profile["memory_mb"]) <= context["memory_budget_mb"]
    ]


def _fallback_index(profiles: Sequence[Mapping[str, object]]) -> int:
    return min(
        range(len(profiles)),
        key=lambda index: (float(profiles[index]["latency_ms"]), float(profiles[index]["memory_mb"])),
    )


class RuleRouter:
    def __init__(self, profiles: str | Path | Sequence[Mapping[str, object]] | Mapping[str, object]) -> None:
        self.profiles = load_profiles(profiles)

    def select(self, context: Mapping[str, object]) -> dict[str, object]:
        values = validate_context(context)
        feasible = feasible_indices(self.profiles, values)
        if not feasible:
            index = _fallback_index(self.profiles)
            return {
                "model": self.profiles[index]["name"],
                "reason": "No calibrated profile satisfies all resource constraints; fastest fallback violates constraints.",
                "constraint_satisfied": False,
            }
        preferred_name = (
            "mobilenet_v3_small"
            if values["complexity"] < 0.4
            else "resnet50"
            if values["complexity"] < 0.7
            else "efficientnet_b0"
        )
        preferred = next(
            (index for index in feasible if self.profiles[index]["name"] == preferred_name),
            None,
        )
        index = preferred if preferred is not None else max(
            feasible, key=lambda candidate: float(self.profiles[candidate]["accuracy"])
        )
        return {
            "model": self.profiles[index]["name"],
            "reason": (
                "Complexity threshold selected its preferred feasible model."
                if preferred is not None
                else "Threshold preference was infeasible or unavailable; selected the highest-accuracy feasible profile."
            ),
            "constraint_satisfied": True,
        }


class RouterNetwork(nn.Module):
    def __init__(self, model_count: int, *, hidden_dim: int = 16) -> None:
        super().__init__()
        self.model_count = model_count
        self.hidden_dim = hidden_dim
        self.hidden = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, model_count)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(contexts))


def normalize_context(context: Mapping[str, float], profiles: Sequence[Mapping[str, object]]) -> list[float]:
    max_latency = max(float(profile["latency_ms"]) for profile in profiles)
    max_memory = max(float(profile["memory_mb"]) for profile in profiles)
    return [
        context["complexity"],
        min(2.0, context["latency_budget_ms"] / max(max_latency, 1e-9)) / 2.0,
        context["cpu_available"],
        min(2.0, context["memory_budget_mb"] / max(max_memory, 1e-9)) / 2.0,
    ]


def save_router_checkpoint(
    network: RouterNetwork,
    path: str | Path,
    profiles: str | Path | Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    train_sample_ids: Iterable[str],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    calibrated = load_profiles(profiles)
    ids = sorted(set(train_sample_ids))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": "four_context_feature_mlp",
            "model_names": [profile["name"] for profile in calibrated],
            "hidden_dim": network.hidden_dim,
            "state_dict": network.state_dict(),
            "train_sample_ids": ids,
            "train_sample_ids_checksum": manifest_checksum(ids),
            "metadata": dict(metadata or {}),
        },
        destination,
    )
    return destination


class LearnedRouter:
    def __init__(
        self,
        checkpoint: str | Path,
        profiles: str | Path | Sequence[Mapping[str, object]] | Mapping[str, object],
        *,
        threads: int = 1,
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Trained router checkpoint not found at {checkpoint_path}. Run scripts/train.py first.")
        if threads < 1:
            raise ValueError("threads must be at least 1")
        torch.set_num_threads(threads)
        self.profiles = load_profiles(profiles)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        names = [profile["name"] for profile in self.profiles]
        if payload.get("model_names") != names:
            raise ValueError("Router checkpoint model order does not match calibrated profiles")
        self.network = RouterNetwork(len(names), hidden_dim=int(payload["hidden_dim"]))
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.metadata = payload

    def select(self, context: Mapping[str, object]) -> dict[str, object]:
        values = validate_context(context)
        feasible = feasible_indices(self.profiles, values)
        if not feasible:
            index = _fallback_index(self.profiles)
            return {
                "model": self.profiles[index]["name"],
                "reason": "No calibrated profile satisfies all resource constraints; fastest fallback violates constraints.",
                "constraint_satisfied": False,
            }
        tensor = torch.tensor([normalize_context(values, self.profiles)], dtype=torch.float32)
        with torch.inference_mode():
            logits = self.network(tensor)[0]
        mask = torch.full_like(logits, float("-inf"))
        mask[feasible] = logits[feasible]
        index = int(torch.argmax(mask).item())
        return {
            "model": self.profiles[index]["name"],
            "reason": "Learned utility selected the highest-scoring feasible model.",
            "constraint_satisfied": True,
        }
