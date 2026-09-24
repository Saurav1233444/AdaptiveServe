from pathlib import Path

import pytest
import torch

from adaptiveserve.routing import LearnedRouter, RouterNetwork, RuleRouter, save_router_checkpoint


PROFILES = [
    {"name": "mobilenet_v3_small", "accuracy": 0.67, "latency_ms": 8.0, "memory_mb": 80.0},
    {"name": "resnet50", "accuracy": 0.76, "latency_ms": 25.0, "memory_mb": 180.0},
    {"name": "efficientnet_b0", "accuracy": 0.78, "latency_ms": 18.0, "memory_mb": 130.0},
]


def test_rule_router_masks_profiles_that_violate_budgets() -> None:
    result = RuleRouter(PROFILES).select(
        {"complexity": 0.99, "latency_budget_ms": 20.0, "cpu_available": 1.0, "memory_budget_mb": 150.0}
    )
    assert result["model"] == "efficientnet_b0"
    assert result["constraint_satisfied"] is True
    assert "feasible" in result["reason"]


def test_rule_router_reports_fallback_constraint_violation() -> None:
    result = RuleRouter(PROFILES).select(
        {"complexity": 0.5, "latency_budget_ms": 1.0, "cpu_available": 0.0, "memory_budget_mb": 10.0}
    )
    assert result["model"] == "mobilenet_v3_small"
    assert result["constraint_satisfied"] is False
    assert "violat" in result["reason"]


@pytest.mark.parametrize(
    ("complexity", "expected"),
    [
        (0.399999, "mobilenet_v3_small"),
        (0.4, "resnet50"),
        (0.699999, "resnet50"),
        (0.7, "efficientnet_b0"),
    ],
)
def test_rule_router_named_threshold_boundaries(complexity: float, expected: str) -> None:
    result = RuleRouter(PROFILES).select(
        {"complexity": complexity, "latency_budget_ms": 100.0, "cpu_available": 1.0, "memory_budget_mb": 500.0}
    )
    assert result["model"] == expected


def test_router_rejects_uncalibrated_profiles() -> None:
    profiles = [{**PROFILES[0], "latency_ms": None}]
    with pytest.raises(ValueError, match="calibrated"):
        RuleRouter(profiles)


def test_learned_router_applies_hard_mask_after_network_choice(tmp_path: Path) -> None:
    path = tmp_path / "router.pt"
    network = RouterNetwork(model_count=3)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.output.bias[:] = torch.tensor([0.0, 1.0, 10.0])
    save_router_checkpoint(network, path, PROFILES, train_sample_ids=["train-1"])

    result = LearnedRouter(path, PROFILES).select(
        {"complexity": 0.7, "latency_budget_ms": 10.0, "cpu_available": 1.0, "memory_budget_mb": 100.0}
    )
    assert result["model"] == "mobilenet_v3_small"
    assert result["constraint_satisfied"] is True


def test_learned_router_rejects_profile_order_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "router.pt"
    save_router_checkpoint(RouterNetwork(model_count=3), path, PROFILES, train_sample_ids=[])
    with pytest.raises(ValueError, match="model order"):
        LearnedRouter(path, list(reversed(PROFILES)))
