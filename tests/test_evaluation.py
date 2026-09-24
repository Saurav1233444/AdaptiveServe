import pandas as pd
import pytest

from experiments.evaluate import summarize, validate_evaluation_ids


def test_summary_uses_measurements_and_correct_units():
    rows = pd.DataFrame([
        {"policy": "rule", "correct": 1, "latency_ms": 10, "memory_mb": 100,
         "cpu_percent": 40, "latency_budget_ms": 15},
        {"policy": "rule", "correct": 0, "latency_ms": 30, "memory_mb": 110,
         "cpu_percent": 60, "latency_budget_ms": 15},
    ])
    row = summarize(rows)[0]
    assert row["accuracy"] == .5
    assert row["latency_ms"] == 20
    assert row["throughput_rps"] == 50
    assert row["sla_violation_rate"] == .5
    assert row["memory_mb"] == 105


def test_rejects_train_test_leakage():
    with pytest.raises(ValueError, match="overlap"):
        validate_evaluation_ids(["a", "b"], ["b"])
    validate_evaluation_ids(["a"], ["b"])


def test_empty_experiment_cannot_produce_metrics():
    with pytest.raises(ValueError, match="empty"):
        summarize(pd.DataFrame())
