import json

import pandas as pd

from experiments.plot import generate_plots


def test_plots_require_real_observation_files(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        generate_plots(tmp_path, tmp_path / "figures")


def test_all_eight_figures_derive_from_fixture_observations(tmp_path):
    # Synthetic values are confined to unit tests, never shipped as experiment output.
    pd.DataFrame([{"policy": "learned", "selected_model": "tiny", "complexity": .4,
                   "label": 1, "class_index": 1}]).to_csv(tmp_path / "observations.csv", index=False)
    report = {"summary": [{"policy": "learned", "latency_ms": 1, "accuracy": 1, "memory_mb": 10, "cpu_percent": 80}],
              "improvements": [{"baseline": "tiny", "latency_improvement_pct": -1}],
              "metadata": {"test_images": 1, "backend": "fixture"}}
    (tmp_path / "summary.json").write_text(json.dumps(report))
    paths = generate_plots(tmp_path, tmp_path / "figures")
    assert len(paths) == 8
    assert "cpu_usage.png" in [path.name for path in paths]
    assert all(path.stat().st_size > 1000 for path in paths)
