"""Eight research figures, derived exclusively from saved experiment observations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NAMES = {"mobilenet_v3_small": "MobileNetV3", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
         "rule": "Threshold router", "learned": "AdaptiveServe"}


def generate_plots(results, output):
    results, output = Path(results), Path(output)
    observations = pd.read_csv(results / "observations.csv")
    report = json.loads((results / "summary.json").read_text())
    if observations.empty or not report["summary"]:
        raise ValueError("No measurements to plot")
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(report["summary"])
    labels = [NAMES.get(value, value) for value in summary.policy]
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    colors = ["#0f766e" if value == "learned" else "#64748b" for value in summary.policy]
    paths = []
    footer = f"{report['metadata'].get('test_images', '?')} held-out images · {report['metadata'].get('backend', '?')} backend · actual observations"

    def save(fig, name):
        fig.text(.02, .015, footer, fontsize=8, color="#64748b")
        fig.tight_layout(rect=(0, .05, 1, 1))
        destination = output / f"{name}.png"
        fig.savefig(destination, dpi=170, facecolor="white")
        plt.close(fig)
        paths.append(destination)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (_, row), color, label in zip(summary.iterrows(), colors, labels):
        ax.scatter(row.latency_ms, row.accuracy * 100, s=100, color=color)
        offset = (6, -18) if row.policy == "learned" else (6, 8)
        ax.annotate(label, (row.latency_ms, row.accuracy * 100), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean request latency (ms)", ylabel="Top-1 accuracy (%)", title="Accuracy–latency tradeoff")
    ax.margins(.2)
    save(fig, "accuracy_vs_latency")

    adaptive = observations[observations.policy.isin(["rule", "learned"])]
    selection = pd.crosstab(adaptive.policy, adaptive.selected_model)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not selection.empty:
        selection.rename(index=NAMES, columns=NAMES).plot.bar(stacked=True, ax=ax, rot=0)
    ax.set(title="Model selection distribution", xlabel="Policy", ylabel="Requests")
    save(fig, "model_selection_distribution")

    for name, column, title, ylabel in [
        ("average_inference_latency", "latency_ms", "Average request latency (including analyzer)", "Milliseconds"),
        ("memory_consumption", "memory_mb", "Resident process memory", "MiB (whole serving process)"),
        ("cpu_usage", "cpu_percent", "Average request CPU usage", "Process CPU (%) · one core = 100%"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, summary[column], color=colors)
        ax.tick_params(axis="x", labelrotation=15)
        ax.set(title=title, ylabel=ylabel)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(8, 5))
    improvements = report.get("improvements", [])
    ax.bar([NAMES.get(row["baseline"], row["baseline"]) for row in improvements],
           [row["latency_improvement_pct"] for row in improvements], color="#0f766e")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set(title="AdaptiveServe latency change vs static models", ylabel="Latency reduction (%) · negative means slower")
    save(fig, "adaptiveserve_improvement")

    learned = observations[observations.policy == "learned"]
    fig, ax = plt.subplots(figsize=(9, 7))
    if not learned.empty:
        # Keep full-head mistakes visible while avoiding an unreadable 1000² chart.
        classes = sorted(learned.label.unique().tolist())
        positions = {value: index for index, value in enumerate(classes)}
        matrix = np.zeros((len(classes), len(classes) + 1), dtype=int)
        for row in learned.itertuples():
            matrix[positions[row.label], positions.get(row.class_index, len(classes))] += 1
        sns.heatmap(matrix, annot=True, fmt="d", cmap="GnBu", ax=ax,
                    xticklabels=[str(i) for i in classes] + ["Other"], yticklabels=[str(i) for i in classes])
    ax.set(title="AdaptiveServe confusion matrix (ImageNet indices)", xlabel="Predicted class · Other = outside subset", ylabel="True class")
    save(fig, "confusion_matrix")

    fig, ax = plt.subplots(figsize=(8, 5))
    scores = learned.complexity.dropna() if "complexity" in learned else pd.Series(dtype=float)
    ax.hist(scores, bins=np.linspace(0, 1, 21), color="#0f766e", edgecolor="white")
    ax.set(title="Predicted input complexity", xlabel="Learned difficulty score", ylabel="Requests", xlim=(0, 1))
    save(fig, "complexity_distribution")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/latest"))
    parser.add_argument("--output", type=Path, default=Path("frontend/public/results"))
    args = parser.parse_args()
    generate_plots(args.results, args.output)
