#!/usr/bin/env python3
from __future__ import annotations

import argparse
from adaptiveserve.training import train_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the difficulty analyzer and learned router.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--observations", default="results/profile/observations.csv")
    parser.add_argument("--profiles", default="models/profiles.json")
    parser.add_argument("--complexity-output", default="models/complexity_predictor.pt")
    parser.add_argument("--router-output", default="models/router.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--complexity-epochs", type=int)
    parser.add_argument("--router-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--contexts-per-sample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    complexity, router = train_artifacts(
        args.manifest, args.observations, args.profiles,
        complexity_output=args.complexity_output, router_output=args.router_output,
        epochs=args.epochs, complexity_epochs=args.complexity_epochs,
        router_epochs=args.router_epochs, batch_size=args.batch_size,
        contexts_per_sample=args.contexts_per_sample, seed=args.seed,
    )
    print(f"complexity_checkpoint={complexity}")
    print(f"router_checkpoint={router}")


if __name__ == "__main__":
    main()
