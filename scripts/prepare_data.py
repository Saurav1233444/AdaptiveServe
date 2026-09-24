#!/usr/bin/env python3
from __future__ import annotations

import argparse
from adaptiveserve.data import prepare_imagenette


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Imagenette-160 and write a deterministic manifest.")
    parser.add_argument("--destination", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--calibration-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(prepare_imagenette(args.destination, args.manifest, train_count=args.train_count,
          calibration_count=args.calibration_count, test_count=args.test_count, seed=args.seed))


if __name__ == "__main__":
    main()
