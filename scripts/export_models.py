#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from adaptiveserve.export import export_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ImageNet teachers to ONNX and validate parity.")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--registry", default="configs/models.json")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    reports = export_models(args.output_dir, args.registry, args.labels, validate=not args.skip_validation)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
