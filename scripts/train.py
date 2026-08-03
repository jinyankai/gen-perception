#!/usr/bin/env python3
"""Unified training entry point.

The current milestone supports configuration validation only. Model training is
intentionally rejected until the model components and gate tests are committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        summary = {
            "status": "CONFIG_VALIDATED",
            "config": str(args.config),
            "task": config["task"]["name"],
            "experiment": config["experiment"]["name"],
            "resume": None if args.resume is None else str(args.resume),
            "formal_training_started": False,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    print(
        "Training is not implemented at this milestone. Run with --dry-run; "
        "do not record a formal experiment.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
