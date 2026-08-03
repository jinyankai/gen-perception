#!/usr/bin/env python3
"""Run real end-to-end inference for one selected perception task."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.inference.runner import run_inference  # noqa: E402
from perception_diffusion.tasks import TASK_NAMES  # noqa: E402
from perception_diffusion.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=TASK_NAMES)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument(
        "--condition-mode",
        choices=("full", "task_only", "text_only", "unconditional"),
        default="full",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_inference(
            load_config(args.config),
            checkpoint=args.checkpoint,
            task_name=args.task,
            output_dir=args.output_dir,
            device_override=args.device,
            precision_override=args.precision,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_steps_override=args.num_steps,
            ensemble_size_override=args.ensemble_size,
            condition_mode=args.condition_mode,
            seed_override=args.seed,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"INFERENCE_FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
