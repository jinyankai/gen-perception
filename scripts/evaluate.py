#!/usr/bin/env python3
"""Evaluate paired prediction/ground-truth arrays for one configured task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.evaluation.runner import (  # noqa: E402
    EvaluationInputs,
    evaluate_array_roots,
    write_evaluation_outputs,
)
from perception_diffusion.task_specs import build_task_specs  # noqa: E402
from perception_diffusion.tasks import TASK_NAMES  # noqa: E402
from perception_diffusion.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--task",
        choices=TASK_NAMES,
        help="Required for a multitask config; inferred for a single-task config.",
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--valid-masks", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-key", help="Array key for multi-array prediction NPZ files.")
    parser.add_argument("--target-key", help="Array key for multi-array target NPZ files.")
    parser.add_argument("--valid-mask-key", help="Array key for multi-array mask NPZ files.")
    parser.add_argument("--prediction-depth-scale", type=float, default=1.0)
    parser.add_argument("--target-depth-scale", type=float, default=1.0)
    parser.add_argument(
        "--prediction-normal-encoding",
        choices=("unit", "zero_one", "uint8"),
        default="unit",
    )
    parser.add_argument(
        "--target-normal-encoding",
        choices=("unit", "zero_one", "uint8"),
        default="unit",
    )
    parser.add_argument("--prediction-label-offset", type=int, default=0)
    parser.add_argument("--target-label-offset", type=int, default=0)
    parser.add_argument("--prediction-ignore-value", type=int)
    parser.add_argument("--target-ignore-value", type=int)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N sorted pairs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        specs = build_task_specs(config)
        task = args.task
        if task is None:
            if len(specs) != 1:
                raise ValueError("--task is required when --config selects multiple tasks")
            task = next(iter(specs))
        if task not in specs:
            raise ValueError(f"task {task!r} is not selected by {config_path}")

        inputs = EvaluationInputs(
            prediction_root=args.predictions.expanduser().resolve(),
            target_root=args.targets.expanduser().resolve(),
            valid_mask_root=(
                None if args.valid_masks is None else args.valid_masks.expanduser().resolve()
            ),
            prediction_key=args.prediction_key,
            target_key=args.target_key,
            valid_mask_key=args.valid_mask_key,
            prediction_depth_scale=args.prediction_depth_scale,
            target_depth_scale=args.target_depth_scale,
            prediction_normal_encoding=args.prediction_normal_encoding,
            target_normal_encoding=args.target_normal_encoding,
            prediction_label_offset=args.prediction_label_offset,
            target_label_offset=args.target_label_offset,
            prediction_ignore_value=args.prediction_ignore_value,
            target_ignore_value=args.target_ignore_value,
            limit=args.limit,
        )
        result = evaluate_array_roots(
            specs[task], inputs, config_path=config_path, resolved_config=config
        )
        json_path, csv_path = write_evaluation_outputs(
            result, args.output_dir.expanduser().resolve(), overwrite=args.overwrite
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"EVALUATION_FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": result["status"],
                "task": result["task"],
                "sample_count": result["input"]["sample_count"],
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
