#!/usr/bin/env python3
"""Train the real shared SD2 perception model from one versioned config."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.training import run_training  # noqa: E402
from perception_diffusion.utils.config import (  # noqa: E402
    configured_task_names,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--tensorboard", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _dry_run_summary(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    task_names = configured_task_names(config)
    model_config = config["model"]  # type: ignore[index]
    conditioning = model_config["conditioning"]  # type: ignore[index]
    noise = model_config.get("noise", {}).get("multi_scale", {})  # type: ignore[union-attr]
    return {
        "status": "CONFIG_VALIDATED",
        "config": str(args.config),
        "task": config["task"]["name"],  # type: ignore[index]
        "tasks": list(task_names),
        "experiment": config["experiment"]["name"],  # type: ignore[index]
        "architecture": "task-token-cross-attention-shared-unet",
        "backbone": model_config["backbone"]["family"],
        "expected_model_revision": model_config["backbone"]["expected_revision"],
        "trainable_scope": model_config["shared_unet"]["trainable_scope"],
        "condition_adapter_placement": model_config["condition_adapter"]["placement"],
        "target_adapter_enabled": model_config["target_adapter"]["enabled"],
        "task_condition_enabled": conditioning.get("use_task_condition", True),
        "text_condition_enabled": conditioning.get("use_text_condition", True),
        "annealed_multi_scale_noise": bool(
            noise.get("enabled", False) and noise.get("annealed", False)
        ),
        "resume": None if args.resume is None else str(args.resume),
        "formal_training_started": False,
    }


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.output_root is not None:
            config["experiment"]["output_root"] = str(args.output_root)
        if args.dry_run:
            print(json.dumps(_dry_run_summary(config, args), indent=2, ensure_ascii=False))
            return 0
        summary = run_training(
            config,
            repo_root=ROOT,
            command=sys.argv,
            device_override=args.device,
            precision_override=args.precision,
            max_steps_override=args.max_steps,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            experiment_name_override=args.experiment_name,
            resume=args.resume,
            enable_tensorboard=args.tensorboard,
            enable_wandb=args.wandb,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"TRAINING_FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    # Under torchrun every rank runs this script; only rank 0 owns the run
    # directory and its summary, so only rank 0 prints. RANK is unset for the
    # single-process path, which is therefore treated as rank 0.
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
