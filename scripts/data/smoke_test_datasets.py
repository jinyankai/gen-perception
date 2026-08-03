#!/usr/bin/env python3
"""Load real ADE20K/NYUv2 batches and report schema, range, and geometry checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_diffusion.data import build_dataloader  # noqa: E402
from perception_diffusion.utils.config import load_config  # noqa: E402


CONFIGS = {
    "segmentation": ROOT / "configs" / "segmentation" / "ade20k.yaml",
    "depth": ROOT / "configs" / "depth" / "nyuv2.yaml",
    "normal": ROOT / "configs" / "normal" / "nyuv2.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "segmentation", "depth", "normal"),
        default="all",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data")),
    )
    parser.add_argument("--split", default=None, help="Override configured split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--nyuv2-source", choices=("auto", "raw", "processed"), default="auto"
    )
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor.float()).all()),
    }
    if tensor.numel():
        result["range"] = [float(tensor.min()), float(tensor.max())]
    return result


def run_task(args: argparse.Namespace, task_name: str) -> dict[str, Any]:
    config = load_config(CONFIGS[task_name])
    if task_name in {"depth", "normal"}:
        config["data"]["source"] = args.nyuv2_source
    split = args.split or str(config["data"]["split"])
    loader = build_dataloader(
        config,
        task_name,
        split=split,
        training=split.casefold() in {"train", "training"},
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        strict_protocol=True,
    )
    batch = next(iter(loader))
    if batch["task_name"] != task_name:
        raise ValueError(f"batch task mismatch: {batch['task_name']} != {task_name}")
    valid = batch["valid_mask"]
    result: dict[str, Any] = {
        "status": "PASSED",
        "dataset": type(loader.dataset).__name__,
        "source": getattr(loader.dataset, "source", "filesystem"),
        "split": split,
        "dataset_size": len(loader.dataset),
        "sample_ids": list(batch["sample_id"]),
        "task_name": batch["task_name"],
        "image": tensor_summary(batch["image"]),
        "target": tensor_summary(batch["target"]),
        "native_target": tensor_summary(batch["native_target"]),
        "valid_mask": {
            **tensor_summary(valid),
            "valid_fraction": float(valid.float().mean()),
        },
        "query_class_ids": batch["query_class_id"].tolist(),
        "text_condition": list(batch["text_condition"]),
    }
    if task_name == "segmentation":
        labels = batch["native_target"]
        legal = (labels == 255) | ((labels >= 0) & (labels < 150))
        if not bool(legal.all()):
            raise ValueError("ADE20K batch contains illegal remapped label IDs")
    elif task_name == "depth":
        depth = batch["native_target"][:, 0]
        valid_depth = depth[valid[:, 0]]
        if valid_depth.numel() == 0 or not bool(
            ((valid_depth >= 0.1) & (valid_depth <= 10.0)).all()
        ):
            raise ValueError("NYUv2 valid depth lies outside configured metric range")
    else:
        normals = batch["native_target"]
        norms = torch.linalg.vector_norm(normals, dim=1)
        valid_norms = norms[valid[:, 0]]
        if valid_norms.numel() == 0 or not torch.allclose(
            valid_norms, torch.ones_like(valid_norms), atol=2.0e-4, rtol=0.0
        ):
            raise ValueError("NYUv2 valid normals are not unit length")
        result["normal_unit_error_max"] = float((valid_norms - 1.0).abs().max())
    close = getattr(loader.dataset, "close", None)
    if callable(close):
        close()
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    data_root = args.data_root.expanduser().resolve()
    os.environ["DATA_ROOT"] = str(data_root)
    os.environ.setdefault("MODEL_CACHE", str(data_root.parent / "models"))
    os.environ.setdefault("OUTPUT_ROOT", str(data_root.parent / "outputs"))
    tasks = tuple(CONFIGS) if args.task == "all" else (args.task,)
    results: dict[str, Any] = {
        "status": "REAL_DATASET_SMOKE_PASSED",
        "formal_experiment": False,
        "data_root": str(data_root),
        "tasks": {},
    }
    for task_name in tasks:
        results["tasks"][task_name] = run_task(args, task_name)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

