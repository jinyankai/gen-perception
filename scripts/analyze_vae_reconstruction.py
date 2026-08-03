#!/usr/bin/env python3
"""Reconstruct real task targets through the frozen SD2 VAE and report fidelity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.analysis import (  # noqa: E402
    aggregate_reconstruction_records,
    reconstruction_metrics,
)
from perception_diffusion.data import build_dataloader  # noqa: E402
from perception_diffusion.models import load_pretrained_system  # noqa: E402
from perception_diffusion.task_specs import build_task_specs  # noqa: E402
from perception_diffusion.tasks import TASK_NAMES  # noqa: E402
from perception_diffusion.visualization import save_reconstruction_panel  # noqa: E402
from perception_diffusion.utils.config import configured_task_names, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", choices=(*TASK_NAMES, "all"), default="all")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_id(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.limit <= 0 or args.num_workers < 0:
        raise ValueError("limit must be positive and num-workers non-negative")
    output_dir = args.output_dir.expanduser()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"analysis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(exist_ok=True)
    config = load_config(args.config)
    configured = configured_task_names(config)
    tasks = configured if args.task == "all" else (args.task,)
    missing = sorted(set(tasks) - set(configured))
    if missing:
        raise ValueError(f"tasks are not selected by the config: {missing}")
    specs = build_task_specs(config)
    system = load_pretrained_system(
        config,
        device_override=args.device,
        precision_override=args.precision,
        for_training=False,
    )
    records: list[dict[str, object]] = []
    for task_name in tasks:
        task_data = (
            config["data"]["datasets"][task_name]
            if config["task"]["name"] == "multitask"
            else config["data"]
        )
        split = args.split or str(task_data["split"])
        loader = build_dataloader(
            config,
            task_name,
            split=split,
            training=False,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            strict_protocol=True,
            max_samples=args.limit,
        )
        spec = specs[task_name]
        for raw_batch in loader:
            target = raw_batch["target"].to(system.device)
            latent, adapted = system.visual_pathway.encode_targets(target, task_name)
            reconstructed = system.visual_pathway.decode_latents(latent)
            valid = raw_batch["valid_mask"][0, 0].cpu().numpy().astype(bool)
            native = raw_batch["native_target"][0].cpu().numpy()
            metrics = reconstruction_metrics(
                task_name,
                encoded_target=adapted[0].detach().cpu().numpy(),
                reconstructed_pixels=reconstructed[0].cpu().numpy(),
                native_target=native,
                valid_mask=valid,
                codec=spec.codec,
                query_class_id=int(raw_batch["query_class_id"][0]),
            )
            sample_id = str(raw_batch["sample_id"][0])
            records.append({"task": task_name, "sample_id": sample_id, "metrics": metrics})
            save_reconstruction_panel(
                adapted[0].detach().cpu(),
                reconstructed[0].cpu(),
                visual_dir / f"{task_name}__{_safe_id(sample_id)}.png",
            )

    summary = aggregate_reconstruction_records(records)
    summary["model_path"] = str(system.model_path)
    summary["model_revision"] = system.model_revision
    (output_dir / "records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "sample_id", "metric", "value"])
        for record in records:
            for name, value in record["metrics"].items():  # type: ignore[union-attr]
                writer.writerow([record["task"], record["sample_id"], name, value])
    ranking = summary["common_encoded_mse_ranking_high_to_low"]
    markdown = [
        "# VAE Target Reconstruction Fidelity",
        "",
        "Status: diagnostic run; not a benchmark result.",
        "",
        f"Common encoded-space MSE ranking (high to low): {', '.join(ranking)}.",
        "",
        str(summary["qualification"]),
        "",
        "See `report.json`, `report.csv`, `records.json`, and `visualizations/`.",
    ]
    (output_dir / "fidelity.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"VAE_ANALYSIS_FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
