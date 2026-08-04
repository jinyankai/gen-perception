#!/usr/bin/env python3
"""Export native-resolution ground truth and a shared image manifest.

This produces exactly the directory layout ``scripts/evaluate.py`` consumes:

    <output-dir>/
      targets/<sample_id>.npy        # seg: HxW int64 (0..149, 255=ignore)
                                     # depth: HxW float32 metric metres
                                     # normal: 3xHxW float32 unit vectors
      valid_masks/<sample_id>.npy    # depth/normal only: HxW bool
      images/<sample_id>.png         # nyuv2 only: decoded RGB (ade20k reuses source)
      image_manifest.jsonl           # {"sample_id", "image_path"} per line

Ground truth is written at native resolution so each baseline resizes its own
prediction to the target and the evaluator compares like-for-like. Segmentation
IDs are mapped to the evaluator's 0..149 convention here, so evaluate.py needs
no --target-label-offset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common import write_prediction  # noqa: E402
from perception_diffusion.data.nyuv2 import NYUv2RawReader, canonical_nyuv2_split  # noqa: E402
from perception_diffusion.data.nyuv2_geometry import (  # noqa: E402
    NYUV2_RGB_INTRINSICS,
    depth_to_normals,
)
from perception_diffusion.tasks import TASK_NAMES  # noqa: E402
from perception_diffusion.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_NAMES)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", help="Override the config validation split.")
    parser.add_argument("--limit", type=int, help="Export only the first N sorted samples.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_data_config(config: dict, task: str) -> dict:
    task_config = config["task"]
    if task_config.get("name") == "multitask":
        return config["data"]["datasets"][task]
    if task_config.get("name") != task:
        raise ValueError(f"config selects {task_config.get('name')!r}, not {task!r}")
    return config["data"]


def _guard_output(output_dir: Path, overwrite: bool) -> None:
    manifest = output_dir / "image_manifest.jsonl"
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest}; pass --overwrite")


def export_ade20k(data: dict, output_dir: Path, split: str, limit: int | None) -> list[dict]:
    """Write native seg targets (0..149/255) and reference the source JPEGs."""

    from perception_diffusion.data.ade20k import ADE20K_SPLITS

    root = Path(data["root"]).expanduser()
    resolved_split = ADE20K_SPLITS[split.casefold()]
    image_root = root / "images" / resolved_split
    annotation_root = root / "annotations" / resolved_split
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(f"missing images/{resolved_split} or annotations under {root}")
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    manifest: list[dict] = []
    for image_path in image_paths:
        relative = image_path.relative_to(image_root).with_suffix("").as_posix()
        sample_id = f"ade20k/{resolved_split}/{relative}"
        mask_path = annotation_root / image_path.relative_to(image_root).with_suffix(".png")
        with Image.open(mask_path) as handle:
            source = np.asarray(handle)
        if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
            raise ValueError(f"ADE20K mask must be integer HxW: {mask_path}")
        labels = np.full(source.shape, 255, dtype=np.int64)
        present = source > 0
        labels[present] = source[present].astype(np.int64) - 1
        write_prediction(output_dir / "targets", sample_id, labels)
        manifest.append({"sample_id": sample_id, "image_path": str(image_path.resolve())})
    return manifest


def export_nyuv2(data: dict, task: str, output_dir: Path, split: str, limit: int | None) -> list[dict]:
    """Decode native RGB/depth from the HDF5 asset and derive normals if needed."""

    root = Path(data["root"]).expanduser()
    reader = NYUv2RawReader(root, depth_field=str(data.get("depth_field", "depths")))
    resolved_split = canonical_nyuv2_split(split)
    indices = reader.splits[resolved_split]
    if limit is not None:
        indices = indices[:limit]
    image_dir = output_dir / "images"
    manifest: list[dict] = []
    for source_index in indices.tolist():
        sample_id = f"nyuv2/{resolved_split}/{source_index + 1:04d}"
        image, depth = reader.read(source_index)
        image_path = image_dir / f"{sample_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_path)
        depth_valid = np.isfinite(depth) & (depth > 0) & (depth <= 10.0)
        if task == "depth":
            write_prediction(output_dir / "targets", sample_id, depth.astype(np.float32))
            write_prediction(output_dir / "valid_masks", sample_id, depth_valid)
        else:
            normals, normal_valid = depth_to_normals(
                depth, depth_valid, intrinsics=NYUV2_RGB_INTRINSICS
            )
            write_prediction(output_dir / "targets", sample_id, normals.astype(np.float32))
            write_prediction(output_dir / "valid_masks", sample_id, normal_valid)
        manifest.append({"sample_id": sample_id, "image_path": str(image_path.resolve())})
    reader.close()
    return manifest


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config.expanduser().resolve())
        data = _resolve_data_config(config, args.task)
        split = args.split or str(data.get("validation_split", "validation"))
        output_dir = args.output_dir.expanduser().resolve()
        _guard_output(output_dir, args.overwrite)
        if args.task == "segmentation":
            manifest = export_ade20k(data, output_dir, split, args.limit)
        else:
            manifest = export_nyuv2(data, args.task, output_dir, split, args.limit)
        manifest_path = output_dir / "image_manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest),
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"EXPORT_FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "EXPORT_COMPLETED",
                "task": args.task,
                "split": split,
                "sample_count": len(manifest),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
