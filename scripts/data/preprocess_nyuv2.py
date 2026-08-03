#!/usr/bin/env python3
"""Materialize official NYUv2 RGB/depth/derived-normal samples as compressed NPZ."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_utils import write_json  # noqa: E402
from perception_diffusion.data.nyuv2 import (  # noqa: E402
    NYUV2_EXPECTED_COUNTS,
    NYUv2RawReader,
)
from perception_diffusion.data.nyuv2_geometry import (  # noqa: E402
    NYUV2_RGB_INTRINSICS,
    depth_to_normals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data")) / "nyuv2",
        help="NYUv2 root containing nyu_depth_v2_labeled.mat and splits.mat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to ROOT/processed.",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--depth-field", choices=("depths", "rawDepths"), default="depths")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--max-relative-depth-jump", type=float, default=0.05)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing per-sample NPZ files.",
    )
    return parser.parse_args()


def validate_existing(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as sample:
            required = {"image", "depth", "depth_valid", "normal", "normal_valid"}
            if not required.issubset(sample.files):
                return False
            return (
                sample["image"].shape == (480, 640, 3)
                and sample["image"].dtype == np.uint8
                and sample["depth"].shape == (480, 640)
                and sample["normal"].shape == (3, 480, 640)
            )
    except (OSError, ValueError, KeyError):
        return False


def write_sample(
    path: Path,
    *,
    image: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    normal: np.ndarray,
    normal_valid: np.ndarray,
    source_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        image=np.asarray(image, dtype=np.uint8),
        depth=np.asarray(depth, dtype=np.float32),
        depth_valid=np.asarray(depth_valid, dtype=np.bool_),
        normal=np.asarray(normal, dtype=np.float32),
        normal_valid=np.asarray(normal_valid, dtype=np.bool_),
        source_index=np.asarray(source_index, dtype=np.int64),
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if not 0 < args.min_depth < args.max_depth:
        raise ValueError("depth bounds must satisfy 0 < min_depth < max_depth")
    root = args.root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "processed"
    )
    reader = NYUv2RawReader(root, depth_field=args.depth_field, strict_protocol=True)
    requested_splits = ("train", "test") if args.split == "all" else (args.split,)
    started = time.time()
    processed_splits: dict[str, list[int]] = {}
    try:
        for split in requested_splits:
            indices = reader.splits[split]
            if indices.size != NYUV2_EXPECTED_COUNTS[split]:
                raise ValueError(f"unexpected {split} split size: {indices.size}")
            for ordinal, source_index_raw in enumerate(indices, start=1):
                source_index = int(source_index_raw)
                destination = output / split / f"{source_index + 1:04d}.npz"
                if destination.is_file() and not args.overwrite:
                    if validate_existing(destination):
                        continue
                    raise ValueError(
                        f"existing processed sample is invalid: {destination}; "
                        "move it aside or rerun with --overwrite"
                    )
                image, depth = reader.read(source_index)
                depth_valid = (
                    np.isfinite(depth)
                    & (depth >= args.min_depth)
                    & (depth <= args.max_depth)
                )
                normal, normal_valid = depth_to_normals(
                    depth,
                    depth_valid,
                    intrinsics=NYUV2_RGB_INTRINSICS,
                    max_relative_depth_jump=args.max_relative_depth_jump,
                )
                write_sample(
                    destination,
                    image=image,
                    depth=depth,
                    depth_valid=depth_valid,
                    normal=normal,
                    normal_valid=normal_valid,
                    source_index=source_index,
                )
                if ordinal == 1 or ordinal % 50 == 0 or ordinal == indices.size:
                    print(f"{split}: {ordinal}/{indices.size} -> {destination}")
            processed_splits[split] = [int(value) for value in indices]
    finally:
        reader.close()

    manifest_path = output / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    existing_splits = manifest.get("splits", {})
    if not isinstance(existing_splits, dict):
        existing_splits = {}
    existing_splits.update(processed_splits)
    write_json(
        manifest_path,
        {
            "format": "gen-perception-nyuv2-v1",
            "source_file": str(reader.mat_path),
            "split_file": str(reader.split_path),
            "depth_field": args.depth_field,
            "depth_unit": "meters",
            "depth_range": [args.min_depth, args.max_depth],
            "camera_intrinsics": NYUV2_RGB_INTRINSICS.to_dict(),
            "normal_convention": {
                "coordinate_system": "camera_x_right_y_down_z_forward",
                "orientation": "toward_camera",
                "channel_order": "xyz",
                "max_relative_depth_jump": args.max_relative_depth_jump,
                "invalid_pixels": "zero_vector_with_normal_valid_false",
            },
            "splits": existing_splits,
            "elapsed_seconds_last_run": round(time.time() - started, 3),
        },
    )
    print(f"NYUv2 preprocessing complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

