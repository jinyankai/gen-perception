#!/usr/bin/env python3
"""Depth Anything V2 depth baseline (optional second depth reference).

Reads local Depth-Anything-V2 weights via the transformers pipeline and writes
HxW float32 depth ``.npy`` predictions.

IMPORTANT: the relative model outputs inverse depth (disparity-like: larger =
closer). The evaluator's per-sample affine alignment fits a linear scale+shift
in depth space and cannot undo the reciprocal relation, so pass ``--invert`` to
turn the disparity into a depth-proportional map before writing. Without it the
reported AbsRel/delta1 are meaningless. Values below --invert-epsilon are floored
to avoid division blow-ups on zero-disparity pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    InferenceTimer,
    iter_manifest,
    read_image_manifest,
    write_prediction,
    write_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Convert disparity output to a depth-proportional map (required for NYUv2 metrics).",
    )
    parser.add_argument("--invert-epsilon", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        entries = read_image_manifest(args.image_manifest, args.limit)
        weights = args.weights_dir.expanduser().resolve()
        processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(
            weights, local_files_only=True
        ).to(args.device)
        model.eval()

        timer = InferenceTimer()
        image_size = (0, 0)
        for entry in iter_manifest(entries):
            with Image.open(entry.image_path) as handle:
                image = handle.convert("RGB")
            native_hw = (image.height, image.width)
            inputs = processor(images=image, return_tensors="pt").to(args.device)
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs)
            timer.record(time.perf_counter() - start)
            post = processor.post_process_depth_estimation(
                outputs, target_sizes=[native_hw]
            )[0]
            depth = np.asarray(post["predicted_depth"].cpu(), dtype=np.float32)
            depth = np.squeeze(depth)
            if depth.shape != native_hw:
                raise ValueError(
                    f"depth shape {depth.shape} != image {native_hw} for {entry.sample_id}"
                )
            if args.invert:
                depth = 1.0 / np.maximum(depth, np.float32(args.invert_epsilon))
            image_size = depth.shape
            write_prediction(args.output_dir, entry.sample_id, depth)

        write_runtime(
            args.output_dir,
            model="depth-anything-v2-base",
            weights_dir=args.weights_dir,
            device=args.device,
            precision=args.precision,
            image_size=image_size,
            timer=timer,
        )
    except (OSError, RuntimeError, TypeError, ValueError, ImportError, KeyError) as exc:
        print(f"BASELINE_FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "BASELINE_COMPLETED",
                "model": "depth-anything-v2-base",
                "sample_count": len(entries),
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
