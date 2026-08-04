#!/usr/bin/env python3
"""Marigold surface-normal baseline.

Reads local Marigold normals weights, runs MarigoldNormalsPipeline on each
manifest image, and writes 3xHxW float32 unit-normal ``.npy`` predictions
(evaluate.py --prediction-normal-encoding unit). The evaluator re-normalizes,
so only the vector direction must be correct.
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
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from diffusers import MarigoldNormalsPipeline

        entries = read_image_manifest(args.image_manifest, args.limit)
        dtype = torch.float16 if args.precision == "fp16" and args.device == "cuda" else torch.float32
        pipe = MarigoldNormalsPipeline.from_pretrained(
            args.weights_dir.expanduser().resolve(),
            torch_dtype=dtype,
            local_files_only=True,
        ).to(args.device)
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

        timer = InferenceTimer()
        image_size = (0, 0)
        for entry in iter_manifest(entries):
            with Image.open(entry.image_path) as handle:
                image = handle.convert("RGB")
            start = time.perf_counter()
            with torch.inference_mode():
                result = pipe(
                    image,
                    num_inference_steps=args.num_steps,
                    ensemble_size=args.ensemble_size,
                    generator=generator,
                    output_type="np",
                )
            timer.record(time.perf_counter() - start)
            normals = np.squeeze(np.asarray(result.prediction, dtype=np.float32))
            if normals.ndim != 3 or 3 not in (normals.shape[0], normals.shape[-1]):
                raise ValueError(f"unexpected Marigold normal shape {normals.shape} for {entry.sample_id}")
            if normals.shape[-1] == 3:
                normals = np.moveaxis(normals, -1, 0)
            image_size = normals.shape[1:]
            write_prediction(args.output_dir, entry.sample_id, normals)

        write_runtime(
            args.output_dir,
            model="marigold-normals",
            weights_dir=args.weights_dir,
            device=args.device,
            precision=args.precision,
            image_size=image_size,
            timer=timer,
        )
    except (OSError, RuntimeError, TypeError, ValueError, ImportError) as exc:
        print(f"BASELINE_FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "BASELINE_COMPLETED",
                "model": "marigold-normals",
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
