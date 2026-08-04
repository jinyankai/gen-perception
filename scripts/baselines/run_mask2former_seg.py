#!/usr/bin/env python3
"""Mask2Former ADE20K semantic-segmentation baseline.

Reads local Mask2Former weights (facebook/mask2former-swin-small-ade-semantic),
runs semantic inference per manifest image, and writes HxW int64 label maps in
the evaluator's 0..149 convention. Post-processing resizes logits back to each
image's native size so predictions pair pixel-for-pixel with exported targets.
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        entries = read_image_manifest(args.image_manifest, args.limit)
        weights = args.weights_dir.expanduser().resolve()
        processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
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
            segmentation = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[native_hw]
            )[0]
            labels = np.asarray(segmentation.cpu(), dtype=np.int64)
            if labels.shape != native_hw:
                raise ValueError(
                    f"segmentation shape {labels.shape} != image {native_hw} for {entry.sample_id}"
                )
            image_size = labels.shape
            write_prediction(args.output_dir, entry.sample_id, labels)

        write_runtime(
            args.output_dir,
            model="mask2former-swin-small-ade-semantic",
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
                "model": "mask2former-swin-small-ade-semantic",
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
