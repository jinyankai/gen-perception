#!/usr/bin/env python3
"""DSINE surface-normal baseline (optional second normal reference).

DSINE has no diffusers/transformers wrapper. It is loaded from its official
repository checkout via torch.hub with ``source="local"``. The exact hub entry
name and preprocessing must be confirmed against the pinned DSINE commit noted
in RUNBOOK.md before formal runs; the wrapper below follows the public
``torch.hub.load(repo, "DSINE", ...)`` interface and NYUv2 intrinsics.

Writes 3xHxW float32 unit-normal ``.npy`` predictions
(evaluate.py --prediction-normal-encoding unit).
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
    parser.add_argument(
        "--dsine-repo",
        required=True,
        type=Path,
        help="Local checkout of the pinned DSINE repository (torch.hub source).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch

        entries = read_image_manifest(args.image_manifest, args.limit)
        model = torch.hub.load(
            str(args.dsine_repo.expanduser().resolve()),
            "DSINE",
            source="local",
            local_file_path=str(args.weights_dir.expanduser().resolve()),
            trust_repo=True,
        )
        model = model.to(args.device).eval()

        timer = InferenceTimer()
        image_size = (0, 0)
        for entry in iter_manifest(entries):
            with Image.open(entry.image_path) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            tensor = (
                torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)[None].to(args.device)
            )
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model(tensor)
            timer.record(time.perf_counter() - start)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[-1]
            normals = np.squeeze(np.asarray(prediction.detach().float().cpu(), dtype=np.float32))
            if normals.ndim != 3 or 3 not in (normals.shape[0], normals.shape[-1]):
                raise ValueError(f"unexpected DSINE normal shape {normals.shape} for {entry.sample_id}")
            if normals.shape[-1] == 3:
                normals = np.moveaxis(normals, -1, 0)
            image_size = normals.shape[1:]
            write_prediction(args.output_dir, entry.sample_id, normals)

        write_runtime(
            args.output_dir,
            model="dsine",
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
                "model": "dsine",
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
