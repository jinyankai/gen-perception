"""Shared helpers for discriminative baseline runners.

Each baseline runner reads an image manifest produced by ``export_targets.py``,
writes one prediction array per sample under the same relative ``sample_id`` the
shared evaluator (``scripts/evaluate.py``) uses to pair predictions with targets,
and records a ``runtime.json`` provenance file. No model download happens here:
weights are resolved once by ``scripts/hf_download.py`` and read locally.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ManifestEntry:
    """One image to run inference on, keyed by its evaluator sample_id."""

    sample_id: str
    image_path: Path


def read_image_manifest(manifest_path: Path, limit: int | None = None) -> list[ManifestEntry]:
    """Load ``image_manifest.jsonl`` written by export_targets.py.

    Each line is ``{"sample_id": ..., "image_path": ...}``. Image paths are
    resolved relative to the manifest's directory when not absolute.
    """

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"image manifest not found: {manifest_path}")
    base = manifest_path.parent
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        record = json.loads(line)
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample_id!r} at line {line_number}")
        seen.add(sample_id)
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = base / image_path
        entries.append(ManifestEntry(sample_id=sample_id, image_path=image_path))
    if not entries:
        raise ValueError(f"image manifest is empty: {manifest_path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        entries = entries[:limit]
    return entries


def write_prediction(output_dir: Path, sample_id: str, array: np.ndarray) -> Path:
    """Write one prediction ``.npy`` at ``output_dir/<sample_id>.npy``.

    The sample_id may contain forward slashes; intermediate directories are
    created so the evaluator recovers the identical relative stem.
    """

    target = output_dir / f"{sample_id}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.ascontiguousarray(array))
    return target


class InferenceTimer:
    """Accumulate per-sample latencies for runtime.json percentiles."""

    def __init__(self) -> None:
        self._latencies_ms: list[float] = []
        self._wall_start = time.perf_counter()

    def record(self, seconds: float) -> None:
        self._latencies_ms.append(float(seconds) * 1000.0)

    def summary(self) -> dict[str, float | int]:
        latencies = np.asarray(self._latencies_ms, dtype=np.float64)
        return {
            "sample_count": int(latencies.size),
            "total_seconds": float(time.perf_counter() - self._wall_start),
            "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
        }


def write_runtime(
    output_dir: Path,
    *,
    model: str,
    weights_dir: Path,
    device: str,
    precision: str,
    image_size: tuple[int, int],
    timer: InferenceTimer,
) -> Path:
    """Record GPU/library provenance and latency percentiles next to predictions."""

    import torch  # imported lazily so export_targets.py need not depend on torch

    peak_bytes = 0
    gpu_name = None
    if device == "cuda" and torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated())
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    runtime = {
        "model": model,
        "weights_dir": str(weights_dir.resolve()),
        "device": device,
        "gpu_name": gpu_name,
        "gpu_count": int(torch.cuda.device_count()) if device == "cuda" else 0,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "precision": precision,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "peak_vram_bytes": peak_bytes,
        **timer.summary(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "runtime.json"
    runtime_path.write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return runtime_path


def iter_manifest(entries: list[ManifestEntry]) -> Iterator[ManifestEntry]:
    """Yield entries after checking each image exists (fail fast, clear message)."""

    for entry in entries:
        if not entry.image_path.is_file():
            raise FileNotFoundError(
                f"image for sample {entry.sample_id!r} is missing: {entry.image_path}"
            )
        yield entry
