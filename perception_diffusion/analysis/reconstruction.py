"""Task-native and common-space VAE reconstruction fidelity metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _valid_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid mask must have shape {shape}, got {valid.shape}")
    if not np.any(valid):
        raise ValueError("valid mask contains no valid pixel")
    return valid


def reconstruction_metrics(
    task_name: str,
    *,
    encoded_target: np.ndarray,
    reconstructed_pixels: np.ndarray,
    native_target: np.ndarray,
    valid_mask: np.ndarray,
    codec: Any,
    query_class_id: int = -1,
) -> dict[str, float]:
    encoded = np.asarray(encoded_target, dtype=np.float32)
    reconstructed = np.asarray(reconstructed_pixels, dtype=np.float32)
    if encoded.shape != reconstructed.shape or encoded.ndim != 3 or encoded.shape[0] != 3:
        raise ValueError("encoded and reconstructed targets must share shape [3,H,W]")
    valid = _valid_mask(valid_mask, tuple(encoded.shape[1:]))
    delta = reconstructed[:, valid] - encoded[:, valid]
    mse = float(np.mean(delta**2))
    mae = float(np.mean(np.abs(delta)))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(2.0 / math.sqrt(mse))
    metrics = {"encoded_mse": mse, "encoded_mae": mae, "encoded_psnr": psnr}

    decoded = codec.decode(reconstructed, valid)
    if task_name == "segmentation":
        native = np.asarray(native_target)
        if native.ndim == 3 and native.shape[0] == 1:
            native = native[0]
        if query_class_id < 0:
            target = codec.decode(encoded, valid)
        else:
            target = (native == query_class_id).astype(np.uint8)
        prediction = np.asarray(decoded, dtype=np.uint8)
        metrics["foreground_fraction"] = float(target[valid].mean())
        intersection = np.logical_and(prediction == 1, target == 1) & valid
        union = np.logical_or(prediction == 1, target == 1) & valid
        metrics["binary_iou"] = (
            1.0 if not np.any(union) else float(intersection.sum() / union.sum())
        )
        metrics["pixel_accuracy"] = float((prediction[valid] == target[valid]).mean())
    elif task_name == "depth":
        native = np.asarray(native_target, dtype=np.float32)
        if native.ndim == 3 and native.shape[0] == 1:
            native = native[0]
        predicted = np.asarray(decoded, dtype=np.float32)
        depth_valid = valid & np.isfinite(native) & (native > 0) & np.isfinite(predicted)
        if not np.any(depth_valid):
            raise ValueError("depth reconstruction contains no jointly valid pixel")
        metrics["abs_rel"] = float(
            np.mean(np.abs(predicted[depth_valid] - native[depth_valid]) / native[depth_valid])
        )
        metrics["rmse"] = float(
            np.sqrt(np.mean((predicted[depth_valid] - native[depth_valid]) ** 2))
        )
    elif task_name == "normal":
        native = np.asarray(native_target, dtype=np.float32)
        predicted = np.asarray(decoded, dtype=np.float32)
        if native.shape != predicted.shape or native.shape[0] != 3:
            raise ValueError("normal targets must have shape [3,H,W]")
        native_norm = np.linalg.norm(native, axis=0)
        predicted_norm = np.linalg.norm(predicted, axis=0)
        normal_valid = valid & (native_norm > 1.0e-6) & (predicted_norm > 1.0e-6)
        if not np.any(normal_valid):
            raise ValueError("normal reconstruction contains no jointly valid pixel")
        cosine = np.sum(native * predicted, axis=0) / np.maximum(
            native_norm * predicted_norm, 1.0e-6
        )
        angles = np.degrees(np.arccos(np.clip(cosine[normal_valid], -1.0, 1.0)))
        metrics["mean_angular_error"] = float(np.mean(angles))
        metrics["median_angular_error"] = float(np.median(angles))
    else:
        raise ValueError(f"unsupported reconstruction task: {task_name}")
    return metrics


def aggregate_reconstruction_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one reconstruction record is required")
    tasks = sorted({str(record["task"]) for record in records})
    per_task: dict[str, Any] = {}
    for task in tasks:
        selected = [record for record in records if record["task"] == task]
        metric_names = sorted(
            set.intersection(*(set(record["metrics"]) for record in selected))
        )
        per_task[task] = {
            "sample_count": len(selected),
            "mean": {
                name: float(np.mean([record["metrics"][name] for record in selected]))
                for name in metric_names
            },
        }
    ranking = sorted(
        tasks, key=lambda task: per_task[task]["mean"]["encoded_mse"], reverse=True
    )
    return {
        "status": "VAE_RECONSTRUCTION_ANALYSIS_COMPLETED",
        "formal_experiment": False,
        "per_task": per_task,
        "common_encoded_mse_ranking_high_to_low": ranking,
        "qualification": (
            "The ranking uses common codec-image MSE only; task-native metrics are not "
            "directly comparable across segmentation, depth, and normals."
        ),
    }
