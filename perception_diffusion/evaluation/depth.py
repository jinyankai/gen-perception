"""Monocular depth metrics with optional affine alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def align_scale_shift(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    valid_mask: NDArray[np.generic],
) -> tuple[NDArray[np.float32], float, float]:
    """Least-squares align prediction to target with scale and shift."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    count = int(mask.sum())
    if count < 2:
        raise ValueError("affine alignment requires at least two valid pixels")
    x = pred[mask]
    y = truth[mask]
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned = scale * pred + shift
    return aligned.astype(np.float32), float(scale), float(shift)


def _depth_metrics(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    valid_mask: NDArray[np.generic],
    epsilon: float,
) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    mask &= np.isfinite(pred) & np.isfinite(truth) & (truth > epsilon)
    count = int(mask.sum())
    if count == 0:
        raise ValueError("no valid depth pixels remain for evaluation")
    p = np.maximum(pred[mask], epsilon)
    t = np.maximum(truth[mask], epsilon)
    ratio = np.maximum(p / t, t / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - t) / t)),
        "delta1": float(np.mean(ratio < 1.25)),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "valid_pixels": count,
    }


@dataclass(frozen=True)
class DepthEvaluator:
    min_depth: float = 1e-3
    max_depth: float | None = None
    affine_align: bool = True

    def evaluate(
        self,
        prediction: NDArray[np.generic],
        target: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> dict[str, Any]:
        pred = np.asarray(prediction, dtype=np.float32)
        truth = np.asarray(target, dtype=np.float32)
        if pred.shape != truth.shape or pred.ndim != 2:
            raise ValueError(f"depth arrays must share shape [H,W], got {pred.shape}/{truth.shape}")
        mask = np.isfinite(truth) & (truth > self.min_depth)
        mask &= np.isfinite(pred)
        if self.max_depth is not None:
            mask &= truth <= self.max_depth
        if valid_mask is not None:
            explicit = np.asarray(valid_mask, dtype=bool)
            if explicit.shape != truth.shape:
                raise ValueError("valid mask must have the same shape as depth")
            mask &= explicit
        result: dict[str, Any] = {
            "raw": _depth_metrics(pred, truth, mask, self.min_depth)
        }
        if self.affine_align:
            aligned, scale, shift = align_scale_shift(pred, truth, mask)
            result["affine_aligned"] = _depth_metrics(
                aligned, truth, mask, self.min_depth
            )
            result["alignment"] = {"scale": scale, "shift": shift}
        return result
