"""Surface-normal angular metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from perception_diffusion.codecs.base import ensure_chw3


@dataclass(frozen=True)
class NormalEvaluator:
    epsilon: float = 1e-6

    def evaluate(
        self,
        prediction: NDArray[np.generic],
        target: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> dict[str, float | int]:
        pred = ensure_chw3(prediction, "predicted normals")
        truth = ensure_chw3(target, "target normals")
        if pred.shape != truth.shape:
            raise ValueError(f"normal shapes differ: {pred.shape} vs {truth.shape}")
        pred_norm = np.linalg.norm(pred, axis=0)
        truth_norm = np.linalg.norm(truth, axis=0)
        mask = np.all(np.isfinite(pred), axis=0) & np.all(np.isfinite(truth), axis=0)
        mask &= (pred_norm > self.epsilon) & (truth_norm > self.epsilon)
        if valid_mask is not None:
            explicit = np.asarray(valid_mask, dtype=bool)
            if explicit.shape != mask.shape:
                raise ValueError(f"valid mask must have shape {mask.shape}, got {explicit.shape}")
            mask &= explicit
        if not np.any(mask):
            raise ValueError("no valid surface-normal pixels remain for evaluation")
        pred_unit = pred[:, mask] / pred_norm[mask]
        truth_unit = truth[:, mask] / truth_norm[mask]
        cosine = np.clip(np.sum(pred_unit * truth_unit, axis=0), -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        return {
            "mean_angular_error": float(np.mean(angles)),
            "median_angular_error": float(np.median(angles)),
            "acc_11_25": float(np.mean(angles < 11.25)),
            "acc_22_5": float(np.mean(angles < 22.5)),
            "acc_30": float(np.mean(angles < 30.0)),
            "valid_pixels": int(mask.sum()),
        }
