"""Semantic segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentationEvaluator:
    num_classes: int
    ignore_label: int = 255
    _confusion: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self._confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        self._confusion.fill(0)

    def update(
        self,
        prediction: NDArray[np.generic],
        target: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> None:
        pred = np.asarray(prediction)
        truth = np.asarray(target)
        if pred.shape != truth.shape:
            raise ValueError(f"prediction/target shapes differ: {pred.shape} vs {truth.shape}")
        if pred.ndim != 2:
            raise ValueError(f"segmentation arrays must have shape [H,W], got {pred.shape}")
        if not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(
            truth.dtype, np.integer
        ):
            raise TypeError("segmentation prediction and target must be integer arrays")
        mask = truth != self.ignore_label
        mask &= (truth >= 0) & (truth < self.num_classes)
        if valid_mask is not None:
            explicit = np.asarray(valid_mask, dtype=bool)
            if explicit.shape != truth.shape:
                raise ValueError(f"valid mask shape differs: {explicit.shape} vs {truth.shape}")
            mask &= explicit
        invalid_prediction = mask & ((pred < 0) | (pred >= self.num_classes))
        if np.any(invalid_prediction):
            values = np.unique(pred[invalid_prediction])[:10].tolist()
            raise ValueError(f"prediction contains invalid class IDs: {values}")
        indices = self.num_classes * truth[mask].astype(np.int64) + pred[mask].astype(np.int64)
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self._confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self._confusion.astype(np.float64)
        true_positive = np.diag(matrix)
        target_count = matrix.sum(axis=1)
        predicted_count = matrix.sum(axis=0)
        union = target_count + predicted_count - true_positive
        per_class = np.divide(
            true_positive,
            union,
            out=np.full(self.num_classes, np.nan, dtype=np.float64),
            where=union > 0,
        )
        total = matrix.sum()
        pixel_accuracy = float(true_positive.sum() / total) if total > 0 else float("nan")
        miou = float(np.nanmean(per_class)) if np.any(union > 0) else float("nan")
        return {
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": [None if np.isnan(v) else float(v) for v in per_class],
            "valid_pixels": int(total),
            "confusion_matrix": self._confusion.tolist(),
        }

    def evaluate(
        self,
        prediction: NDArray[np.generic],
        target: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> dict[str, Any]:
        self.reset()
        self.update(prediction, target, valid_mask)
        return self.compute()
