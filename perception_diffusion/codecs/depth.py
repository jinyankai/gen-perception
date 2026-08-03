"""Monocular depth target codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .base import EncodedTarget, ensure_chw3, ensure_hw_mask


DepthRepresentation = Literal["linear", "inverse", "log"]


@dataclass(frozen=True)
class DepthCodec:
    min_depth: float
    max_depth: float
    representation: DepthRepresentation = "linear"

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_depth) or not np.isfinite(self.max_depth):
            raise ValueError("depth bounds must be finite")
        if self.min_depth <= 0 or self.max_depth <= self.min_depth:
            raise ValueError("depth bounds must satisfy 0 < min_depth < max_depth")
        if self.representation not in {"linear", "inverse", "log"}:
            raise ValueError(f"unsupported depth representation: {self.representation}")

    def _transform(self, depth: NDArray[np.floating]) -> NDArray[np.float32]:
        if self.representation == "linear":
            transformed = depth
        elif self.representation == "inverse":
            transformed = 1.0 / depth
        else:
            transformed = np.log(depth)
        return np.asarray(transformed, dtype=np.float32)

    def _inverse_transform(self, values: NDArray[np.floating]) -> NDArray[np.float32]:
        if self.representation == "linear":
            depth = values
        elif self.representation == "inverse":
            depth = 1.0 / np.maximum(values, np.finfo(np.float32).eps)
        else:
            depth = np.exp(values)
        return np.asarray(depth, dtype=np.float32)

    @property
    def transformed_bounds(self) -> tuple[float, float]:
        bounds = self._transform(np.array([self.min_depth, self.max_depth], dtype=np.float32))
        return float(np.min(bounds)), float(np.max(bounds))

    def encode(
        self,
        depth: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> EncodedTarget:
        array = np.asarray(depth, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"depth must have shape [H,W], got {array.shape}")
        valid = np.isfinite(array) & (array > 0)
        valid &= ensure_hw_mask(valid_mask, array.shape)
        clipped = np.clip(np.where(valid, array, self.min_depth), self.min_depth, self.max_depth)
        transformed = self._transform(clipped)
        lower, upper = self.transformed_bounds
        normalized = 2.0 * (transformed - lower) / (upper - lower) - 1.0
        normalized[~valid] = 0.0
        values = np.repeat(normalized[None, ...], 3, axis=0).astype(np.float32)
        return EncodedTarget(values, valid)

    def decode(
        self,
        encoded: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> NDArray[np.float32]:
        chw = ensure_chw3(encoded, "encoded depth")
        height, width = chw.shape[1:]
        valid = ensure_hw_mask(valid_mask, (height, width))
        scalar = np.clip(np.mean(chw, axis=0), -1.0, 1.0)
        lower, upper = self.transformed_bounds
        transformed = (scalar + 1.0) * 0.5 * (upper - lower) + lower
        depth = np.clip(self._inverse_transform(transformed), self.min_depth, self.max_depth)
        depth[~valid] = np.nan
        return depth.astype(np.float32)
