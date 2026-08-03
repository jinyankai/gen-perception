"""Surface-normal target codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import EncodedTarget, ensure_chw3, ensure_hw_mask


@dataclass(frozen=True)
class NormalCodec:
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def encode(
        self,
        normals: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> EncodedTarget:
        chw = ensure_chw3(normals, "surface normals")
        norms = np.linalg.norm(chw, axis=0)
        valid = np.all(np.isfinite(chw), axis=0) & (norms > self.epsilon)
        valid &= ensure_hw_mask(valid_mask, chw.shape[1:])
        normalized = np.zeros_like(chw, dtype=np.float32)
        normalized[:, valid] = chw[:, valid] / norms[valid]
        return EncodedTarget(normalized, valid)

    def decode(
        self,
        encoded: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> NDArray[np.float32]:
        chw = ensure_chw3(encoded, "encoded normals")
        norms = np.linalg.norm(chw, axis=0)
        valid = np.all(np.isfinite(chw), axis=0) & (norms > self.epsilon)
        valid &= ensure_hw_mask(valid_mask, chw.shape[1:])
        normalized = np.zeros_like(chw, dtype=np.float32)
        normalized[:, valid] = chw[:, valid] / norms[valid]
        return normalized
