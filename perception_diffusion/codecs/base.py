"""Shared codec types and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class EncodedTarget:
    """A three-channel target representation and its valid-pixel mask."""

    values: FloatArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_mask)
        if values.ndim != 3 or values.shape[0] != 3:
            raise ValueError(f"encoded values must have shape [3,H,W], got {values.shape}")
        if mask.shape != values.shape[1:]:
            raise ValueError(
                f"valid mask must have shape {values.shape[1:]}, got {mask.shape}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"encoded values must be floating point, got {values.dtype}")
        if mask.dtype != np.bool_:
            raise TypeError(f"valid mask must be bool, got {mask.dtype}")


def ensure_hw_mask(mask: NDArray[np.generic] | None, shape: tuple[int, int]) -> BoolArray:
    """Validate or create a boolean HxW mask."""

    if mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {result.shape}")
    return result


def ensure_chw3(values: NDArray[np.generic], name: str) -> FloatArray:
    """Accept CHW or HWC three-channel arrays and return float32 CHW."""

    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(f"{name} must be rank 3, got {array.shape}")
    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"{name} must have three channels, got {array.shape}")
    return np.asarray(chw, dtype=np.float32)
