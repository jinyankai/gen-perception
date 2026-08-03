"""Discrete semantic segmentation target codecs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import EncodedTarget, ensure_chw3, ensure_hw_mask


@dataclass(frozen=True)
class SegmentationBinaryMaskCodec:
    """Encode a class-query binary mask in the frozen VAE's [-1,1] input range."""

    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("binary mask threshold must be in (0,1)")

    def encode(
        self,
        mask: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> EncodedTarget:
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"binary mask must have shape [H,W], got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("binary mask must be finite")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("binary mask values must be exactly 0 or 1")
        valid = ensure_hw_mask(valid_mask, array.shape)
        normalized = 2.0 * array.astype(np.float32) - 1.0
        normalized[~valid] = 0.0
        return EncodedTarget(np.repeat(normalized[None, ...], 3, axis=0), valid)

    def decode_scores(
        self,
        encoded: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> NDArray[np.float32]:
        chw = ensure_chw3(encoded, "encoded binary segmentation mask")
        valid = ensure_hw_mask(valid_mask, chw.shape[1:])
        scores = np.clip((np.mean(chw, axis=0) + 1.0) * 0.5, 0.0, 1.0)
        scores[~valid] = 0.0
        return scores.astype(np.float32)

    def decode(
        self,
        encoded: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> NDArray[np.uint8]:
        scores = self.decode_scores(encoded, valid_mask)
        return (scores >= self.threshold).astype(np.uint8)


def pascal_palette(size: int) -> NDArray[np.uint8]:
    """Create the deterministic bit-interleaved PASCAL palette."""

    if size <= 0 or size > 256:
        raise ValueError(f"palette size must be in [1,256], got {size}")
    palette = np.zeros((size, 3), dtype=np.uint8)
    for class_id in range(size):
        label = class_id
        bit = 0
        while label:
            palette[class_id, 0] |= ((label >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((label >> 1) & 1) << (7 - bit)
            palette[class_id, 2] |= ((label >> 2) & 1) << (7 - bit)
            bit += 1
            label >>= 3
    return palette


def _validate_labels(
    labels: NDArray[np.generic], num_classes: int, ignore_label: int
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(f"segmentation labels must have shape [H,W], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"segmentation labels must be integer, got {array.dtype}")
    result = np.asarray(array, dtype=np.int64)
    valid = result != ignore_label
    invalid_ids = valid & ((result < 0) | (result >= num_classes))
    if np.any(invalid_ids):
        bad = np.unique(result[invalid_ids])[:10].tolist()
        raise ValueError(f"labels outside [0,{num_classes - 1}] and ignore={ignore_label}: {bad}")
    return result, valid


@dataclass(frozen=True)
class SegmentationPaletteCodec:
    """Map class IDs to RGB palette values before VAE encoding.

    Decoding uses nearest-palette matching and never interpolates class IDs.
    The explicit valid mask disambiguates ignore pixels from class colors.
    """

    num_classes: int
    ignore_label: int = 255
    palette: NDArray[np.uint8] | None = None
    decode_chunk_size: int = 65_536

    def __post_init__(self) -> None:
        if self.num_classes <= 0 or self.num_classes > 256:
            raise ValueError("palette codec supports 1-256 classes")
        palette = pascal_palette(self.num_classes) if self.palette is None else self.palette
        palette = np.asarray(palette, dtype=np.uint8)
        if palette.shape != (self.num_classes, 3):
            raise ValueError(
                f"palette must have shape {(self.num_classes, 3)}, got {palette.shape}"
            )
        if self.decode_chunk_size <= 0:
            raise ValueError("decode_chunk_size must be positive")
        object.__setattr__(self, "palette", palette.copy())

    def encode(
        self,
        labels: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> EncodedTarget:
        label_ids, label_valid = _validate_labels(
            labels, self.num_classes, self.ignore_label
        )
        valid = label_valid & ensure_hw_mask(valid_mask, label_ids.shape)
        rgb = np.zeros((*label_ids.shape, 3), dtype=np.float32)
        rgb[valid] = self.palette[label_ids[valid]].astype(np.float32)
        normalized = rgb / 127.5 - 1.0
        return EncodedTarget(np.moveaxis(normalized, -1, 0), valid)

    def decode(
        self,
        encoded: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> NDArray[np.int64]:
        chw = ensure_chw3(encoded, "encoded segmentation")
        height, width = chw.shape[1:]
        valid = ensure_hw_mask(valid_mask, (height, width))
        pixels = np.moveaxis(chw, 0, -1).reshape(-1, 3)
        pixels = np.clip((pixels + 1.0) * 127.5, 0.0, 255.0)
        palette = self.palette.astype(np.float32)
        decoded = np.empty(pixels.shape[0], dtype=np.int64)
        for start in range(0, pixels.shape[0], self.decode_chunk_size):
            stop = min(start + self.decode_chunk_size, pixels.shape[0])
            delta = pixels[start:stop, None, :] - palette[None, :, :]
            decoded[start:stop] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        decoded = decoded.reshape(height, width)
        decoded[~valid] = self.ignore_label
        return decoded


@dataclass(frozen=True)
class SegmentationIdCodec:
    """Normalized-ID diagnostic codec.

    This codec is included only for round-trip comparison. Palette encoding is
    the default because RGB-VAE perturbations in a continuous ID axis can cause
    semantically arbitrary class substitutions.
    """

    num_classes: int
    ignore_label: int = 255

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("normalized ID codec requires at least two classes")

    def encode(
        self,
        labels: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> EncodedTarget:
        label_ids, label_valid = _validate_labels(
            labels, self.num_classes, self.ignore_label
        )
        valid = label_valid & ensure_hw_mask(valid_mask, label_ids.shape)
        normalized = np.zeros(label_ids.shape, dtype=np.float32)
        normalized[valid] = (
            2.0 * label_ids[valid].astype(np.float32) / (self.num_classes - 1) - 1.0
        )
        return EncodedTarget(np.repeat(normalized[None, ...], 3, axis=0), valid)

    def decode(
        self,
        encoded: NDArray[np.generic],
        valid_mask: NDArray[np.generic] | None = None,
    ) -> NDArray[np.int64]:
        chw = ensure_chw3(encoded, "encoded segmentation")
        height, width = chw.shape[1:]
        valid = ensure_hw_mask(valid_mask, (height, width))
        scalar = np.mean(chw, axis=0)
        label_ids = np.rint((np.clip(scalar, -1.0, 1.0) + 1.0) * 0.5 * (self.num_classes - 1))
        result = label_ids.astype(np.int64)
        result[~valid] = self.ignore_label
        return result
