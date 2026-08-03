"""Geometry-safe tensor transforms shared by dense prediction datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def rgb_to_vae_tensor(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Convert uint8 HWC RGB to float32 CHW in the frozen VAE's range."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"RGB image must be uint8 [H,W,3], got {array.shape}/{array.dtype}")
    tensor = torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1).float()
    tensor = F.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    return (tensor / 127.5 - 1.0).contiguous()


def resize_labels(labels: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Resize integer labels with nearest-neighbor interpolation only."""

    array = np.asarray(labels)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("labels must be an integer HxW array")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32)
    resized = F.interpolate(tensor[None, None], size=size, mode="nearest")[0, 0]
    return resized.to(torch.int64).contiguous()


def resize_depth_with_mask(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize depth without allowing invalid zeros to bleed into valid pixels."""

    values = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))[None, None]
    valid = torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=np.bool_))[None, None]
    weights = valid.to(torch.float32)
    clean_values = torch.where(valid, values, torch.zeros_like(values))
    numerator = F.interpolate(
        clean_values * weights, size=size, mode="bilinear", align_corners=False
    )
    denominator = F.interpolate(weights, size=size, mode="bilinear", align_corners=False)
    resized = numerator / denominator.clamp_min(1.0e-6)
    resized_valid = F.interpolate(weights, size=size, mode="nearest") > 0.5
    resized = torch.where(resized_valid, resized, torch.zeros_like(resized))
    return resized[0, 0].contiguous(), resized_valid[0, 0].contiguous()


def resize_normals_with_mask(
    normals: np.ndarray,
    valid_mask: np.ndarray,
    size: tuple[int, int],
    *,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly resize normal vectors, then restore unit length and validity."""

    array = np.asarray(normals, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"normals must have shape [3,H,W], got {array.shape}")
    values = torch.from_numpy(np.ascontiguousarray(array))[None]
    valid = torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=np.bool_))[None, None]
    values = torch.where(valid, values, torch.zeros_like(values))
    resized = F.interpolate(values, size=size, mode="bilinear", align_corners=False)[0]
    resized_valid = F.interpolate(valid.float(), size=size, mode="nearest")[0, 0] > 0.5
    norms = torch.linalg.vector_norm(resized, dim=0)
    resized_valid &= torch.isfinite(resized).all(dim=0) & (norms > epsilon)
    normalized = torch.zeros_like(resized)
    normalized[:, resized_valid] = resized[:, resized_valid] / norms[resized_valid]
    return normalized.contiguous(), resized_valid.contiguous()


def should_horizontal_flip(probability: float) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("horizontal flip probability must lie in [0,1]")
    return probability > 0.0 and bool(torch.rand(()) < probability)
