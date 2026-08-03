"""Dependency-light visualizations for segmentation, depth, and normals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from perception_diffusion.codecs.segmentation import pascal_palette


def vae_pixels_to_rgb(pixels: torch.Tensor | np.ndarray) -> np.ndarray:
    array = (
        pixels.detach().float().cpu().numpy()
        if isinstance(pixels, torch.Tensor)
        else np.asarray(pixels)
    )
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("batched visualization accepts one sample")
        array = array[0]
    if array.shape[0] != 3:
        raise ValueError("VAE pixels must have shape [3,H,W]")
    rgb = np.moveaxis(np.clip(array, -1.0, 1.0), 0, -1)
    return np.rint((rgb + 1.0) * 127.5).astype(np.uint8)


def _depth_to_rgb(depth: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid &= valid_mask.astype(bool)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        lower, upper = np.quantile(depth[valid], [0.02, 0.98])
        if upper <= lower:
            upper = lower + 1.0
        normalized[valid] = np.clip((depth[valid] - lower) / (upper - lower), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~valid] = 0.0
    return np.rint(rgb * 255.0).astype(np.uint8)


def task_value_to_rgb(
    task_name: str,
    values: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    num_classes: int = 150,
) -> np.ndarray:
    array = np.asarray(values)
    if task_name == "segmentation":
        if array.ndim != 2:
            raise ValueError("segmentation visualization requires [H,W] IDs")
        palette = pascal_palette(max(1, min(num_classes, 256)))
        valid = array != 255
        if valid_mask is not None:
            valid &= valid_mask.astype(bool)
        rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
        safe = np.clip(array, 0, len(palette) - 1)
        rgb[valid] = palette[safe[valid]]
        return rgb
    if task_name == "depth":
        if array.ndim != 2:
            raise ValueError("depth visualization requires [H,W]")
        return _depth_to_rgb(array.astype(np.float32), valid_mask)
    if task_name == "normal":
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("normal visualization requires [3,H,W]")
        normals = np.moveaxis(np.clip(array, -1.0, 1.0), 0, -1)
        rgb = np.rint((normals + 1.0) * 127.5).astype(np.uint8)
        if valid_mask is not None:
            rgb[~valid_mask.astype(bool)] = 0
        return rgb
    raise ValueError(f"unsupported visualization task: {task_name}")


def _save_horizontal(images: list[np.ndarray], path: str | Path) -> Path:
    if not images:
        raise ValueError("at least one image is required")
    height = images[0].shape[0]
    if any(image.shape[0] != height or image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("panel images must share H and have three channels")
    panel = np.concatenate(images, axis=1)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(destination)
    return destination


def save_reconstruction_panel(
    original_pixels: torch.Tensor,
    reconstructed_pixels: torch.Tensor,
    path: str | Path,
) -> Path:
    original = vae_pixels_to_rgb(original_pixels)
    reconstructed = vae_pixels_to_rgb(reconstructed_pixels)
    error = np.abs(original.astype(np.int16) - reconstructed.astype(np.int16))
    error = np.clip(error * 4, 0, 255).astype(np.uint8)
    return _save_horizontal([original, reconstructed, error], path)


def save_prediction_panel(
    task_name: str,
    image_pixels: torch.Tensor,
    prediction: np.ndarray,
    target: np.ndarray,
    path: str | Path,
    *,
    valid_mask: np.ndarray | None = None,
    num_classes: int = 150,
) -> Path:
    image = vae_pixels_to_rgb(image_pixels)
    predicted_rgb = task_value_to_rgb(
        task_name, prediction, valid_mask=valid_mask, num_classes=num_classes
    )
    target_rgb = task_value_to_rgb(
        task_name, target, valid_mask=valid_mask, num_classes=num_classes
    )
    error = np.abs(predicted_rgb.astype(np.int16) - target_rgb.astype(np.int16))
    return _save_horizontal(
        [image, target_rgb, predicted_rgb, np.clip(error * 3, 0, 255).astype(np.uint8)],
        path,
    )
