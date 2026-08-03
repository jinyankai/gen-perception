"""Unified sample and collation contract for dense perception datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate


REQUIRED_SAMPLE_KEYS = frozenset(
    {
        "image",
        "target",
        "valid_mask",
        "native_target",
        "task_name",
        "sample_id",
        "source_index",
        "original_size",
        "query_class_id",
        "text_condition",
    }
)


def validate_unified_sample(sample: Mapping[str, Any]) -> None:
    """Validate the dataset-side contract before latent encoding."""

    missing = sorted(REQUIRED_SAMPLE_KEYS - set(sample))
    if missing:
        raise ValueError(f"dataset sample is missing keys: {missing}")
    image = sample["image"]
    target = sample["target"]
    valid_mask = sample["valid_mask"]
    native_target = sample["native_target"]
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be a [3,H,W] tensor")
    if not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] != 3:
        raise ValueError("target must be a [3,H,W] tensor")
    if image.shape[1:] != target.shape[1:]:
        raise ValueError("image and encoded target spatial shapes must match")
    if image.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("image and target must use torch.float32")
    if not torch.isfinite(image).all() or not torch.isfinite(target).all():
        raise ValueError("image and encoded target must be finite")
    if image.numel() and (float(image.min()) < -1.0001 or float(image.max()) > 1.0001):
        raise ValueError("image values must lie in [-1,1]")
    if target.numel() and (
        float(target.min()) < -1.0001 or float(target.max()) > 1.0001
    ):
        raise ValueError("encoded target values must lie in [-1,1]")
    if (
        not isinstance(valid_mask, torch.Tensor)
        or valid_mask.shape != (1, *image.shape[1:])
        or valid_mask.dtype != torch.bool
    ):
        raise ValueError("valid_mask must be a bool [1,H,W] tensor")
    if not bool(valid_mask.any()):
        raise ValueError("sample contains no valid target pixels")
    if not isinstance(native_target, torch.Tensor) or native_target.ndim != 3:
        raise ValueError("native_target must be a [C,H,W] tensor")
    if native_target.shape[1:] != image.shape[1:]:
        raise ValueError("native_target and image spatial shapes must match")
    if not isinstance(sample["task_name"], str) or not sample["task_name"]:
        raise ValueError("task_name must be a non-empty string")
    if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
        raise ValueError("sample_id must be a non-empty string")
    original_size = sample["original_size"]
    if not isinstance(original_size, torch.Tensor) or original_size.shape != (2,):
        raise ValueError("original_size must be a [2] tensor containing H,W")


def collate_task_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate a task-homogeneous batch and keep ``task_name`` scalar."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    for sample in samples:
        validate_unified_sample(sample)
    task_names = {str(sample["task_name"]) for sample in samples}
    if len(task_names) != 1:
        raise ValueError(f"mixed-task batches are forbidden, got {sorted(task_names)}")
    batch = default_collate(samples)
    batch["task_name"] = next(iter(task_names))
    return batch

