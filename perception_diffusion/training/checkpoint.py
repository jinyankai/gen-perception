"""Checkpoint save/resume support for the shared model and optimizer state."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a mapping")
    return payload


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    denoiser: nn.Module,
    target_adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    generator: torch.Generator,
) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "step": step,
        "denoiser": denoiser.state_dict(),
        "target_adapter": target_adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    denoiser: nn.Module,
    target_adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _torch_load(source, device)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    denoiser.load_state_dict(payload["denoiser"], strict=True)
    target_adapter.load_state_dict(payload["target_adapter"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    scaler.load_state_dict(payload["scaler"])
    generator.set_state(payload["generator_state"].cpu())
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_state_all"]]
        )
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    step = int(payload["step"])
    if step < 0:
        raise ValueError("checkpoint contains a negative step")
    return step


def load_model_checkpoint(
    path: str | Path,
    *,
    denoiser: nn.Module,
    target_adapter: nn.Module,
    device: torch.device,
) -> int:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _torch_load(source, device)
    denoiser.load_state_dict(payload["denoiser"], strict=True)
    target_adapter.load_state_dict(payload.get("target_adapter", {}), strict=True)
    return int(payload.get("step", 0))
