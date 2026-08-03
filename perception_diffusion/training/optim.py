"""Optimizer, learning-rate scheduler, and gradient helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def build_optimizer(
    denoiser: nn.Module,
    target_adapter: nn.Module,
    training_config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
    optimizer_config = training_config["optimizer"]
    if str(optimizer_config.get("name", "adamw")).casefold() != "adamw":
        raise ValueError("the unified runner currently supports optimizer.name=adamw")
    unet_parameters = [
        parameter
        for parameter in denoiser.shared_unet.parameters()  # type: ignore[attr-defined]
        if parameter.requires_grad
    ]
    conditioner_parameters: list[nn.Parameter] = []
    condition_adapter_parameters: list[nn.Parameter] = []
    for name, parameter in denoiser.conditioner.named_parameters():  # type: ignore[attr-defined]
        if not parameter.requires_grad:
            continue
        target = (
            condition_adapter_parameters
            if name.startswith("adapters.")
            else conditioner_parameters
        )
        target.append(parameter)
    target_adapter_parameters = [
        parameter for parameter in target_adapter.parameters() if parameter.requires_grad
    ]
    candidates = (
        ("shared_unet", unet_parameters, "shared_unet_lr"),
        ("conditioner", conditioner_parameters, "conditioner_lr"),
        ("condition_adapter", condition_adapter_parameters, "adapter_lr"),
        ("target_adapter", target_adapter_parameters, "target_adapter_lr"),
    )
    groups = [
        {
            "params": parameters,
            "lr": float(optimizer_config[learning_rate_key]),
            "group_name": group_name,
        }
        for group_name, parameters, learning_rate_key in candidates
        if parameters
    ]
    if not groups:
        raise ValueError("no trainable model parameter was selected")
    all_parameters = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise ValueError("optimizer parameter groups contain duplicate parameters")
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
        betas=tuple(float(value) for value in optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1.0e-8)),
    )
    return optimizer, all_parameters


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    *,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler_config = training_config.get("lr_scheduler", {})
    name = str(scheduler_config.get("name", "constant")).casefold()
    warmup_steps = int(scheduler_config.get("warmup_steps", 0))
    if warmup_steps < 0 or warmup_steps >= max_steps:
        if not (warmup_steps == 0 and max_steps > 0):
            raise ValueError("warmup_steps must lie in [0,max_steps)")
    if name not in {"constant", "linear", "cosine"}:
        raise ValueError(f"unsupported lr scheduler: {name}")

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if name == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def gradient_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        raise ValueError("no trainable parameter received a gradient")
    return torch.stack(squared).sum().sqrt()
