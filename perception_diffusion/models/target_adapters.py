"""Optional task-specific residual adaptation before the frozen RGB VAE."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from perception_diffusion.tasks import validate_task_names


class IdentityPreVAEAdapter(nn.Module):
    """Keep deterministic codec output unchanged."""

    def forward(
        self, pixels: torch.Tensor, task_names: str | Sequence[str]
    ) -> torch.Tensor:
        del task_names
        return pixels


class ResidualPreVAEAdapter(nn.Module):
    """Adapt canonical three-channel targets while preserving their value range."""

    def __init__(
        self,
        *,
        channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0 or num_blocks <= 0:
            raise ValueError("pre-VAE adapter dimensions must be positive")
        layers: list[nn.Module] = [
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_blocks - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size=3, padding=1
                    ),
                    nn.SiLU(),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.output = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4:
            raise ValueError(f"pre-VAE pixels must have shape [B,C,H,W], got {pixels.shape}")
        if pixels.shape[1] != self.output.out_channels:
            raise ValueError(
                f"expected {self.output.out_channels} pre-VAE channels, got {pixels.shape[1]}"
            )
        if not torch.isfinite(pixels).all():
            raise ValueError("pre-VAE pixels must be finite")
        lower = float(pixels.detach().amin())
        upper = float(pixels.detach().amax())
        if lower < -1.0001 or upper > 1.0001:
            raise ValueError(
                f"deterministic codec output must be in [-1,1], got [{lower},{upper}]"
            )
        residual = self.output(self.body(pixels))
        return torch.clamp(pixels + self.residual_scale * residual, -1.0, 1.0)


class TaskPreVAEAdapterBank(nn.Module):
    """Dispatch canonical target images through a task-owned residual CNN."""

    def __init__(
        self,
        task_names: Sequence[str],
        *,
        channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.task_names = validate_task_names(task_names)
        self.adapters = nn.ModuleDict(
            {
                task_name: ResidualPreVAEAdapter(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    num_blocks=num_blocks,
                    residual_scale_init=residual_scale_init,
                )
                for task_name in self.task_names
            }
        )

    def forward(
        self, pixels: torch.Tensor, task_names: str | Sequence[str]
    ) -> torch.Tensor:
        if isinstance(task_names, str):
            names = (task_names,) * pixels.shape[0]
        else:
            names = tuple(task_names)
        if len(names) != pixels.shape[0]:
            raise ValueError(
                f"received {len(names)} task names for batch size {pixels.shape[0]}"
            )
        unknown = sorted(set(names) - set(self.task_names))
        if unknown:
            raise ValueError(f"pre-VAE adapter is not configured for: {unknown}")
        if len(set(names)) == 1:
            return self.adapters[names[0]](pixels)
        outputs = [
            self.adapters[task_name](sample.unsqueeze(0))
            for sample, task_name in zip(pixels, names, strict=True)
        ]
        return torch.cat(outputs, dim=0)


def build_pre_vae_adapter(config: dict[str, Any]) -> nn.Module:
    """Build the configured identity or task-specific target adapter."""

    adapter_config = config["model"]["target_adapter"]
    if not adapter_config["enabled"]:
        return IdentityPreVAEAdapter()
    if adapter_config["type"] != "residual_cnn":
        raise ValueError(f"unsupported pre-VAE adapter: {adapter_config['type']}")
    return TaskPreVAEAdapterBank(
        adapter_config["task_names"],
        channels=int(adapter_config["channels"]),
        hidden_channels=int(adapter_config["hidden_channels"]),
        num_blocks=int(adapter_config["num_blocks"]),
        residual_scale_init=float(adapter_config["residual_scale_init"]),
    )
