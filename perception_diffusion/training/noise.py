"""Configurable Gaussian and Marigold-style annealed multi-resolution noise."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.nn import functional as F


DownscaleStrategy = Literal["original", "every_layer", "power_of_two", "random_step"]


def annealed_noise_strength(
    base_strength: float,
    timesteps: torch.Tensor,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Apply Marigold's linear timestep-dependent multi-resolution strength."""

    if not 0.0 <= base_strength <= 1.0:
        raise ValueError("multi-resolution noise strength must lie in [0,1]")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [B]")
    return base_strength * timesteps.float() / float(num_train_timesteps)


def multi_resolution_noise_like(
    reference: torch.Tensor,
    *,
    strength: float | torch.Tensor = 0.9,
    downscale_strategy: DownscaleStrategy = "original",
    generator: torch.Generator | None = None,
    max_levels: int = 10,
) -> torch.Tensor:
    """Sample the multi-resolution noise field used by Marigold trainers.

    This is an independent, validated adaptation of Marigold's Apache-2.0
    ``multi_res_noise_like`` helper. It preserves the four downscale strategies,
    per-sample annealed strength, and final unit-variance normalization.
    """

    if reference.ndim != 4:
        raise ValueError("reference must have shape [B,C,H,W]")
    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    if downscale_strategy not in {
        "original",
        "every_layer",
        "power_of_two",
        "random_step",
    }:
        raise ValueError(f"unknown downscale strategy: {downscale_strategy}")
    if isinstance(strength, torch.Tensor):
        if strength.ndim != 1 or strength.shape[0] != reference.shape[0]:
            raise ValueError("tensor strength must have shape [B]")
        if not torch.isfinite(strength).all():
            raise ValueError("strength must be finite")
        if bool(((strength < 0.0) | (strength > 1.0)).any()):
            raise ValueError("strength must lie in [0,1]")
        strength_value: float | torch.Tensor = strength.to(
            device=reference.device, dtype=reference.dtype
        ).reshape(-1, 1, 1, 1)
    else:
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("strength must lie in [0,1]")
        strength_value = float(strength)

    batch, channels, height, width = reference.shape
    noise = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    current_height, current_width = height, width
    levels = (
        min(max_levels, int(math.log2(min(height, width))))
        if downscale_strategy == "every_layer"
        else max_levels
    )

    for level in range(levels):
        if downscale_strategy == "original":
            ratio = float(
                torch.rand((), device=reference.device, generator=generator) * 2.0
                + 2.0
            )
            current_height = max(1, int(height / (ratio**level)))
            current_width = max(1, int(width / (ratio**level)))
        elif downscale_strategy == "power_of_two":
            current_height = max(1, int(height / (2**level)))
            current_width = max(1, int(width / (2**level)))
        elif downscale_strategy == "random_step":
            ratio = float(
                torch.rand((), device=reference.device, generator=generator) * 2.0
                + 2.0
            )
            current_height = max(1, int(current_height / ratio))
            current_width = max(1, int(current_width / ratio))
        else:
            current_height = max(1, current_height // 2)
            current_width = max(1, current_width // 2)

        low_resolution = torch.randn(
            (batch, channels, current_height, current_width),
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        upsampled = F.interpolate(
            low_resolution,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        noise = noise + upsampled * (strength_value**level)
        if current_height == 1 or current_width == 1:
            break

    standard_deviation = noise.float().std()
    if not torch.isfinite(standard_deviation) or float(standard_deviation) <= 0.0:
        raise FloatingPointError("multi-resolution noise has invalid variance")
    return noise / standard_deviation.to(dtype=noise.dtype)


@dataclass(frozen=True)
class ConfiguredNoiseSampler:
    """Resolve the versioned noise configuration into one sampling interface."""

    enabled: bool = False
    strength: float = 0.9
    annealed: bool = True
    downscale_strategy: DownscaleStrategy = "original"
    max_levels: int = 10

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConfiguredNoiseSampler":
        noise = config["model"].get("noise", {})
        multi_scale = noise.get("multi_scale", {})
        return cls(
            enabled=bool(multi_scale.get("enabled", False)),
            strength=float(multi_scale.get("strength", 0.9)),
            annealed=bool(multi_scale.get("annealed", True)),
            downscale_strategy=str(
                multi_scale.get("downscale_strategy", "original")
            ),  # type: ignore[arg-type]
            max_levels=int(multi_scale.get("max_levels", 10)),
        )

    def sample_like(
        self,
        reference: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.randn(
                reference.shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
        strength: float | torch.Tensor = self.strength
        if self.annealed:
            strength = annealed_noise_strength(
                self.strength, timesteps, num_train_timesteps
            )
        return multi_resolution_noise_like(
            reference,
            strength=strength,
            downscale_strategy=self.downscale_strategy,
            generator=generator,
            max_levels=self.max_levels,
        )
