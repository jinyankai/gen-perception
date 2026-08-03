"""Frozen RGB-VAE pathway shared by image and task-target representations."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .target_adapters import IdentityPreVAEAdapter


@dataclass(frozen=True)
class VisualLatentPair:
    image_latent: torch.Tensor
    target_latent: torch.Tensor
    latent_valid_mask: torch.Tensor
    adapted_target: torch.Tensor


class VisualLatentPathway(nn.Module):
    """Encode RGB and canonical three-channel targets with one frozen VAE.

    Target semantics remain owned by deterministic codecs. The optional
    task-specific pre-VAE adapter can receive gradients through the frozen VAE,
    but VAE parameters are never trainable.
    """

    def __init__(
        self,
        vae: nn.Module,
        target_adapter: nn.Module | None = None,
        *,
        scaling_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.target_adapter = target_adapter or IdentityPreVAEAdapter()
        if scaling_factor is None:
            config = getattr(vae, "config", None)
            scaling_factor = (
                config.get("scaling_factor")
                if isinstance(config, dict)
                else getattr(config, "scaling_factor", None)
            )
        if not isinstance(scaling_factor, (int, float)) or scaling_factor <= 0:
            raise ValueError("VAE must expose a positive scaling_factor")
        self.scaling_factor = float(scaling_factor)
        self.vae.requires_grad_(False)
        self.vae.eval()

    @staticmethod
    def _validate_pixels(pixels: torch.Tensor, label: str) -> None:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(f"{label} must have shape [B,3,H,W]")
        if not torch.isfinite(pixels).all():
            raise ValueError(f"{label} must be finite")
        if pixels.numel():
            minimum = float(pixels.detach().amin())
            maximum = float(pixels.detach().amax())
            if minimum < -1.0001 or maximum > 1.0001:
                raise ValueError(
                    f"{label} must lie in [-1,1], got [{minimum},{maximum}]"
                )

    @staticmethod
    def _latent_from_encoder_output(
        encoded: Any,
        *,
        sample_posterior: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        distribution = getattr(encoded, "latent_dist", encoded)
        if sample_posterior:
            sampler = getattr(distribution, "sample", None)
            if not callable(sampler):
                raise TypeError("VAE encoder output has no latent distribution sampler")
            try:
                latent = sampler(generator=generator)
            except TypeError:
                latent = sampler()
        else:
            mode = getattr(distribution, "mode", None)
            if callable(mode):
                latent = mode()
            elif isinstance(distribution, torch.Tensor):
                latent = distribution
            else:
                raise TypeError("VAE encoder output has no deterministic latent mode")
        if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
            raise TypeError("VAE encoder must produce a [B,C,H,W] tensor")
        return latent

    def _encode(
        self,
        pixels: torch.Tensor,
        *,
        sample_posterior: bool,
        generator: torch.Generator | None,
        track_input_gradients: bool,
    ) -> torch.Tensor:
        context = nullcontext() if track_input_gradients else torch.no_grad()
        with context:
            encoded = self.vae.encode(pixels)
            latent = self._latent_from_encoder_output(
                encoded,
                sample_posterior=sample_posterior,
                generator=generator,
            )
            latent = latent * self.scaling_factor
        if not torch.isfinite(latent).all():
            raise FloatingPointError("VAE encoder produced non-finite latents")
        return latent

    def encode_images(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self._validate_pixels(images, "images")
        return self._encode(
            images,
            sample_posterior=sample_posterior,
            generator=generator,
            track_input_gradients=False,
        )

    def encode_targets(
        self,
        targets: torch.Tensor,
        task_names: str | Sequence[str],
        *,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_pixels(targets, "targets")
        adapted = self.target_adapter(targets, task_names)
        self._validate_pixels(adapted, "adapted targets")
        adapter_trainable = any(
            parameter.requires_grad for parameter in self.target_adapter.parameters()
        )
        latent = self._encode(
            adapted,
            sample_posterior=sample_posterior,
            generator=generator,
            track_input_gradients=adapter_trainable,
        )
        return latent, adapted

    @staticmethod
    def downsample_valid_mask(
        valid_mask: torch.Tensor, latent_size: tuple[int, int]
    ) -> torch.Tensor:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
            raise ValueError("valid_mask must have shape [B,1,H,W]")
        if min(latent_size) <= 0:
            raise ValueError("latent_size must be positive")
        invalid = (~valid_mask.bool()).float()
        latent_invalid = F.adaptive_max_pool2d(invalid, latent_size).bool()
        latent_valid = ~latent_invalid
        if not bool(latent_valid.any()):
            raise ValueError("valid mask contains no full valid latent cell")
        return latent_valid

    def encode_pair(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        task_names: str | Sequence[str],
        *,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> VisualLatentPair:
        image_latent = self.encode_images(
            images,
            sample_posterior=sample_posterior,
            generator=generator,
        )
        target_latent, adapted = self.encode_targets(
            targets,
            task_names,
            sample_posterior=sample_posterior,
            generator=generator,
        )
        if image_latent.shape != target_latent.shape:
            raise ValueError(
                "image and target VAE latents must have identical shapes, got "
                f"{image_latent.shape} and {target_latent.shape}"
            )
        latent_valid = self.downsample_valid_mask(
            valid_mask, tuple(target_latent.shape[-2:])
        )
        return VisualLatentPair(
            image_latent=image_latent,
            target_latent=target_latent,
            latent_valid_mask=latent_valid,
            adapted_target=adapted,
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        *,
        clamp: bool = True,
        track_gradients: bool = False,
    ) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("latents must have shape [B,C,H,W]")
        context = nullcontext() if track_gradients else torch.no_grad()
        with context:
            decoded = self.vae.decode(latents / self.scaling_factor)
            pixels = decoded.sample if hasattr(decoded, "sample") else decoded
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise TypeError("VAE decoder must return a [B,3,H,W] tensor")
        if not torch.isfinite(pixels).all():
            raise FloatingPointError("VAE decoder produced non-finite pixels")
        return pixels.clamp(-1.0, 1.0) if clamp else pixels
