"""One task-conditioned denoiser shared by all stage-one perception tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .conditioning import TaskTokenConditioner


@dataclass
class UnifiedDenoiserOutput:
    sample: torch.Tensor
    encoder_hidden_states: torch.Tensor


class UnifiedPerceptionDenoiser(nn.Module):
    """Concatenate image/target latents and call one cross-attention U-Net."""

    def __init__(
        self,
        shared_unet: nn.Module,
        conditioner: TaskTokenConditioner,
        *,
        image_latent_channels: int = 4,
        target_latent_channels: int = 4,
    ) -> None:
        super().__init__()
        if image_latent_channels <= 0 or target_latent_channels <= 0:
            raise ValueError("latent channel counts must be positive")
        self.shared_unet = shared_unet
        self.conditioner = conditioner
        self.image_latent_channels = image_latent_channels
        self.target_latent_channels = target_latent_channels

    def forward(
        self,
        image_latent: torch.Tensor,
        noisy_target_latent: torch.Tensor,
        timestep: torch.Tensor | int,
        task_names: str | Sequence[str],
        *,
        text_hidden_states: torch.Tensor | None = None,
        use_task_condition: bool | None = None,
        use_text_condition: bool | None = None,
        **unet_kwargs: object,
    ) -> UnifiedDenoiserOutput:
        if image_latent.ndim != 4 or noisy_target_latent.ndim != 4:
            raise ValueError("image and noisy target latents must have shape [B,C,H,W]")
        if image_latent.shape[0] != noisy_target_latent.shape[0]:
            raise ValueError("image and target latent batch sizes must match")
        if image_latent.shape[2:] != noisy_target_latent.shape[2:]:
            raise ValueError("image and target latent spatial shapes must match")
        if image_latent.shape[1] != self.image_latent_channels:
            raise ValueError(
                f"expected {self.image_latent_channels} image latent channels, "
                f"got {image_latent.shape[1]}"
            )
        if noisy_target_latent.shape[1] != self.target_latent_channels:
            raise ValueError(
                f"expected {self.target_latent_channels} target latent channels, "
                f"got {noisy_target_latent.shape[1]}"
            )

        conditioning = self.conditioner(
            task_names,
            batch_size=image_latent.shape[0],
            text_hidden_states=text_hidden_states,
            use_task_condition=use_task_condition,
            use_text_condition=use_text_condition,
        )
        model_input = torch.cat([image_latent, noisy_target_latent], dim=1)
        raw_output = self.shared_unet(
            model_input,
            timestep,
            encoder_hidden_states=conditioning,
            **unet_kwargs,
        )
        sample = raw_output.sample if hasattr(raw_output, "sample") else raw_output
        if not isinstance(sample, torch.Tensor):
            raise TypeError("shared U-Net must return a tensor or an object with tensor .sample")
        return UnifiedDenoiserOutput(sample=sample, encoder_hidden_states=conditioning)
