"""One latent-diffusion loss path shared by all registered tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from perception_diffusion.models import UnifiedPerceptionDenoiser

from .noise import ConfiguredNoiseSampler


@dataclass(frozen=True)
class UnifiedLatentBatch:
    image_latent: torch.Tensor
    clean_target_latent: torch.Tensor
    task_name: str
    latent_valid_mask: torch.Tensor | None = None
    text_hidden_states: torch.Tensor | None = None
    use_task_condition: bool | None = None
    use_text_condition: bool | None = None

    def validate(self) -> None:
        if self.image_latent.ndim != 4 or self.clean_target_latent.ndim != 4:
            raise ValueError("latent batches must have shape [B,C,H,W]")
        if self.image_latent.shape[0] != self.clean_target_latent.shape[0]:
            raise ValueError("image and target latent batch sizes must match")
        if self.image_latent.shape[2:] != self.clean_target_latent.shape[2:]:
            raise ValueError("image and target latent spatial shapes must match")
        if not self.task_name:
            raise ValueError("a task-homogeneous batch requires task_name")
        if self.latent_valid_mask is not None:
            if self.latent_valid_mask.ndim not in {3, 4}:
                raise ValueError("latent_valid_mask must have shape [B,H,W] or [B,1,H,W]")
            if self.latent_valid_mask.shape[0] != self.image_latent.shape[0]:
                raise ValueError("latent mask batch size must match latents")


@dataclass(frozen=True)
class DiffusionStepOutput:
    loss: torch.Tensor
    predicted_noise: torch.Tensor
    sampled_noise: torch.Tensor
    noisy_target_latent: torch.Tensor
    timesteps: torch.Tensor


def _num_train_timesteps(scheduler: Any) -> int:
    config = getattr(scheduler, "config", None)
    if isinstance(config, dict):
        value = config.get("num_train_timesteps")
    else:
        value = getattr(config, "num_train_timesteps", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("noise scheduler must expose positive config.num_train_timesteps")
    return value


_SUPPORTED_PREDICTION_TYPES = ("epsilon", "v_prediction", "sample")


def _scheduler_prediction_type(scheduler: Any) -> str:
    config = getattr(scheduler, "config", None)
    if isinstance(config, dict):
        value = config.get("prediction_type", "epsilon")
    else:
        value = getattr(config, "prediction_type", "epsilon")
    prediction_type = str(value)
    if prediction_type not in _SUPPORTED_PREDICTION_TYPES:
        raise ValueError(
            "unsupported scheduler prediction_type "
            f"{prediction_type!r}; expected one of {_SUPPORTED_PREDICTION_TYPES}"
        )
    return prediction_type


def _masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    squared_error = (prediction.float() - target.float()).square()
    if mask is None:
        return squared_error.mean()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=prediction.device, dtype=squared_error.dtype)
    if mask.shape[2:] != prediction.shape[2:]:
        mask = F.interpolate(mask, size=prediction.shape[2:], mode="nearest")
    if mask.shape[1] != 1:
        raise ValueError("latent_valid_mask must have exactly one channel")
    valid = mask.sum() * prediction.shape[1]
    if float(valid.detach()) <= 0:
        raise ValueError("latent_valid_mask contains no valid elements")
    return (squared_error * mask).sum() / valid


class UnifiedDiffusionTrainerCore(nn.Module):
    """Compute the same epsilon-prediction loss for every task."""

    def __init__(
        self,
        denoiser: UnifiedPerceptionDenoiser,
        noise_scheduler: Any,
        noise_sampler: ConfiguredNoiseSampler | None = None,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.noise_scheduler = noise_scheduler
        self.noise_sampler = noise_sampler
        self.prediction_type = _scheduler_prediction_type(noise_scheduler)

    def forward(
        self,
        batch: UnifiedLatentBatch,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DiffusionStepOutput:
        batch.validate()
        target = batch.clean_target_latent
        if timesteps is None:
            timesteps = torch.randint(
                0,
                _num_train_timesteps(self.noise_scheduler),
                (target.shape[0],),
                device=target.device,
                dtype=torch.long,
                generator=generator,
            )
        if timesteps.shape != (target.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        if noise is None:
            if self.noise_sampler is None:
                noise = torch.randn(
                    target.shape,
                    device=target.device,
                    dtype=target.dtype,
                    generator=generator,
                )
            else:
                noise = self.noise_sampler.sample_like(
                    target,
                    timesteps=timesteps,
                    num_train_timesteps=_num_train_timesteps(self.noise_scheduler),
                    generator=generator,
                )
        if noise.shape != target.shape:
            raise ValueError("sampled noise must match clean target latent shape")

        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)
        output = self.denoiser(
            batch.image_latent,
            noisy_target,
            timesteps,
            batch.task_name,
            text_hidden_states=batch.text_hidden_states,
            use_task_condition=batch.use_task_condition,
            use_text_condition=batch.use_text_condition,
        )
        target_for_loss = self._regression_target(target, noise, timesteps)
        if output.sample.shape != target_for_loss.shape:
            raise ValueError(
                "denoiser output/target shapes differ: "
                f"{output.sample.shape}/{target_for_loss.shape}"
            )
        loss = _masked_mse(output.sample, target_for_loss, batch.latent_valid_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("diffusion loss is not finite")
        return DiffusionStepOutput(
            loss=loss,
            predicted_noise=output.sample,
            sampled_noise=noise,
            noisy_target_latent=noisy_target,
            timesteps=timesteps,
        )

    def _regression_target(
        self,
        clean_target: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Build the per-timestep regression target for the configured objective."""

        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "sample":
            return clean_target
        # v_prediction
        get_velocity = getattr(self.noise_scheduler, "get_velocity", None)
        if not callable(get_velocity):
            raise TypeError(
                "prediction_type 'v_prediction' requires a scheduler with "
                "get_velocity(sample, noise, timesteps)"
            )
        return get_velocity(clean_target, noise, timesteps)
