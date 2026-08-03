"""One latent sampling loop shared by all perception tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from perception_diffusion.models import UnifiedPerceptionDenoiser


class UnifiedLatentSampler:
    """Run a Diffusers-compatible scheduler against the shared denoiser."""

    def __init__(self, denoiser: UnifiedPerceptionDenoiser, scheduler: Any) -> None:
        self.denoiser = denoiser
        self.scheduler = scheduler

    def _set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        try:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)

    @staticmethod
    def _previous_sample(step_output: Any) -> torch.Tensor:
        if hasattr(step_output, "prev_sample"):
            sample = step_output.prev_sample
        elif isinstance(step_output, tuple) and step_output:
            sample = step_output[0]
        else:
            sample = step_output
        if not isinstance(sample, torch.Tensor):
            raise TypeError("scheduler.step must return a tensor or an object with prev_sample")
        return sample

    @torch.no_grad()
    def sample(
        self,
        image_latent: torch.Tensor,
        task_names: str | Sequence[str],
        *,
        num_inference_steps: int,
        text_hidden_states: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if image_latent.ndim != 4:
            raise ValueError("image_latent must have shape [B,C,H,W]")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        target_shape = (
            image_latent.shape[0],
            self.denoiser.target_latent_channels,
            image_latent.shape[2],
            image_latent.shape[3],
        )
        if initial_noise is None:
            target_latent = torch.randn(
                target_shape,
                device=image_latent.device,
                dtype=image_latent.dtype,
                generator=generator,
            )
        else:
            if tuple(initial_noise.shape) != target_shape:
                raise ValueError(
                    f"initial_noise must have shape {target_shape}, got {initial_noise.shape}"
                )
            target_latent = initial_noise.clone()
        init_noise_sigma = float(getattr(self.scheduler, "init_noise_sigma", 1.0))
        target_latent = target_latent * init_noise_sigma

        self._set_timesteps(num_inference_steps, image_latent.device)
        for timestep in self.scheduler.timesteps:
            model_target = target_latent
            if hasattr(self.scheduler, "scale_model_input"):
                model_target = self.scheduler.scale_model_input(
                    target_latent, timestep
                )
            output = self.denoiser(
                image_latent,
                model_target,
                timestep,
                task_names,
                text_hidden_states=text_hidden_states,
            )
            try:
                step_output = self.scheduler.step(
                    output.sample,
                    timestep,
                    target_latent,
                    generator=generator,
                )
            except TypeError:
                step_output = self.scheduler.step(
                    output.sample, timestep, target_latent
                )
            target_latent = self._previous_sample(step_output)
        return target_latent
