import unittest
from types import SimpleNamespace

import torch
from torch import nn

from perception_diffusion.inference import UnifiedLatentSampler
from perception_diffusion.models import TaskTokenConditioner, UnifiedPerceptionDenoiser
from perception_diffusion.training import (
    ConfiguredNoiseSampler,
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
)


class _DenoisingUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(8, 4, kernel_size=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        *,
        encoder_hidden_states: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep
        condition = encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
        return SimpleNamespace(sample=self.conv_in(sample) + condition)


class _Scheduler:
    init_noise_sigma = 1.0

    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=10)
        self.timesteps = torch.empty(0, dtype=torch.long)

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        del timesteps
        return clean + noise

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        self.timesteps = torch.arange(steps - 1, -1, -1, device=device)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor | int
    ) -> torch.Tensor:
        del timestep
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> SimpleNamespace:
        del timestep, generator
        return SimpleNamespace(prev_sample=sample - 0.1 * model_output)


def _system() -> tuple[UnifiedDiffusionTrainerCore, UnifiedLatentSampler]:
    conditioner = TaskTokenConditioner(
        ["segmentation", "depth", "normal"],
        cross_attention_dim=4,
        num_task_tokens=2,
        adapter_bottleneck_dim=2,
        text_input_dim=4,
    )
    denoiser = UnifiedPerceptionDenoiser(_DenoisingUNet(), conditioner)
    scheduler = _Scheduler()
    return (
        UnifiedDiffusionTrainerCore(denoiser, scheduler),
        UnifiedLatentSampler(denoiser, scheduler),
    )


class UnifiedTrainingTest(unittest.TestCase):
    def test_same_training_core_backpropagates_for_every_task(self):
        trainer, _ = _system()
        for task_name in ("segmentation", "depth", "normal"):
            trainer.zero_grad(set_to_none=True)
            batch = UnifiedLatentBatch(
                image_latent=torch.randn(1, 4, 4, 4),
                clean_target_latent=torch.randn(1, 4, 4, 4),
                task_name=task_name,
                latent_valid_mask=torch.ones(1, 1, 4, 4),
            )
            output = trainer(batch)
            output.loss.backward()
            self.assertTrue(torch.isfinite(output.loss))
            self.assertIsNotNone(trainer.denoiser.shared_unet.conv_in.weight.grad)

    def test_training_core_uses_configured_annealed_noise(self):
        trainer, _ = _system()
        trainer.noise_sampler = ConfiguredNoiseSampler(
            enabled=True,
            strength=0.9,
            annealed=True,
            downscale_strategy="original",
            max_levels=4,
        )
        batch = UnifiedLatentBatch(
            image_latent=torch.randn(2, 4, 8, 8),
            clean_target_latent=torch.randn(2, 4, 8, 8),
            task_name="depth",
        )
        output = trainer(
            batch,
            timesteps=torch.tensor([0, 9]),
            generator=torch.Generator().manual_seed(17),
        )

        self.assertEqual((2, 4, 8, 8), tuple(output.sampled_noise.shape))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertAlmostEqual(1.0, float(output.sampled_noise.std()), places=5)

    def test_same_sampler_switches_task_by_token(self):
        _, sampler = _system()
        image = torch.randn(1, 4, 4, 4)
        initial_noise = torch.zeros(1, 4, 4, 4)

        depth = sampler.sample(
            image,
            "depth",
            num_inference_steps=2,
            initial_noise=initial_noise,
        )
        normal = sampler.sample(
            image,
            "normal",
            num_inference_steps=2,
            initial_noise=initial_noise,
        )

        self.assertEqual((1, 4, 4, 4), tuple(depth.shape))
        self.assertFalse(torch.allclose(depth, normal))


if __name__ == "__main__":
    unittest.main()
