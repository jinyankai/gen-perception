import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from perception_diffusion.models import ResidualPreVAEAdapter, VisualLatentPathway
from perception_diffusion.training import (
    ConfiguredNoiseSampler,
    annealed_noise_strength,
    multi_resolution_noise_like,
)


class _Distribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        del generator
        return self.value


class _TinyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.decoder = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        latent = F.avg_pool2d(self.encoder(pixels), 2)
        return SimpleNamespace(latent_dist=_Distribution(latent))

    def decode(self, latent: torch.Tensor) -> SimpleNamespace:
        pixels = self.decoder(F.interpolate(latent, scale_factor=2, mode="nearest"))
        return SimpleNamespace(sample=pixels)


class MultiResolutionNoiseTest(unittest.TestCase):
    def test_annealed_strength_matches_marigold_timestep_rule(self):
        timesteps = torch.tensor([0, 500, 999])
        strength = annealed_noise_strength(0.9, timesteps, 1000)
        self.assertTrue(torch.allclose(strength, torch.tensor([0.0, 0.45, 0.8991])))

    def test_multi_resolution_noise_is_finite_unit_variance_and_reproducible(self):
        reference = torch.zeros(2, 4, 8, 8)
        first = multi_resolution_noise_like(
            reference,
            strength=torch.tensor([0.0, 0.9]),
            downscale_strategy="power_of_two",
            generator=torch.Generator().manual_seed(7),
        )
        second = multi_resolution_noise_like(
            reference,
            strength=torch.tensor([0.0, 0.9]),
            downscale_strategy="power_of_two",
            generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertAlmostEqual(1.0, float(first.std()), places=5)

    def test_configured_sampler_uses_standard_noise_when_disabled(self):
        sampler = ConfiguredNoiseSampler(enabled=False)
        reference = torch.zeros(1, 4, 4, 4)
        noise = sampler.sample_like(
            reference,
            timesteps=torch.tensor([2]),
            num_train_timesteps=10,
            generator=torch.Generator().manual_seed(11),
        )
        expected = torch.randn(reference.shape, generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(noise, expected))


class VisualLatentPathwayTest(unittest.TestCase):
    def test_encode_pair_freezes_vae_and_downsamples_mask_conservatively(self):
        vae = _TinyVAE()
        pathway = VisualLatentPathway(vae)
        images = torch.rand(1, 3, 8, 8) * 2.0 - 1.0
        targets = torch.rand(1, 3, 8, 8) * 2.0 - 1.0
        valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        valid[:, :, :2, :2] = False

        pair = pathway.encode_pair(images, targets, valid, "depth")
        decoded = pathway.decode_latents(pair.target_latent)

        self.assertEqual((1, 4, 4, 4), tuple(pair.image_latent.shape))
        self.assertEqual((1, 1, 4, 4), tuple(pair.latent_valid_mask.shape))
        self.assertFalse(bool(pair.latent_valid_mask[0, 0, 0, 0]))
        self.assertEqual((1, 3, 8, 8), tuple(decoded.shape))
        self.assertFalse(any(parameter.requires_grad for parameter in vae.parameters()))

    def test_target_adapter_receives_gradient_through_frozen_vae(self):
        adapter = ResidualPreVAEAdapter(hidden_channels=4, num_blocks=1)
        pathway = VisualLatentPathway(_TinyVAE(), adapter)
        targets = torch.rand(1, 3, 8, 8) * 2.0 - 1.0

        latent, _ = pathway.encode_targets(targets, "normal")
        latent.square().mean().backward()

        self.assertIsNotNone(adapter.output.weight.grad)
        self.assertGreater(float(adapter.output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
