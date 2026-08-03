import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from perception_diffusion.models import (
    IdentityPreVAEAdapter,
    ResidualPreVAEAdapter,
    TaskTokenConditioner,
    UnifiedPerceptionDenoiser,
    build_unified_denoiser,
    configure_unet_trainability,
    expand_unet_conv_in,
)
from perception_diffusion.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class _AttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = nn.Linear(4, 4)
        self.attn2 = nn.Linear(4, 4)


class _FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config: dict[str, int] = {"in_channels": 4}
        self.conv_in = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.block = _AttentionBlock()
        self.last_input: torch.Tensor | None = None
        self.last_conditioning: torch.Tensor | None = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        *,
        encoder_hidden_states: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        del timestep
        self.last_input = sample
        self.last_conditioning = encoder_hidden_states
        return SimpleNamespace(sample=sample[:, :4])


class SharedUNetTest(unittest.TestCase):
    def test_repeat_half_preserves_pretrained_response_for_duplicated_input(self):
        unet = _FakeUNet()
        image_latent = torch.randn(2, 4, 8, 8)
        expected = unet.conv_in(image_latent)

        replaced = expand_unet_conv_in(unet, 4, 4, initialization="repeat_half")
        actual = unet.conv_in(torch.cat([image_latent, image_latent], dim=1))

        self.assertTrue(replaced)
        self.assertEqual(8, unet.conv_in.in_channels)
        self.assertTrue(torch.allclose(expected, actual, atol=1.0e-6))

    def test_cross_attention_scope_does_not_unfreeze_self_attention(self):
        unet = _FakeUNet()
        summary = configure_unet_trainability(unet, "cross_attention")

        trainable = set(summary.trainable_names)
        self.assertIn("conv_in.weight", trainable)
        self.assertIn("block.attn2.weight", trainable)
        self.assertNotIn("block.attn1.weight", trainable)


class ConditioningTest(unittest.TestCase):
    def test_task_and_text_tokens_are_concatenated(self):
        conditioner = TaskTokenConditioner(
            ["segmentation", "depth", "normal"],
            cross_attention_dim=8,
            num_task_tokens=2,
            adapter_bottleneck_dim=4,
            text_input_dim=6,
        )
        text = torch.randn(2, 3, 6)
        states = conditioner(
            ["segmentation", "depth"], batch_size=2, text_hidden_states=text
        )

        self.assertEqual((2, 5, 8), tuple(states.shape))
        self.assertFalse(torch.allclose(states[0, :2], states[1, :2]))

    def test_unified_denoiser_passes_one_eight_channel_input(self):
        unet = _FakeUNet()
        expand_unet_conv_in(unet, 4, 4)
        conditioner = TaskTokenConditioner(
            ["segmentation", "depth", "normal"],
            cross_attention_dim=8,
            num_task_tokens=2,
            adapter_bottleneck_dim=4,
            text_input_dim=8,
        )
        model = UnifiedPerceptionDenoiser(unet, conditioner)

        output = model(
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 4, 8, 8),
            torch.tensor([5, 5]),
            "normal",
        )

        self.assertEqual((2, 8, 8, 8), tuple(unet.last_input.shape))
        self.assertEqual((2, 2, 8), tuple(unet.last_conditioning.shape))
        self.assertEqual((2, 4, 8, 8), tuple(output.sample.shape))

    def test_builder_uses_resolved_engineering_config(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        config["model"]["conditioning"].update(
            cross_attention_dim=8, text_input_dim=8, num_task_tokens=2
        )
        config["model"]["condition_adapter"]["bottleneck_dim"] = 4

        model, summary = build_unified_denoiser(_FakeUNet(), config)

        self.assertEqual(8, model.shared_unet.conv_in.in_channels)
        self.assertEqual("cross_attention", summary.scope)


class PreVAEAdapterTest(unittest.TestCase):
    def test_identity_adapter_preserves_codec_output(self):
        pixels = torch.rand(2, 3, 8, 8) * 2.0 - 1.0
        adapted = IdentityPreVAEAdapter()(pixels, "depth")
        self.assertIs(pixels, adapted)

    def test_residual_adapter_starts_as_bounded_identity(self):
        adapter = ResidualPreVAEAdapter(hidden_channels=4, num_blocks=1)
        pixels = torch.rand(2, 3, 8, 8) * 2.0 - 1.0

        adapted = adapter(pixels)
        self.assertTrue(torch.equal(pixels, adapted))

        adapted.square().mean().backward()
        self.assertIsNotNone(adapter.output.weight.grad)
        self.assertGreater(float(adapter.output.weight.grad.abs().sum()), 0.0)

    def test_residual_adapter_rejects_noncanonical_range(self):
        adapter = ResidualPreVAEAdapter(hidden_channels=4, num_blocks=1)
        with self.assertRaises(ValueError):
            adapter(torch.full((1, 3, 4, 4), 1.5))


if __name__ == "__main__":
    unittest.main()
