import unittest
from types import SimpleNamespace

import torch
from torch import nn

from perception_diffusion.inference import UnifiedLatentSampler
from perception_diffusion.inference.runner import condition_mode_switches
from perception_diffusion.models import TaskTokenConditioner, UnifiedPerceptionDenoiser
from perception_diffusion.training import (
    ConfiguredNoiseSampler,
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
    build_optimizer,
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


class _TypedScheduler(_Scheduler):
    """Scheduler exposing a prediction_type and a deterministic get_velocity."""

    def __init__(self, prediction_type: str) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_train_timesteps=10, prediction_type=prediction_type
        )

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps
        return noise - sample


class PredictionTypeTargetTest(unittest.TestCase):
    def _trainer(self, prediction_type: str) -> UnifiedDiffusionTrainerCore:
        conditioner = TaskTokenConditioner(
            ["segmentation", "depth", "normal"],
            cross_attention_dim=4,
            num_task_tokens=2,
            adapter_bottleneck_dim=2,
            text_input_dim=4,
        )
        denoiser = UnifiedPerceptionDenoiser(_DenoisingUNet(), conditioner)
        return UnifiedDiffusionTrainerCore(denoiser, _TypedScheduler(prediction_type))

    def _batch(self) -> UnifiedLatentBatch:
        return UnifiedLatentBatch(
            image_latent=torch.randn(1, 4, 4, 4),
            clean_target_latent=torch.randn(1, 4, 4, 4),
            task_name="depth",
            latent_valid_mask=torch.ones(1, 1, 4, 4),
        )

    def test_each_prediction_type_produces_finite_loss(self):
        for prediction_type in ("epsilon", "v_prediction", "sample"):
            trainer = self._trainer(prediction_type)
            self.assertEqual(prediction_type, trainer.prediction_type)
            output = trainer(
                self._batch(), generator=torch.Generator().manual_seed(0)
            )
            self.assertTrue(torch.isfinite(output.loss))
            self.assertEqual((1, 4, 4, 4), tuple(output.predicted_noise.shape))

    def test_prediction_types_yield_distinct_targets(self):
        # Same inputs, different objective => different loss (branch is live).
        batch = self._batch()
        noise = torch.randn(1, 4, 4, 4)
        timesteps = torch.tensor([3])
        losses = {}
        for prediction_type in ("epsilon", "v_prediction", "sample"):
            trainer = self._trainer(prediction_type)
            losses[prediction_type] = float(
                trainer(batch, noise=noise, timesteps=timesteps).loss.detach()
            )
        self.assertNotAlmostEqual(losses["epsilon"], losses["v_prediction"])
        self.assertNotAlmostEqual(losses["epsilon"], losses["sample"])

    def test_epsilon_default_needs_no_get_velocity(self):
        # The bare _Scheduler has no prediction_type and no get_velocity;
        # it must still train (defaults to epsilon) for backward compatibility.
        trainer, _ = _system()
        self.assertEqual("epsilon", trainer.prediction_type)
        self.assertFalse(hasattr(trainer.noise_scheduler, "get_velocity"))
        output = trainer(self._batch())
        self.assertTrue(torch.isfinite(output.loss))


class ConditionSwitchTest(unittest.TestCase):
    def test_condition_mode_switches_maps_each_mode(self):
        self.assertEqual((True, True), condition_mode_switches("full"))
        self.assertEqual((True, False), condition_mode_switches("task_only"))
        self.assertEqual((False, True), condition_mode_switches("text_only"))
        self.assertEqual((False, False), condition_mode_switches("unconditional"))
        with self.assertRaises(ValueError):
            condition_mode_switches("bogus")  # type: ignore[arg-type]

    def test_disabling_task_condition_changes_sampled_output(self):
        # _DenoisingUNet adds the condition mean to its output, so zeroing the
        # task tokens must shift the sampled latent end-to-end through sampler.
        _, sampler = _system()
        image = torch.randn(1, 4, 4, 4)
        initial_noise = torch.zeros(1, 4, 4, 4)
        full = sampler.sample(
            image, "depth", num_inference_steps=2, initial_noise=initial_noise
        )
        task_off = sampler.sample(
            image,
            "depth",
            num_inference_steps=2,
            initial_noise=initial_noise,
            use_task_condition=False,
        )
        self.assertFalse(torch.allclose(full, task_off))

    def test_unconditional_sampling_is_reproducible(self):
        # Both condition sources off => output is deterministic for a fixed
        # image and initial noise, independent of task name.
        _, sampler = _system()
        image = torch.randn(1, 4, 4, 4)
        initial_noise = torch.zeros(1, 4, 4, 4)
        kwargs = dict(
            num_inference_steps=2,
            initial_noise=initial_noise,
            use_task_condition=False,
            use_text_condition=False,
        )
        depth = sampler.sample(image, "depth", **kwargs)
        normal = sampler.sample(image, "normal", **kwargs)
        torch.testing.assert_close(depth, normal)


class OptimizerWeightDecayGroupingTest(unittest.TestCase):
    def _build(self) -> tuple[torch.optim.Optimizer, UnifiedPerceptionDenoiser]:
        conditioner = TaskTokenConditioner(
            ["segmentation", "depth", "normal"],
            cross_attention_dim=8,
            num_task_tokens=2,
            adapter_bottleneck_dim=4,
            text_input_dim=6,  # forces a real text_projection Linear (weight + bias)
        )
        denoiser = UnifiedPerceptionDenoiser(nn.Conv2d(8, 4, kernel_size=1), conditioner)
        training_config = {
            "optimizer": {
                "shared_unet_lr": 1.0e-5,
                "conditioner_lr": 1.0e-4,
                "adapter_lr": 1.0e-4,
                "weight_decay": 0.01,
            }
        }
        optimizer, _ = build_optimizer(denoiser, nn.Identity(), training_config)
        return optimizer, denoiser

    def _decay_class_by_id(
        self, optimizer: torch.optim.Optimizer
    ) -> dict[int, bool]:
        decays: dict[int, bool] = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                decays[id(parameter)] = group["weight_decay"] > 0
        return decays

    def test_biases_norms_scales_and_embeddings_are_exempt_from_decay(self):
        optimizer, denoiser = self._build()
        decays = self._decay_class_by_id(optimizer)
        no_decay_by_name = {
            "conditioner.task_embeddings.weight",  # embedding lookup, 2-D but exempt
            "conditioner.text_projection.bias",  # bias
            "shared_unet.bias",  # conv bias
        }
        decay_by_name = {
            "conditioner.text_projection.weight",  # Linear weight matrix
            "shared_unet.weight",  # conv weight
        }
        by_name = dict(denoiser.named_parameters())
        for name in no_decay_by_name:
            self.assertIn(name, by_name, name)
            self.assertFalse(decays[id(by_name[name])], f"{name} should be no-decay")
        for name in decay_by_name:
            self.assertIn(name, by_name, name)
            self.assertTrue(decays[id(by_name[name])], f"{name} should be decayed")
        # LayerNorm weight/bias and the residual_scale scalar inside every task
        # adapter must all be exempt.
        for name, parameter in denoiser.named_parameters():
            if name.startswith("conditioner.adapters.") and (
                ".norm." in name or name.endswith(".residual_scale")
            ):
                self.assertFalse(decays[id(parameter)], f"{name} should be no-decay")

    def test_every_no_decay_group_sets_zero_weight_decay(self):
        optimizer, _ = self._build()
        saw_no_decay = False
        for group in optimizer.param_groups:
            if group["group_name"].endswith("_no_decay"):
                saw_no_decay = True
                self.assertEqual(0.0, group["weight_decay"])
        self.assertTrue(saw_no_decay)


if __name__ == "__main__":
    unittest.main()
