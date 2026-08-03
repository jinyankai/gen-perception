import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from perception_diffusion.analysis import (
    aggregate_reconstruction_records,
    reconstruction_metrics,
)
from perception_diffusion.codecs import DepthCodec, SegmentationBinaryMaskCodec
from perception_diffusion.inference.runner import condition_mode_switches
from perception_diffusion.models import TaskTokenConditioner, UnifiedPerceptionDenoiser
from perception_diffusion.training import (
    TrainingLogger,
    load_training_checkpoint,
    run_training,
    save_checkpoint,
)
from perception_diffusion.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class _Scaler:
    def __init__(self) -> None:
        self.value = 1

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(8, 4, kernel_size=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep
        condition = encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
        return SimpleNamespace(sample=self.conv_in(sample) + condition)


class _TinyNoiseScheduler:
    config = SimpleNamespace(num_train_timesteps=10)

    @staticmethod
    def add_noise(
        clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        del timesteps
        return clean + noise


class _TinyVisualPathway(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_adapter = nn.Identity()

    def encode_pair(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        task_name: str,
        **_: object,
    ) -> SimpleNamespace:
        del task_name
        image_latent = torch.cat([images, images[:, :1]], dim=1)
        target_latent = torch.cat([targets, targets[:, :1]], dim=1)
        return SimpleNamespace(
            image_latent=image_latent,
            target_latent=target_latent,
            latent_valid_mask=valid_mask,
        )


class _TinySystem:
    def __init__(self) -> None:
        conditioner = TaskTokenConditioner(
            ["segmentation", "depth", "normal"],
            cross_attention_dim=4,
            num_task_tokens=2,
            adapter_bottleneck_dim=2,
            text_input_dim=4,
        )
        self.denoiser = UnifiedPerceptionDenoiser(_TinyUNet(), conditioner)
        self.visual_pathway = _TinyVisualPathway()
        self.training_scheduler = _TinyNoiseScheduler()
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.autocast_enabled = False
        self.model_path = Path("tiny-sd2")
        self.model_revision = "0" * 40

    @staticmethod
    def encode_prompts(prompts: list[str]) -> torch.Tensor:
        return torch.zeros(len(prompts), 1, 4)


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_restores_model_optimizer_scheduler_and_step(self):
        denoiser = nn.Linear(2, 2)
        adapter = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW([*denoiser.parameters(), *adapter.parameters()])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = _Scaler()
        generator = torch.Generator().manual_seed(9)
        before = denoiser.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(
                Path(temporary) / "step.pt",
                step=4,
                denoiser=denoiser,
                target_adapter=adapter,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                scaler=scaler,
                generator=generator,
            )
            with torch.no_grad():
                denoiser.weight.add_(10.0)
            step = load_training_checkpoint(
                path,
                denoiser=denoiser,
                target_adapter=adapter,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                device=torch.device("cpu"),
            )
        self.assertEqual(4, step)
        self.assertTrue(torch.equal(before, denoiser.weight))


class LoggingTest(unittest.TestCase):
    def test_offline_logger_writes_jsonl_without_optional_backends(self):
        config = {
            "experiment": {"name": "unit"},
            "training": {
                "logging": {
                    "tensorboard": False,
                    "wandb": {"enabled": False, "mode": "offline"},
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            logger = TrainingLogger(temporary, config)
            logger.log(3, {"train/loss": 1.25, "train/task": "depth"})
            logger.close()
            text = (Path(temporary) / "metrics.jsonl").read_text(encoding="utf-8")
        self.assertIn('"step": 3', text)
        self.assertIn('"train/loss": 1.25', text)


class TrainingRunnerTest(unittest.TestCase):
    def test_runner_executes_optimizer_log_and_checkpoint_control_flow(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        config["training"]["max_steps"] = 1
        config["training"]["checkpoint_every"] = 1
        config["training"]["log_every"] = 1
        config["training"]["gradient_checkpointing"] = False
        config["training"]["logging"]["tensorboard"] = False
        batch = {
            "image": torch.zeros(1, 3, 8, 8),
            "target": torch.zeros(1, 3, 8, 8),
            "valid_mask": torch.ones(1, 1, 8, 8, dtype=torch.bool),
            "task_name": "segmentation",
            "text_condition": ["semantic segmentation"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            config["experiment"]["output_root"] = temporary
            with (
                patch(
                    "perception_diffusion.training.runner.load_pretrained_system",
                    return_value=_TinySystem(),
                ),
                patch(
                    "perception_diffusion.training.runner.build_dataloader",
                    return_value=[batch],
                ),
            ):
                summary = run_training(
                    config,
                    repo_root=ROOT,
                    command=["test"],
                    enable_tensorboard=False,
                )
            run_root = Path(summary["run_root"])
            checkpoint = Path(summary["last_checkpoint"])
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((run_root / "metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual("TRAINING_COMPLETED", summary["status"])
            self.assertEqual(1, summary["steps"])


class ReconstructionAnalysisTest(unittest.TestCase):
    def test_perfect_binary_and_depth_reconstructions_have_perfect_metrics(self):
        binary_codec = SegmentationBinaryMaskCodec()
        mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8).numpy()
        encoded_mask = binary_codec.encode(mask)
        segmentation = reconstruction_metrics(
            "segmentation",
            encoded_target=encoded_mask.values,
            reconstructed_pixels=encoded_mask.values,
            native_target=mask[None],
            valid_mask=encoded_mask.valid_mask,
            codec=binary_codec,
            query_class_id=-1,
        )
        depth_codec = DepthCodec(0.1, 10.0)
        depth = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
        encoded_depth = depth_codec.encode(depth)
        depth_metrics = reconstruction_metrics(
            "depth",
            encoded_target=encoded_depth.values,
            reconstructed_pixels=encoded_depth.values,
            native_target=depth[None],
            valid_mask=encoded_depth.valid_mask,
            codec=depth_codec,
        )
        self.assertEqual(1.0, segmentation["binary_iou"])
        self.assertAlmostEqual(0.0, depth_metrics["abs_rel"], places=6)
        summary = aggregate_reconstruction_records(
            [
                {"task": "segmentation", "metrics": segmentation},
                {"task": "depth", "metrics": depth_metrics},
            ]
        )
        self.assertEqual(2, len(summary["per_task"]))

    def test_condition_modes_are_explicit(self):
        self.assertEqual((True, True), condition_mode_switches("full"))
        self.assertEqual((True, False), condition_mode_switches("task_only"))
        self.assertEqual((False, True), condition_mode_switches("text_only"))
        self.assertEqual((False, False), condition_mode_switches("unconditional"))


if __name__ == "__main__":
    unittest.main()
