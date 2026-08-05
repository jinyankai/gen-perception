import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy
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
    resolve_distributed_context,
    run_training,
    save_checkpoint,
    single_process_context,
    wrap_ddp,
)
from perception_diffusion.training.distributed import barrier, destroy, reduce_mean
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

    def test_log_image_forwards_hwc_panel_to_writer(self):
        config = {
            "experiment": {"name": "unit"},
            "training": {
                "logging": {
                    "tensorboard": False,
                    "wandb": {"enabled": False, "mode": "offline"},
                }
            },
        }
        calls: list[dict[str, object]] = []

        class _FakeWriter:
            def add_image(self, tag, image, step, dataformats):
                calls.append(
                    {
                        "tag": tag,
                        "shape": image.shape,
                        "step": step,
                        "dataformats": dataformats,
                    }
                )

            def flush(self):
                pass

            def close(self):
                pass

        panel = numpy.zeros((4, 12, 3), dtype=numpy.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            logger = TrainingLogger(temporary, config, enable_tensorboard=False)
            logger.writer = _FakeWriter()
            logger.log_image("val/depth/sample", panel, 5)
            logger.close()
        self.assertEqual(1, len(calls))
        self.assertEqual("val/depth/sample", calls[0]["tag"])
        self.assertEqual((4, 12, 3), calls[0]["shape"])
        self.assertEqual(5, calls[0]["step"])
        self.assertEqual("HWC", calls[0]["dataformats"])


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

    def test_amp_skipped_step_does_not_advance_schedule_or_counter(self):
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

        class _SkipOnceScaler:
            """Report one AMP backoff (scale drop) before behaving normally."""

            def __init__(self) -> None:
                self._scales = [2.0, 1.0, 1.0, 1.0]
                self._index = 0

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                pass

            def update(self):
                self._index = min(self._index + 1, len(self._scales) - 1)

            def get_scale(self):
                return self._scales[self._index]

            def state_dict(self):
                return {}

            def load_state_dict(self, state):
                pass

        steps_seen: list[int] = []
        real_scheduler_step = torch.optim.lr_scheduler.LambdaLR.step

        def _counting_step(self, *args, **kwargs):
            # LambdaLR.__init__ invokes step() once to initialize; count only the
            # explicit calls made inside the training loop (epoch advances past 0).
            result = real_scheduler_step(self, *args, **kwargs)
            if self.last_epoch >= 1:
                steps_seen.append(self.last_epoch)
            return result

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
                patch(
                    "perception_diffusion.training.runner._make_scaler",
                    return_value=_SkipOnceScaler(),
                ),
                patch.object(
                    torch.optim.lr_scheduler.LambdaLR, "step", _counting_step
                ),
            ):
                summary = run_training(
                    config,
                    repo_root=ROOT,
                    command=["test"],
                    enable_tensorboard=False,
                )
        # The first optimizer step is skipped, so exactly one real step lands and
        # the schedule advances exactly once despite two loop iterations.
        self.assertEqual(1, summary["steps"])
        self.assertEqual(1, summary["amp_skipped_steps"])
        self.assertEqual(1, len(steps_seen))


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


class DistributedRuntimeTest(unittest.TestCase):
    def test_disabled_mode_returns_single_process_context(self):
        ctx = resolve_distributed_context({"runtime": {}}, device_type="cpu")
        self.assertFalse(ctx.enabled)
        self.assertEqual(0, ctx.rank)
        self.assertEqual(1, ctx.world_size)
        self.assertTrue(ctx.is_main)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_distributed_context(
                {"runtime": {"distributed": "horovod"}}, device_type="cpu"
            )

    def test_ddp_requires_cuda_device(self):
        with self.assertRaises(RuntimeError):
            resolve_distributed_context(
                {"runtime": {"distributed": "ddp"}}, device_type="cpu"
            )

    def test_ddp_without_launcher_env_is_rejected(self):
        # No RANK/WORLD_SIZE/LOCAL_RANK (torchrun injects them), so this must
        # raise before touching the process group even when CUDA is claimed.
        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(RuntimeError):
                    resolve_distributed_context(
                        {"runtime": {"distributed": "ddp"}}, device_type="cuda"
                    )

    def test_helpers_are_identity_when_disabled(self):
        ctx = single_process_context()
        module = nn.Linear(2, 2)
        self.assertIs(module, wrap_ddp(module, ctx))
        self.assertIsNone(barrier(ctx))
        self.assertIsNone(destroy(ctx))
        self.assertEqual(
            2.5, reduce_mean(2.5, ctx, device=torch.device("cpu"))
        )

    def test_runner_rejects_ddp_with_target_adapter_enabled(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        config["runtime"]["distributed"] = "ddp"
        config["model"]["target_adapter"]["enabled"] = True
        # An enabled context can only arise under torchrun+CUDA, so stub it to hit
        # the guard on CPU. The guard fires before load_pretrained_system, proving
        # the incompatible combination is refused rather than silently mistrained.
        enabled_ctx = SimpleNamespace(
            enabled=True, rank=0, local_rank=0, world_size=2, is_main=True
        )
        with patch(
            "perception_diffusion.training.runner.resolve_distributed_context",
            return_value=enabled_ctx,
        ):
            with self.assertRaises(ValueError) as raised:
                run_training(config, repo_root=ROOT, command=["test"])
        self.assertIn("target_adapter", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
