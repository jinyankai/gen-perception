#!/usr/bin/env python3
"""Run one shared forward/backward/sample path for every configured task.

This is a structural CPU smoke test with tiny injected components. It does not
load pretrained weights and must not be reported as a model experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.inference import UnifiedLatentSampler  # noqa: E402
from perception_diffusion.models import (  # noqa: E402
    build_pre_vae_adapter,
    build_unified_denoiser,
)
from perception_diffusion.task_specs import build_task_specs  # noqa: E402
from perception_diffusion.training import (  # noqa: E402
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
)
from perception_diffusion.utils.config import configured_task_names, load_config  # noqa: E402


class _TinyConditionBlock(nn.Module):
    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        self.attn2 = nn.Linear(condition_dim, 4)


class _TinySharedUNet(nn.Module):
    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        self.config: dict[str, int] = {"in_channels": 4}
        self.conv_in = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.condition = _TinyConditionBlock(condition_dim)
        self.conv_out = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        *,
        encoder_hidden_states: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep
        condition = self.condition.attn2(encoder_hidden_states.mean(dim=1))
        condition = condition[:, :, None, None]
        hidden = torch.nn.functional.silu(self.conv_in(sample) + condition)
        return SimpleNamespace(sample=self.conv_out(hidden))


class _TinyScheduler:
    init_noise_sigma = 1.0

    def __init__(self, num_train_timesteps: int = 10) -> None:
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        self.timesteps: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._inference_steps = 1

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        scale = (timesteps.float() + 1.0) / float(self.config.num_train_timesteps)
        return clean + scale[:, None, None, None] * noise

    def set_timesteps(
        self, num_inference_steps: int, *, device: torch.device | None = None
    ) -> None:
        self._inference_steps = num_inference_steps
        self.timesteps = torch.arange(
            num_inference_steps - 1, -1, -1, device=device, dtype=torch.long
        )

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
        return SimpleNamespace(
            prev_sample=sample - model_output / float(self._inference_steps)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "multitask" / "stage1_shared_unet.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(1337)
    config = load_config(args.config, expand_environment=False)
    task_names = configured_task_names(config)
    task_specs = build_task_specs(config)

    smoke_config = dict(config)
    smoke_config["model"] = dict(config["model"])
    smoke_config["model"]["conditioning"] = dict(config["model"]["conditioning"])
    smoke_config["model"]["condition_adapter"] = dict(
        config["model"]["condition_adapter"]
    )
    smoke_config["model"]["conditioning"].update(
        cross_attention_dim=16,
        text_input_dim=16,
        num_task_tokens=2,
    )
    smoke_config["model"]["condition_adapter"]["bottleneck_dim"] = 8

    denoiser, trainability = build_unified_denoiser(
        _TinySharedUNet(condition_dim=16), smoke_config
    )
    target_adapter = build_pre_vae_adapter(config)
    scheduler = _TinyScheduler()
    trainer = UnifiedDiffusionTrainerCore(denoiser, scheduler)
    sampler = UnifiedLatentSampler(denoiser, scheduler)
    trainable_parameters = [
        parameter
        for module in (denoiser, target_adapter)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=1.0e-3)

    task_results: dict[str, dict[str, object]] = {}
    for task_name in task_names:
        optimizer.zero_grad(set_to_none=True)
        canonical_target = torch.rand(1, 3, 8, 8) * 2.0 - 1.0
        adapted_target = target_adapter(canonical_target, task_name)
        batch = UnifiedLatentBatch(
            image_latent=torch.randn(1, 4, 4, 4),
            clean_target_latent=torch.randn(1, 4, 4, 4),
            task_name=task_name,
            latent_valid_mask=torch.ones(1, 1, 4, 4),
        )
        step = trainer(batch)
        step.loss.backward()
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in trainable_parameters
                if parameter.grad is not None
            )
        )
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError(f"{task_name}: expected finite non-zero gradients")
        optimizer.step()

        sampled = sampler.sample(
            batch.image_latent,
            task_name,
            num_inference_steps=2,
            initial_noise=torch.zeros_like(batch.clean_target_latent),
        )
        task_results[task_name] = {
            "codec": type(task_specs[task_name].codec).__name__,
            "evaluator": type(task_specs[task_name].evaluator).__name__,
            "loss": float(step.loss.detach()),
            "gradient_norm": gradient_norm,
            "sample_shape": list(sampled.shape),
            "sample_finite": bool(torch.isfinite(sampled).all()),
            "pre_vae_shape": list(adapted_target.shape),
            "pre_vae_range": [
                float(adapted_target.detach().amin()),
                float(adapted_target.detach().amax()),
            ],
        }

    print(
        json.dumps(
            {
                "status": "FRAMEWORK_SMOKE_PASSED",
                "formal_experiment": False,
                "config": str(args.config),
                "tasks": list(task_names),
                "shared_unet_scope": trainability.scope,
                "task_results": task_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
