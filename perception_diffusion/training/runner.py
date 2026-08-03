"""Single-process real-data training runner shared by all stage-one tasks."""

from __future__ import annotations

import json
import random
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perception_diffusion.data import build_dataloader
from perception_diffusion.models import load_pretrained_system
from perception_diffusion.utils.config import configured_task_names, load_config
from perception_diffusion.utils.experiment import (
    ExperimentPaths,
    create_experiment_directory,
)

from .checkpoint import load_training_checkpoint, save_checkpoint
from .logging import TrainingLogger
from .noise import ConfiguredNoiseSampler
from .optim import build_lr_scheduler, build_optimizer, gradient_norm
from .unified_trainer import UnifiedDiffusionTrainerCore, UnifiedLatentBatch


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RoundRobinBatchStream:
    """Cycle task-homogeneous DataLoaders without creating task trainers."""

    def __init__(self, loaders: Mapping[str, Any]) -> None:
        if not loaders:
            raise ValueError("at least one task DataLoader is required")
        self.loaders = dict(loaders)
        self.task_names = tuple(loaders)
        self.iterators: dict[str, Iterator[dict[str, Any]]] = {
            name: iter(loader) for name, loader in self.loaders.items()
        }
        self.index = 0

    def __next__(self) -> tuple[str, dict[str, Any]]:
        task_name = self.task_names[self.index % len(self.task_names)]
        self.index += 1
        try:
            batch = next(self.iterators[task_name])
        except StopIteration:
            self.iterators[task_name] = iter(self.loaders[task_name])
            batch = next(self.iterators[task_name])
        if batch.get("task_name") != task_name:
            raise ValueError("DataLoader task name disagrees with round-robin route")
        return task_name, batch


def _make_scaler(device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:  # pragma: no cover - older supported torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


def _resume_paths(checkpoint: Path) -> ExperimentPaths:
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("resume checkpoint must live under <run>/checkpoints")
    root = checkpoint.parent.parent
    required = ("config.yaml", "command.txt", "git_commit.txt")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"resume run directory is incomplete: {missing}")
    return ExperimentPaths(
        root=root,
        checkpoints=root / "checkpoints",
        predictions=root / "predictions",
        visualizations=root / "visualizations",
        tensorboard=root / "tensorboard",
    )


def run_training(
    config: dict[str, Any],
    *,
    repo_root: str | Path,
    command: list[str] | None = None,
    device_override: str | None = None,
    precision_override: str | None = None,
    max_steps_override: int | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
    experiment_name_override: str | None = None,
    resume: str | Path | None = None,
    enable_tensorboard: bool | None = None,
    enable_wandb: bool | None = None,
) -> dict[str, Any]:
    """Execute a real optimizer loop and return an evidence summary."""

    training_config = config["training"]
    max_steps = int(max_steps_override or training_config["max_steps"])
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    resume_checkpoint: Path | None = None
    resume_paths: ExperimentPaths | None = None
    if resume is not None:
        resume_checkpoint = Path(resume).expanduser().resolve()
        resume_paths = _resume_paths(resume_checkpoint)
        saved_config = load_config(resume_paths.root / "config.yaml")
        if saved_config != config:
            raise ValueError(
                "resume config differs from the resolved config saved with the checkpoint"
            )
    system = load_pretrained_system(
        config,
        device_override=device_override,
        precision_override=precision_override,
        for_training=True,
    )
    device = system.device
    generator = torch.Generator(device=device).manual_seed(seed)
    task_names = configured_task_names(config)
    max_samples = training_config.get("max_samples")
    loaders = {
        task_name: build_dataloader(
            config,
            task_name,
            training=True,
            batch_size=batch_size_override,
            num_workers=num_workers_override,
            strict_protocol=True,
            max_samples=None if max_samples is None else int(max_samples),
        )
        for task_name in task_names
    }
    stream = RoundRobinBatchStream(loaders)
    noise_sampler = ConfiguredNoiseSampler.from_config(config)
    trainer = UnifiedDiffusionTrainerCore(
        system.denoiser, system.training_scheduler, noise_sampler
    )
    optimizer, trainable_parameters = build_optimizer(
        system.denoiser, system.visual_pathway.target_adapter, training_config
    )
    lr_scheduler = build_lr_scheduler(optimizer, training_config, max_steps=max_steps)
    scaler = _make_scaler(
        device, enabled=(device.type == "cuda" and system.autocast_dtype == torch.float16)
    )

    if resume is None:
        task_label = config["task"]["name"]
        experiment_name = experiment_name_override or str(config["experiment"]["name"])
        output_root = Path(str(config["experiment"]["output_root"])).expanduser()
        if "$" in str(output_root):
            raise ValueError("experiment.output_root contains an unresolved environment variable")
        paths = create_experiment_directory(
            output_root=output_root,
            task=task_label,
            experiment_name=experiment_name,
            config=config,
            command=command or sys.argv,
            repo_root=repo_root,
        )
        global_step = 0
    else:
        if resume_checkpoint is None or resume_paths is None:
            raise AssertionError("resume paths were not initialized")
        paths = resume_paths
        global_step = load_training_checkpoint(
            resume_checkpoint,
            denoiser=system.denoiser,
            target_adapter=system.visual_pathway.target_adapter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            generator=generator,
            device=device,
        )
        if global_step >= max_steps:
            raise ValueError(
                f"checkpoint step {global_step} is not below requested max_steps {max_steps}"
            )

    logger = TrainingLogger(
        paths.root,
        config,
        enable_tensorboard=enable_tensorboard,
        enable_wandb=enable_wandb,
    )
    log_every = int(training_config.get("log_every", 1))
    checkpoint_every = int(training_config.get("checkpoint_every", 500))
    gradient_clip = float(training_config.get("gradient_clip_norm", 1.0))
    if log_every <= 0 or checkpoint_every <= 0 or gradient_clip <= 0:
        raise ValueError("logging/checkpoint intervals and gradient clip must be positive")
    loss_history: list[float] = []
    started = time.perf_counter()
    try:
        while global_step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            last_task = ""
            step_started = time.perf_counter()
            for _ in range(accumulation_steps):
                task_name, raw_batch = next(stream)
                last_task = task_name
                images = _move_tensor(raw_batch["image"], device)
                targets = _move_tensor(raw_batch["target"], device)
                valid_mask = _move_tensor(raw_batch["valid_mask"], device)
                text_hidden = system.encode_prompts(raw_batch["text_condition"])
                with torch.autocast(
                    device_type=device.type,
                    dtype=system.autocast_dtype,
                    enabled=system.autocast_enabled,
                ):
                    pair = system.visual_pathway.encode_pair(
                        images,
                        targets,
                        valid_mask,
                        task_name,
                        sample_posterior=bool(
                            training_config.get("sample_vae_posterior", False)
                        ),
                        generator=generator,
                    )
                    latent_batch = UnifiedLatentBatch(
                        image_latent=pair.image_latent,
                        clean_target_latent=pair.target_latent,
                        task_name=task_name,
                        latent_valid_mask=pair.latent_valid_mask,
                        text_hidden_states=text_hidden,
                    )
                    output = trainer(latent_batch, generator=generator)
                    scaled_loss = output.loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(output.loss.detach()) / accumulation_steps

            scaler.unscale_(optimizer)
            unclipped_norm = gradient_norm(trainable_parameters)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            global_step += 1
            loss_history.append(accumulated_loss)
            if global_step % log_every == 0 or global_step == 1:
                metrics: dict[str, float | int | str] = {
                    "train/loss": accumulated_loss,
                    "train/gradient_norm": float(unclipped_norm),
                    "train/seconds_per_step": time.perf_counter() - step_started,
                    "train/task": last_task,
                }
                for group in optimizer.param_groups:
                    metrics[f"lr/{group.get('group_name', 'group')}"] = float(group["lr"])
                if device.type == "cuda":
                    metrics["runtime/max_memory_gib"] = torch.cuda.max_memory_allocated(
                        device
                    ) / (1024**3)
                logger.log(global_step, metrics)
            if global_step % checkpoint_every == 0 or global_step == max_steps:
                save_checkpoint(
                    paths.checkpoints / f"step-{global_step:08d}.pt",
                    step=global_step,
                    denoiser=system.denoiser,
                    target_adapter=system.visual_pathway.target_adapter,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    generator=generator,
                )
    finally:
        logger.close()

    summary = {
        "status": "TRAINING_COMPLETED",
        "formal_experiment": False,
        "steps": global_step,
        "tasks": list(task_names),
        "run_root": str(paths.root.resolve()),
        "last_checkpoint": str(
            (paths.checkpoints / f"step-{global_step:08d}.pt").resolve()
        ),
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "elapsed_seconds": time.perf_counter() - started,
        "model_path": str(system.model_path),
        "model_revision": system.model_revision,
        "noise": {
            "multi_scale": noise_sampler.enabled,
            "annealed": noise_sampler.annealed,
            "strength": noise_sampler.strength,
            "downscale_strategy": noise_sampler.downscale_strategy,
        },
    }
    (paths.root / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
