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


def _safe_sample_id(sample_id: str) -> str:
    value = sample_id.replace("\\", "__").replace("/", "__")
    return value or "sample"


@torch.no_grad()
def _run_validation(
    system: Any,
    sampler: Any,
    specs: Mapping[str, Any],
    loaders: Mapping[str, Any],
    trainer: UnifiedDiffusionTrainerCore,
    logger: TrainingLogger,
    *,
    step: int,
    num_steps: int,
    device: torch.device,
    adapter_enabled: bool,
    paths: ExperimentPaths,
) -> dict[str, float | int | str]:
    """Sample + score a fixed held-out set, saving panels and returning scalars.

    Wrapped in denoiser.eval(); a constant-seed generator keeps the validation
    loss comparable across steps so only the changing weights move the curve.
    """

    metrics: dict[str, float | int | str] = {}
    losses: list[float] = []
    step_dir = paths.visualizations / f"step-{step:08d}"
    was_training = system.denoiser.training
    system.denoiser.eval()
    try:
        for task_name, loader in loaders.items():
            spec = specs[task_name]
            task_losses: list[float] = []
            for raw_batch in loader:
                images = _move_tensor(raw_batch["image"], device)
                targets = _move_tensor(raw_batch["target"], device)
                valid_mask = _move_tensor(raw_batch["valid_mask"], device)
                text_hidden = system.encode_prompts(raw_batch["text_condition"])
                validation_generator = torch.Generator(device=device).manual_seed(0)
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
                        sample_posterior=False,
                        generator=validation_generator,
                    )
                    latent_batch = UnifiedLatentBatch(
                        image_latent=pair.image_latent,
                        clean_target_latent=pair.target_latent,
                        task_name=task_name,
                        latent_valid_mask=pair.latent_valid_mask,
                        text_hidden_states=text_hidden,
                    )
                    output = trainer(latent_batch, generator=validation_generator)
                task_losses.append(float(output.loss.detach()))
                if adapter_enabled:
                    continue
                _save_validation_panels(
                    system,
                    sampler,
                    spec,
                    raw_batch,
                    task_name,
                    text_hidden,
                    logger,
                    step=step,
                    num_steps=num_steps,
                    step_dir=step_dir,
                    generator=validation_generator,
                )
            if task_losses:
                task_mean = sum(task_losses) / len(task_losses)
                metrics[f"val/loss_{task_name}"] = task_mean
                losses.extend(task_losses)
        if losses:
            metrics["val/loss"] = sum(losses) / len(losses)
        if adapter_enabled:
            metrics["val/visualization"] = "skipped_target_adapter_enabled"
    finally:
        if was_training:
            system.denoiser.train()
    return metrics


def _save_validation_panels(
    system: Any,
    sampler: Any,
    spec: Any,
    raw_batch: Mapping[str, Any],
    task_name: str,
    text_hidden: torch.Tensor,
    logger: TrainingLogger,
    *,
    step: int,
    num_steps: int,
    step_dir: Path,
    generator: torch.Generator,
) -> None:
    """Sample, decode, and persist an [input|GT|prediction|error] panel per sample.

    A durable PNG lands under ``step_dir`` and the same panel is mirrored to the
    logger so TensorBoard/W&B show predictions converging on the fixed samples.
    """

    import numpy as np
    from PIL import Image

    from perception_diffusion.visualization import save_prediction_panel

    device = system.device
    images = _move_tensor(raw_batch["image"], device)
    with torch.autocast(
        device_type=device.type,
        dtype=system.autocast_dtype,
        enabled=system.autocast_enabled,
    ):
        image_latent = system.visual_pathway.encode_images(images)
        target_latent = sampler.sample(
            image_latent,
            task_name,
            num_inference_steps=num_steps,
            text_hidden_states=text_hidden,
            generator=generator,
            use_task_condition=True,
            use_text_condition=True,
        )
        decoded = system.visual_pathway.decode_latents(target_latent).clamp(-1.0, 1.0)
    for sample_index, sample_id in enumerate(raw_batch["sample_id"]):
        valid = raw_batch["valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
        prediction = spec.codec.decode(decoded[sample_index].cpu().numpy(), valid)
        target_chw = raw_batch["native_target"][sample_index].cpu().numpy()
        num_classes = 150
        if task_name == "segmentation":
            query_class_id = int(raw_batch["query_class_id"][sample_index])
            target = (target_chw[0] == query_class_id).astype(np.uint8)
            num_classes = 2
        elif task_name == "depth":
            target = target_chw[0]
        else:
            target = target_chw
        safe_id = _safe_sample_id(str(sample_id))
        panel_path = step_dir / f"{task_name}_{safe_id}.png"
        save_prediction_panel(
            task_name,
            raw_batch["image"][sample_index],
            prediction,
            target,
            panel_path,
            valid_mask=valid,
            num_classes=num_classes,
        )
        panel = np.asarray(Image.open(panel_path).convert("RGB"))
        logger.log_image(f"val/{task_name}/{safe_id}", panel, step)


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

    # In-training periodic validation (opt-in via training.validation.every > 0).
    # The sampler and task specs live under perception_diffusion.inference, whose
    # runner imports back into training; importing here (not at module top) keeps
    # the load order acyclic.
    validation_config = training_config.get("validation") or {}
    validation_every = int(validation_config.get("every", 0))
    validation_samples = int(validation_config.get("num_samples", 2))
    # A learned pre-VAE target adapter has no codec inverse (see inference guard),
    # so viz is skipped when it is on; the latent-space val-loss still computes.
    adapter_enabled = bool(config["model"]["target_adapter"]["enabled"])
    validation_loaders: dict[str, Any] = {}
    sampler = None
    specs = None
    if validation_every > 0:
        from perception_diffusion.inference import UnifiedLatentSampler
        from perception_diffusion.task_specs import build_task_specs

        specs = build_task_specs(config)
        sampler = UnifiedLatentSampler(system.denoiser, system.inference_scheduler)
        is_multitask = config["task"]["name"] == "multitask"
        for task_name in task_names:
            task_data = (
                config["data"]["datasets"][task_name] if is_multitask else config["data"]
            )
            validation_split = str(
                task_data.get("validation_split", task_data.get("split"))
            )
            validation_loaders[task_name] = build_dataloader(
                config,
                task_name,
                split=validation_split,
                training=False,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                strict_protocol=True,
                max_samples=validation_samples,
            )
    validation_steps = int(config["inference"]["num_steps"])
    loss_history: list[float] = []
    skipped_steps = 0
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
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                # AMP found non-finite gradients and skipped optimizer.step().
                # A skipped update must not advance the LR schedule or the step
                # counter, and its (inf/NaN) gradient norm must not be recorded
                # as though real optimization happened.
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
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
            if validation_every > 0 and (
                global_step % validation_every == 0 or global_step == max_steps
            ):
                assert sampler is not None and specs is not None
                validation_metrics = _run_validation(
                    system,
                    sampler,
                    specs,
                    validation_loaders,
                    trainer,
                    logger,
                    step=global_step,
                    num_steps=validation_steps,
                    device=device,
                    adapter_enabled=adapter_enabled,
                    paths=paths,
                )
                logger.log(global_step, validation_metrics)
    finally:
        logger.close()

    summary = {
        "status": "TRAINING_COMPLETED",
        "formal_experiment": False,
        "steps": global_step,
        "amp_skipped_steps": skipped_steps,
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
