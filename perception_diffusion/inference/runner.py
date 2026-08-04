"""End-to-end RGB-to-task inference using a trained shared checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Subset

from perception_diffusion.codecs import SegmentationBinaryMaskCodec
from perception_diffusion.data import build_dataloader
from perception_diffusion.data.transforms import resize_labels
from perception_diffusion.models import load_pretrained_system
from perception_diffusion.task_specs import (
    build_segmentation_query_planner,
    build_task_specs,
)
from perception_diffusion.training import load_model_checkpoint, seed_everything
from perception_diffusion.visualization import save_prediction_panel

from .latent_sampler import UnifiedLatentSampler
from .segmentation_queries import merge_query_scores


ConditionMode = Literal["full", "task_only", "text_only", "unconditional"]


def condition_mode_switches(mode: ConditionMode) -> tuple[bool, bool]:
    try:
        return {
            "full": (True, True),
            "task_only": (True, False),
            "text_only": (False, True),
            "unconditional": (False, False),
        }[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported condition mode: {mode}") from exc


def _safe_sample_id(sample_id: str) -> str:
    value = sample_id.replace("\\", "__").replace("/", "__")
    if not value or value in {".", ".."}:
        raise ValueError(f"unsafe sample ID: {sample_id!r}")
    return value


def _prepare_output_root(path: str | Path, *, overwrite: bool) -> dict[str, Path]:
    root = Path(path).expanduser()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"inference output is not empty: {root}")
    roots = {
        "root": root,
        "predictions": root / "predictions",
        "targets": root / "targets",
        "valid_masks": root / "valid_masks",
        "visualizations": root / "visualizations",
    }
    for directory in roots.values():
        directory.mkdir(parents=True, exist_ok=True)
    return roots


@torch.no_grad()
def _sample_decoded_pixels(
    sampler: UnifiedLatentSampler,
    visual_pathway: Any,
    image_latent: torch.Tensor,
    task_name: str,
    text_hidden_states: torch.Tensor,
    *,
    num_steps: int,
    ensemble_size: int,
    generator: torch.Generator,
    use_task_condition: bool,
    use_text_condition: bool,
    initial_noises: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    decoded: list[torch.Tensor] = []
    if initial_noises is not None and len(initial_noises) != ensemble_size:
        raise ValueError("initial_noises must contain one tensor per ensemble member")
    for ensemble_index in range(ensemble_size):
        initial_noise = None
        if initial_noises is not None:
            initial_noise = initial_noises[ensemble_index]
            if initial_noise.shape[0] == 1 and image_latent.shape[0] != 1:
                initial_noise = initial_noise.repeat(image_latent.shape[0], 1, 1, 1)
            if initial_noise.shape != image_latent.shape:
                raise ValueError("initial noise and image latent shapes must match")
        target_latent = sampler.sample(
            image_latent,
            task_name,
            num_inference_steps=num_steps,
            text_hidden_states=text_hidden_states,
            generator=generator,
            initial_noise=initial_noise,
            use_task_condition=use_task_condition,
            use_text_condition=use_text_condition,
        )
        decoded.append(visual_pathway.decode_latents(target_latent))
    return torch.stack(decoded).mean(dim=0).clamp(-1.0, 1.0)


def run_inference(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    task_name: str,
    output_dir: str | Path,
    device_override: str | None = None,
    precision_override: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    num_steps_override: int | None = None,
    ensemble_size_override: int | None = None,
    condition_mode: ConditionMode = "full",
    seed_override: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    specs = build_task_specs(config)
    if task_name not in specs:
        raise ValueError(f"task {task_name!r} is not selected by the config")
    if config["model"]["target_adapter"]["enabled"]:
        raise ValueError(
            "inference requires deterministic codecs; learned pre-VAE target adaptation "
            "has no declared inverse and is an analysis-only ablation"
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    roots = _prepare_output_root(output_dir, overwrite=overwrite)
    seed = int(config["experiment"]["seed"] if seed_override is None else seed_override)
    seed_everything(seed)
    system = load_pretrained_system(
        config,
        device_override=device_override,
        precision_override=precision_override,
        for_training=False,
    )
    checkpoint_step = load_model_checkpoint(
        checkpoint,
        denoiser=system.denoiser,
        target_adapter=system.visual_pathway.target_adapter,
        device=system.device,
    )
    system.denoiser.eval()
    sampler = UnifiedLatentSampler(system.denoiser, system.inference_scheduler)
    data_config = (
        config["data"]["datasets"][task_name]
        if config["task"]["name"] == "multitask"
        else config["data"]
    )
    selected_split = split or str(data_config.get("validation_split", data_config["split"]))
    loader = build_dataloader(
        config,
        task_name,
        split=selected_split,
        training=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        strict_protocol=True,
        max_samples=limit,
    )
    num_steps = int(num_steps_override or config["inference"]["num_steps"])
    ensemble_size = int(
        ensemble_size_override or config["inference"].get("ensemble_size", 1)
    )
    if num_steps <= 0 or ensemble_size <= 0:
        raise ValueError("inference steps and ensemble size must be positive")
    use_task, use_text = condition_mode_switches(condition_mode)
    generator = torch.Generator(device=system.device).manual_seed(seed)
    spec = specs[task_name]
    sample_count = 0

    if task_name == "segmentation":
        if not isinstance(spec.codec, SegmentationBinaryMaskCodec):
            raise ValueError("closed-set query inference requires binary_query_mask codec")
        planner = build_segmentation_query_planner(spec)
        queries = planner.all_queries()
        query_batch_size = int(spec.query_config["query"]["batch_size"])  # type: ignore[index]
        native_dataset = loader.dataset
        if isinstance(native_dataset, Subset):
            native_dataset = native_dataset.dataset
        for raw_batch in loader:
            for sample_index, sample_id in enumerate(raw_batch["sample_id"]):
                image = raw_batch["image"][sample_index : sample_index + 1].to(system.device)
                valid = raw_batch["valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
                score_maps: list[np.ndarray] = []
                with torch.autocast(
                    device_type=system.device.type,
                    dtype=system.autocast_dtype,
                    enabled=system.autocast_enabled,
                ):
                    image_latent = system.visual_pathway.encode_images(image)
                    query_initial_noises = [
                        torch.randn(
                            (
                                1,
                                system.denoiser.target_latent_channels,
                                *image_latent.shape[-2:],
                            ),
                            device=system.device,
                            dtype=image_latent.dtype,
                            generator=generator,
                        )
                        for _ in range(ensemble_size)
                    ]
                    for start in range(0, len(queries), query_batch_size):
                        chunk = queries[start : start + query_batch_size]
                        prompts = [query.prompt for query in chunk]
                        text_hidden = system.encode_prompts(prompts)
                        repeated_image = image_latent.repeat(len(chunk), 1, 1, 1)
                        decoded = _sample_decoded_pixels(
                            sampler,
                            system.visual_pathway,
                            repeated_image,
                            task_name,
                            text_hidden,
                            num_steps=num_steps,
                            ensemble_size=ensemble_size,
                            generator=generator,
                            use_task_condition=use_task,
                            use_text_condition=use_text,
                            initial_noises=query_initial_noises,
                        )
                        score_maps.extend(
                            spec.codec.decode_scores(sample.cpu().numpy(), valid)
                            for sample in decoded
                        )
                prediction_512 = merge_query_scores(
                    np.stack(score_maps), queries, valid_mask=valid
                )
                native_gt = native_dataset.load_native_labels(
                    int(raw_batch["source_index"][sample_index])
                )
                native_h, native_w = native_gt.shape
                prediction = (
                    resize_labels(prediction_512, (native_h, native_w))
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                # Harmonize the prediction's ignore region to the native GT so
                # scored (GT-valid) pixels always carry an in-range 0..149 label.
                prediction[native_gt == 255] = 255
                native_valid = native_gt != 255
                safe_id = _safe_sample_id(str(sample_id))
                np.save(roots["predictions"] / f"{safe_id}.npy", prediction)
                np.save(roots["targets"] / f"{safe_id}.npy", native_gt)
                np.save(roots["valid_masks"] / f"{safe_id}.npy", native_valid)
                # The panel stays at the 512 decode resolution so it matches the
                # VAE image tensor; the saved .npy arrays are native resolution.
                save_prediction_panel(
                    task_name,
                    raw_batch["image"][sample_index],
                    prediction_512,
                    raw_batch["native_target"][sample_index, 0].cpu().numpy(),
                    roots["visualizations"] / f"{safe_id}.png",
                    valid_mask=valid,
                    num_classes=len(queries),
                )
                sample_count += 1
    else:
        for raw_batch in loader:
            image = raw_batch["image"].to(system.device)
            with torch.autocast(
                device_type=system.device.type,
                dtype=system.autocast_dtype,
                enabled=system.autocast_enabled,
            ):
                image_latent = system.visual_pathway.encode_images(image)
                text_hidden = system.encode_prompts(raw_batch["text_condition"])
                decoded = _sample_decoded_pixels(
                    sampler,
                    system.visual_pathway,
                    image_latent,
                    task_name,
                    text_hidden,
                    num_steps=num_steps,
                    ensemble_size=ensemble_size,
                    generator=generator,
                    use_task_condition=use_task,
                    use_text_condition=use_text,
                )
            for sample_index, sample_id in enumerate(raw_batch["sample_id"]):
                valid = raw_batch["valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
                prediction = spec.codec.decode(decoded[sample_index].cpu().numpy(), valid)
                target_chw = raw_batch["native_target"][sample_index].cpu().numpy()
                target = target_chw[0] if task_name == "depth" else target_chw
                safe_id = _safe_sample_id(str(sample_id))
                np.save(roots["predictions"] / f"{safe_id}.npy", prediction)
                np.save(roots["targets"] / f"{safe_id}.npy", target)
                np.save(roots["valid_masks"] / f"{safe_id}.npy", valid)
                save_prediction_panel(
                    task_name,
                    raw_batch["image"][sample_index],
                    prediction,
                    target,
                    roots["visualizations"] / f"{safe_id}.png",
                    valid_mask=valid,
                )
                sample_count += 1

    summary = {
        "status": "INFERENCE_COMPLETED",
        "formal_experiment": False,
        "task": task_name,
        "split": selected_split,
        "sample_count": sample_count,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_step": checkpoint_step,
        "model_path": str(system.model_path),
        "model_revision": system.model_revision,
        "num_steps": num_steps,
        "ensemble_size": ensemble_size,
        "condition_mode": condition_mode,
        "output_dir": str(roots["root"].resolve()),
    }
    (roots["root"] / "inference.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
