#!/usr/bin/env python3
"""Run a bounded real-component ADE20K segmentation training gate.

This operator check loads one real ADE20K image and a class-query binary mask,
encodes both with the frozen SD2 VAE, and runs a fixed-noise diffusion loss
through the project's shared U-Net. It verifies finite gradients and a real
optimizer update. It is a smoke/overfit gate, not a formal benchmark run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_diffusion.codecs import SegmentationBinaryMaskCodec  # noqa: E402
from perception_diffusion.data import ADE20KDataset  # noqa: E402
from perception_diffusion.models import build_unified_denoiser  # noqa: E402
from perception_diffusion.training import (  # noqa: E402
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
)
from perception_diffusion.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_CACHE", "models")) / "stable-diffusion-2",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data")),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "segmentation" / "ade20k.yaml",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square gate resolution; defaults to the first configured image dimension.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help="Defaults to training.mixed_precision in the resolved config.",
    )
    parser.add_argument("--class-id", type=int, default=None, help="Raw ADE20K ID, 1-150.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _segmentation_sections(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config["task"]["name"] == "multitask":
        return config["data"]["datasets"]["segmentation"], config["evaluation"][
            "segmentation"
        ]
    if config["task"]["name"] != "segmentation":
        raise ValueError("training gate requires a segmentation or multitask config")
    return config["data"], config["evaluation"]


def _precision_dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        precision
    ]


def _optimizer_groups(
    denoiser: torch.nn.Module, training: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter]]:
    optimizer_config = training["optimizer"]
    unet_parameters = [
        parameter for parameter in denoiser.shared_unet.parameters() if parameter.requires_grad
    ]
    adapter_parameters: list[torch.nn.Parameter] = []
    conditioner_parameters: list[torch.nn.Parameter] = []
    for name, parameter in denoiser.conditioner.named_parameters():
        if not parameter.requires_grad:
            continue
        target = adapter_parameters if name.startswith("adapters.") else conditioner_parameters
        target.append(parameter)
    groups = [
        {
            "params": unet_parameters,
            "lr": float(optimizer_config["shared_unet_lr"]),
            "group_name": "shared_unet",
        },
        {
            "params": conditioner_parameters,
            "lr": float(optimizer_config["conditioner_lr"]),
            "group_name": "conditioner",
        },
        {
            "params": adapter_parameters,
            "lr": float(optimizer_config["adapter_lr"]),
            "group_name": "condition_adapter",
        },
    ]
    nonempty = [group for group in groups if group["params"]]
    all_parameters = [parameter for group in nonempty for parameter in group["params"]]
    return nonempty, all_parameters


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        raise ValueError("no trainable parameter received a gradient")
    return torch.stack(squared).sum().sqrt()


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    if args.steps <= 0 or args.steps > 500:
        raise ValueError("--steps must be in 1..500 for this bounded gate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    model_dir = args.model_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    os.environ["MODEL_CACHE"] = str(model_dir.parent)
    os.environ["DATA_ROOT"] = str(data_root)
    os.environ.setdefault("OUTPUT_ROOT", str(data_root.parent / "outputs"))
    config = load_config(args.config.expanduser().resolve())
    data_config, evaluation_config = _segmentation_sections(config)
    if Path(config["model"]["backbone"]["pretrained_model_path"]) != model_dir:
        raise ValueError("resolved config model path does not match --model-dir")
    ade_root = data_root / "ADEChallengeData2016"
    if Path(data_config["root"]) != ade_root:
        raise ValueError("resolved config ADE20K path does not match --data-root")

    configured_size = tuple(int(value) for value in data_config["image_size"])
    image_size = (
        configured_size
        if args.image_size is None
        else (int(args.image_size), int(args.image_size))
    )
    if min(image_size) <= 0 or any(value % 8 for value in image_size):
        raise ValueError("image dimensions must be positive multiples of 8")
    precision = args.precision or str(config["training"]["mixed_precision"])
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    if args.device == "cpu" and precision != "fp32":
        raise ValueError("CPU gate requires --precision fp32")
    if (
        precision == "bf16"
        and args.device == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("current CUDA device does not support bf16")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    query_config = evaluation_config["query"]
    codec_config = data_config.get("codec", config["task"].get("codec"))
    if not isinstance(codec_config, dict) or codec_config.get("name") != "binary_query_mask":
        raise ValueError("segmentation training gate requires the binary_query_mask codec")
    query_sampling = "first_present" if args.class_id is None else "fixed"
    fixed_query_class_id = None
    if args.class_id is not None:
        if not 1 <= args.class_id <= 150:
            raise ValueError("--class-id is a raw ADE20K ID in 1..150")
        fixed_query_class_id = args.class_id - 1
    dataset = ADE20KDataset(
        ade_root,
        split=str(data_config["split"]),
        image_size=image_size,
        codec=SegmentationBinaryMaskCodec(threshold=float(codec_config["threshold"])),
        horizontal_flip_probability=0.0,
        query_sampling=query_sampling,
        fixed_query_class_id=fixed_query_class_id,
        prompt_template=str(query_config["prompt_template"]),
        strict_protocol=True,
    )
    sample = dataset[0]
    query_class_id = int(sample["query_class_id"])
    positive_pixels = int((sample["native_target"] == query_class_id).sum())
    if positive_pixels <= 0:
        raise ValueError(f"query class {query_class_id} is absent from gate sample")
    sample_metadata = {
        "sample_id": sample["sample_id"],
        "source_index": sample["source_index"],
        "original_size": sample["original_size"].tolist(),
        "image_size": list(image_size),
        "raw_class_id": query_class_id + 1,
        "model_class_id": query_class_id,
        "class_name": dataset.class_names[query_class_id],
        "prompt": sample["text_condition"],
        "positive_pixels": positive_pixels,
        "valid_pixels": int(sample["valid_mask"].sum()),
        "image_range": [float(sample["image"].min()), float(sample["image"].max())],
        "target_range": [float(sample["target"].min()), float(sample["target"].max())],
    }
    image = sample["image"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    valid_mask = sample["valid_mask"].unsqueeze(0).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_dir / "tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_dir / "text_encoder", local_files_only=True, use_safetensors=True
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_dir / "vae", local_files_only=True, use_safetensors=True
    ).to(device)
    scheduler = DDIMScheduler.from_pretrained(
        model_dir / "scheduler", local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_dir / "unet", local_files_only=True, use_safetensors=True
    ).to(device)
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()

    denoiser, trainability = build_unified_denoiser(unet, config)
    denoiser.to(device).train()
    if args.gradient_checkpointing:
        denoiser.shared_unet.enable_gradient_checkpointing()
    trainer = UnifiedDiffusionTrainerCore(denoiser, scheduler)

    tokens = tokenizer(
        sample_metadata["prompt"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    dtype = _precision_dtype(precision)
    autocast_enabled = precision != "fp32"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=autocast_enabled
    ):
        text_hidden = text_encoder(**tokens).last_hidden_state
        scale = float(vae.config.scaling_factor)
        image_latent = vae.encode(image).latent_dist.mode() * scale
        target_latent = vae.encode(target).latent_dist.mode() * scale
    del image, target, text_encoder, vae, tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()

    batch = UnifiedLatentBatch(
        image_latent=image_latent,
        clean_target_latent=target_latent,
        task_name="segmentation",
        latent_valid_mask=valid_mask,
        text_hidden_states=text_hidden,
    )
    groups, trainable_parameters = _optimizer_groups(denoiser, config["training"])
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler(
        device.type, enabled=(device.type == "cuda" and precision == "fp16")
    )
    reference_name, reference_parameter = next(
        (name, parameter)
        for name, parameter in denoiser.named_parameters()
        if parameter.requires_grad
    )
    reference_before = reference_parameter.detach().float().clone()

    generator = torch.Generator(device=device).manual_seed(seed)
    fixed_noise = torch.randn(
        target_latent.shape,
        generator=generator,
        device=device,
        dtype=target_latent.dtype,
    )
    timestep_count = int(scheduler.config.num_train_timesteps)
    fixed_timesteps = torch.randint(
        0,
        timestep_count,
        (target_latent.shape[0],),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_records: list[dict[str, Any]] = []
    for step_index in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=autocast_enabled
        ):
            output = trainer(
                batch,
                noise=fixed_noise,
                timesteps=fixed_timesteps,
            )
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = _gradient_norm(trainable_parameters)
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
            raise FloatingPointError(f"invalid gradient norm: {float(grad_norm)}")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_records.append(
            {
                "step": step_index + 1,
                "loss": float(output.loss.detach()),
                "gradient_norm": float(grad_norm.detach()),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )

    update_l2 = float(
        (reference_parameter.detach().float() - reference_before).square().sum().sqrt()
    )
    if not np.isfinite(update_l2) or update_l2 <= 0:
        raise FloatingPointError(f"trainable parameter did not update: {update_l2}")
    losses = [record["loss"] for record in step_records]
    result: dict[str, Any] = {
        "status": "SEGMENTATION_TRAINING_GATE_PASSED",
        "formal_experiment": False,
        "network_used": False,
        "checkpoint_written": False,
        "config": str(args.config.expanduser().resolve()),
        "model": str(model_dir),
        "device": str(device),
        "visible_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "precision": precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": seed,
        "steps": args.steps,
        "sample": sample_metadata,
        "image_latent_shape": list(image_latent.shape),
        "target_latent_shape": list(target_latent.shape),
        "timestep": int(fixed_timesteps.item()),
        "trainable_scope": trainability.scope,
        "trainable_unet_parameters": trainability.trainable_parameters,
        "optimizer_groups": [
            {
                "name": group["group_name"],
                "learning_rate": group["lr"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in groups
        ],
        "reference_parameter": reference_name,
        "reference_parameter_update_l2": update_l2,
        "steps_detail": step_records,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_change": losses[-1] - losses[0],
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**20, 2
        )
    return result


def main() -> int:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        print(json.dumps(run_gate(args), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "SEGMENTATION_TRAINING_GATE_FAILED",
                    "formal_experiment": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
