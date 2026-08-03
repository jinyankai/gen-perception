#!/usr/bin/env python3
"""Run a bounded real ADE20K/SD2 DistributedDataParallel training gate.

Every rank loads a distinct ADE20K sample, encodes image and binary-query target
with the frozen SD2 components, performs the shared diffusion loss through DDP,
and applies an optimizer update. The gate verifies NCCL all-reduce, finite
gradients, non-zero updates, and post-step replica consistency. It writes no
checkpoint and is not a formal training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_diffusion.codecs import SegmentationBinaryMaskCodec  # noqa: E402
from perception_diffusion.data import ADE20KDataset, collate_task_samples  # noqa: E402
from perception_diffusion.models import build_unified_denoiser  # noqa: E402
from perception_diffusion.training import (  # noqa: E402
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
)
from perception_diffusion.utils.config import load_config  # noqa: E402
from scripts.operator.segmentation_training_gate import (  # noqa: E402
    _gradient_norm,
    _optimizer_groups,
    _precision_dtype,
    _segmentation_sections,
)


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
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--process-group-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def _event(rank: int, phase: str, **values: object) -> None:
    print(
        json.dumps(
            {"event": phase, "rank": rank, **values},
            ensure_ascii=False,
        ),
        flush=True,
    )


def validate_replica_measurements(
    updates: list[float], checksums: list[float], squared_checksums: list[float]
) -> dict[str, float]:
    """Validate post-step measurements gathered from every DDP replica."""

    if not updates or len(updates) != len(checksums) or len(updates) != len(
        squared_checksums
    ):
        raise ValueError("replica measurement lists must be non-empty and equal length")
    values = updates + checksums + squared_checksums
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("replica measurements contain non-finite values")
    if min(updates) <= 0:
        raise FloatingPointError(f"at least one replica did not update: {updates}")
    if not all(
        math.isclose(value, updates[0], rel_tol=1.0e-5, abs_tol=1.0e-7)
        for value in updates[1:]
    ):
        raise FloatingPointError(f"replica update norms diverged: {updates}")
    if not all(
        math.isclose(value, checksums[0], rel_tol=1.0e-5, abs_tol=1.0e-5)
        for value in checksums[1:]
    ):
        raise FloatingPointError(f"replica parameter checksums diverged: {checksums}")
    if not all(
        math.isclose(value, squared_checksums[0], rel_tol=1.0e-5, abs_tol=1.0e-5)
        for value in squared_checksums[1:]
    ):
        raise FloatingPointError(
            f"replica squared parameter checksums diverged: {squared_checksums}"
        )
    return {
        "update_min": min(updates),
        "update_max": max(updates),
        "checksum_spread": max(checksums) - min(checksums),
        "squared_checksum_spread": max(squared_checksums)
        - min(squared_checksums),
    }


def _all_gather_vector(vector: torch.Tensor, world_size: int) -> list[list[float]]:
    gathered = [torch.zeros_like(vector) for _ in range(world_size)]
    dist.all_gather(gathered, vector)
    return [item.detach().cpu().double().tolist() for item in gathered]


def _sample_metadata(
    batch: dict[str, Any], dataset: ADE20KDataset
) -> dict[str, Any]:
    query_class_id = int(batch["query_class_id"].item())
    positive_pixels = int((batch["native_target"] == query_class_id).sum())
    if positive_pixels <= 0:
        raise ValueError(f"query class {query_class_id} is absent from rank sample")
    return {
        "sample_id": batch["sample_id"][0],
        "source_index": int(batch["source_index"].item()),
        "query_class_id": query_class_id,
        "class_name": dataset.class_names[query_class_id],
        "prompt": batch["text_condition"][0],
        "positive_pixels": positive_pixels,
        "valid_pixels": int(batch["valid_mask"].sum()),
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = (
        int(os.environ.get("LOCAL_RANK", "0"))
        if args.local_rank is None
        else args.local_rank
    )
    if world_size < 2:
        raise ValueError("DDP training gate requires at least two ranks")
    if args.steps <= 0 or args.steps > 10:
        raise ValueError("--steps must be in 1..10 for this bounded DDP gate")
    if args.image_size <= 0 or args.image_size % 8:
        raise ValueError("--image-size must be a positive multiple of 8")
    if args.process_group_timeout_seconds <= 0:
        raise ValueError("process-group timeout must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"visible CUDA devices {torch.cuda.device_count()} < world size {world_size}"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("current CUDA device does not support bf16")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.process_group_timeout_seconds),
    )
    _event(rank, "process_group_initialized", world_size=world_size, device=local_rank)

    probe = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(probe)
    expected_probe = world_size * (world_size + 1) / 2
    if probe.item() != expected_probe:
        raise ValueError(f"NCCL probe {probe.item()} != expected {expected_probe}")
    _event(rank, "nccl_all_reduce_passed", value=probe.item())

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

    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    codec_config = data_config.get("codec", config["task"].get("codec"))
    if not isinstance(codec_config, dict) or codec_config.get("name") != "binary_query_mask":
        raise ValueError("DDP segmentation gate requires binary_query_mask")
    dataset = ADE20KDataset(
        ade_root,
        split=str(data_config["split"]),
        image_size=(args.image_size, args.image_size),
        codec=SegmentationBinaryMaskCodec(threshold=float(codec_config["threshold"])),
        horizontal_flip_probability=0.0,
        query_sampling="first_present",
        prompt_template=str(evaluation_config["query"]["prompt_template"]),
        strict_protocol=True,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_task_samples,
        pin_memory=True,
        drop_last=False,
    )
    raw_batch = next(iter(loader))
    sample_info = _sample_metadata(raw_batch, dataset)
    _event(rank, "sample_loaded", **sample_info)

    image = raw_batch["image"].to(device, non_blocking=True)
    target = raw_batch["target"].to(device, non_blocking=True)
    valid_mask = raw_batch["valid_mask"].to(device, non_blocking=True)
    del raw_batch
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
    _event(rank, "components_loaded")

    denoiser, trainability = build_unified_denoiser(unet, config)
    denoiser.to(device).train()
    if args.gradient_checkpointing:
        denoiser.shared_unet.enable_gradient_checkpointing()
    trainer = UnifiedDiffusionTrainerCore(denoiser, scheduler).to(device)
    ddp_trainer = DistributedDataParallel(
        trainer,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    _event(rank, "ddp_wrapped", trainable_unet_parameters=trainability.trainable_parameters)

    tokens = tokenizer(
        sample_info["prompt"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    dtype = _precision_dtype(args.precision)
    autocast_enabled = args.precision != "fp32"
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=dtype, enabled=autocast_enabled
    ):
        text_hidden = text_encoder(**tokens).last_hidden_state
        scaling_factor = float(vae.config.scaling_factor)
        image_latent = vae.encode(image).latent_dist.mode() * scaling_factor
        target_latent = vae.encode(target).latent_dist.mode() * scaling_factor
    del image, target, text_encoder, vae, tokens, tokenizer
    torch.cuda.empty_cache()
    _event(rank, "latents_encoded", shape=list(image_latent.shape))

    batch = UnifiedLatentBatch(
        image_latent=image_latent,
        clean_target_latent=target_latent,
        task_name="segmentation",
        latent_valid_mask=valid_mask,
        text_hidden_states=text_hidden,
    )
    groups, trainable_parameters = _optimizer_groups(denoiser, config["training"])
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    reference_name, reference_parameter = next(
        (name, parameter)
        for name, parameter in denoiser.named_parameters()
        if parameter.requires_grad
    )
    reference_before = reference_parameter.detach().float().clone()
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    fixed_noise = torch.randn(
        target_latent.shape,
        generator=generator,
        device=device,
        dtype=target_latent.dtype,
    )
    fixed_timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps),
        (target_latent.shape[0],),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    torch.cuda.reset_peak_memory_stats(device)

    step_summaries: list[dict[str, Any]] = []
    for step_index in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=autocast_enabled
        ):
            output = ddp_trainer(
                batch,
                noise=fixed_noise,
                timesteps=fixed_timesteps,
            )
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = _gradient_norm(trainable_parameters)
        finite = torch.tensor(
            [int(torch.isfinite(grad_norm) and float(grad_norm) > 0)],
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise FloatingPointError("at least one rank produced invalid gradients")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        metrics = torch.tensor(
            [
                float(output.loss.detach()),
                float(grad_norm.detach()),
                time.perf_counter() - started,
            ],
            device=device,
            dtype=torch.float64,
        )
        gathered_metrics = _all_gather_vector(metrics, world_size)
        if rank == 0:
            step_summaries.append(
                {
                    "step": step_index + 1,
                    "loss_by_rank": [row[0] for row in gathered_metrics],
                    "gradient_norm_by_rank": [row[1] for row in gathered_metrics],
                    "seconds_by_rank": [round(row[2], 3) for row in gathered_metrics],
                }
            )
        _event(
            rank,
            "optimizer_step_completed",
            step=step_index + 1,
            loss=float(output.loss.detach()),
            gradient_norm=float(grad_norm.detach()),
        )

    update_l2 = float(
        (reference_parameter.detach().float() - reference_before).square().sum().sqrt()
    )
    checksum = float(reference_parameter.detach().float().sum())
    squared_checksum = float(reference_parameter.detach().float().square().sum())
    gathered_replicas = _all_gather_vector(
        torch.tensor(
            [update_l2, checksum, squared_checksum],
            device=device,
            dtype=torch.float64,
        ),
        world_size,
    )
    replica_summary = validate_replica_measurements(
        [row[0] for row in gathered_replicas],
        [row[1] for row in gathered_replicas],
        [row[2] for row in gathered_replicas],
    )
    dist.barrier()

    result = {
        "status": "SEGMENTATION_DDP_TRAINING_GATE_PASSED",
        "formal_experiment": False,
        "network_used": False,
        "checkpoint_written": False,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "config": str(args.config.expanduser().resolve()),
        "model": str(model_dir),
        "precision": args.precision,
        "image_size": args.image_size,
        "steps": args.steps,
        "sample": sample_info,
        "reference_parameter": reference_name,
        "reference_parameter_update_l2": update_l2,
        "replica_consistency": replica_summary,
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
    }
    if rank == 0:
        result["steps_detail"] = step_summaries
    return result


def main() -> int:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rank = int(os.environ.get("RANK", "0"))
    try:
        result = run_gate(args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "SEGMENTATION_DDP_TRAINING_GATE_FAILED",
                    "formal_experiment": False,
                    "rank": rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
