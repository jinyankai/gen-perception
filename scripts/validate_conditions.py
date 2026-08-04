#!/usr/bin/env python3
"""Visualize and quantify output changes under task/text condition ablations."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.data import build_dataloader  # noqa: E402
from perception_diffusion.inference.latent_sampler import UnifiedLatentSampler  # noqa: E402
from perception_diffusion.inference.runner import condition_mode_switches  # noqa: E402
from perception_diffusion.models import load_pretrained_system  # noqa: E402
from perception_diffusion.training import load_model_checkpoint, seed_everything  # noqa: E402
from perception_diffusion.utils.config import configured_task_names, load_config  # noqa: E402
from perception_diffusion.visualization import vae_pixels_to_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", required=True, choices=("segmentation", "depth", "normal"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten()
    right_flat = right.float().flatten()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) == 0.0:
        return 0.0 if torch.equal(left_flat, right_flat) else 1.0
    return float(1.0 - torch.dot(left_flat, right_flat) / denominator)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.num_steps <= 0:
        raise ValueError("num-steps must be positive")
    output_dir = args.output_dir.expanduser()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"condition output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    tasks = configured_task_names(config)
    if args.task not in tasks:
        raise ValueError(f"task {args.task!r} is not selected by the config")
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    seed_everything(seed)
    system = load_pretrained_system(
        config,
        device_override=args.device,
        precision_override=args.precision,
        for_training=False,
    )
    if args.checkpoint is not None:
        checkpoint_step = load_model_checkpoint(
            args.checkpoint,
            denoiser=system.denoiser,
            target_adapter=system.visual_pathway.target_adapter,
            device=system.device,
        )
    else:
        checkpoint_step = None
    task_data = (
        config["data"]["datasets"][args.task]
        if config["task"]["name"] == "multitask"
        else config["data"]
    )
    loader = build_dataloader(
        config,
        args.task,
        split=str(task_data.get("validation_split", task_data["split"])),
        training=False,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        strict_protocol=True,
        max_samples=1,
    )
    batch = next(iter(loader))
    image = batch["image"].to(system.device)
    image_latent = system.visual_pathway.encode_images(image)
    generator = torch.Generator(device=system.device).manual_seed(seed)
    initial_noise = torch.randn(
        (
            1,
            system.denoiser.target_latent_channels,
            image_latent.shape[-2],
            image_latent.shape[-1],
        ),
        device=system.device,
        dtype=image_latent.dtype,
        generator=generator,
    )
    original_prompt = str(batch["text_condition"][0])
    prompts = [original_prompt, *args.prompt]
    if len(prompts) == 1:
        prompts.extend(["", "an unrelated visual concept"])
    variants: list[tuple[str, str, str]] = []
    for mode in ("full", "task_only", "text_only", "unconditional"):
        variants.append((f"mode-{mode}", args.task, original_prompt))
    for index, prompt in enumerate(prompts[1:], start=1):
        variants.append((f"prompt-{index}", args.task, prompt))
    for task_name in tasks:
        if task_name != args.task:
            variants.append((f"task-{task_name}", task_name, original_prompt))

    sampler = UnifiedLatentSampler(system.denoiser, system.inference_scheduler)
    outputs: dict[str, dict[str, object]] = {}
    decoded_images: list[np.ndarray] = []
    for label, task_name, prompt in variants:
        mode = label.removeprefix("mode-") if label.startswith("mode-") else "full"
        use_task, use_text = condition_mode_switches(mode)  # type: ignore[arg-type]
        text_hidden = system.encode_prompts([prompt])
        variant_generator = torch.Generator(device=system.device).manual_seed(seed + 1)
        latent = sampler.sample(
            image_latent,
            task_name,
            num_inference_steps=args.num_steps,
            text_hidden_states=text_hidden,
            initial_noise=initial_noise,
            generator=variant_generator,
            use_task_condition=use_task,
            use_text_condition=use_text,
        )
        pixels = system.visual_pathway.decode_latents(latent)
        rgb = vae_pixels_to_rgb(pixels[0])
        Image.fromarray(rgb).save(output_dir / f"{label}.png")
        decoded_images.append(rgb)
        outputs[label] = {
            "task": task_name,
            "prompt": prompt,
            "use_task_condition": use_task,
            "use_text_condition": use_text,
            "latent": latent.detach().cpu(),
            "pixels": pixels.detach().cpu(),
        }

    baseline = outputs["mode-full"]
    records: dict[str, object] = {}
    for label, output in outputs.items():
        records[label] = {
            "task": output["task"],
            "prompt": output["prompt"],
            "use_task_condition": output["use_task_condition"],
            "use_text_condition": output["use_text_condition"],
            "latent_l2_from_full": float(
                torch.mean((output["latent"] - baseline["latent"]) ** 2).sqrt()
            ),
            "latent_cosine_distance_from_full": _cosine_distance(
                output["latent"], baseline["latent"]
            ),
            "pixel_l1_from_full": float(
                torch.mean(torch.abs(output["pixels"] - baseline["pixels"]))
            ),
        }
    Image.fromarray(np.concatenate(decoded_images, axis=1)).save(output_dir / "comparison.png")
    summary = {
        "status": "CONDITION_VALIDATION_COMPLETED",
        "formal_experiment": False,
        "sample_id": str(batch["sample_id"][0]),
        "checkpoint_step": checkpoint_step,
        "checkpoint_loaded": args.checkpoint is not None,
        "model_path": str(system.model_path),
        "model_revision": system.model_revision,
        "seed": seed,
        "num_steps": args.num_steps,
        "baseline": "mode-full",
        "variants": records,
        "qualification": (
            "Non-zero differences prove condition sensitivity, not semantic correctness. "
            "Interpret task/prompt behavior only after an overfit or trained checkpoint."
        ),
    }
    (output_dir / "condition_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"CONDITION_VALIDATION_FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
