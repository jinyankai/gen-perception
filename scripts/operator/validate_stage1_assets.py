#!/usr/bin/env python3
"""Validate downloaded stage-one datasets and SD2 components offline.

This is an asset/integration smoke test, not a training run or benchmark. It:

1. resolves the real project configuration against the supplied asset roots;
2. reads one ADE20K image/mask and sends a binary class mask through the
   project's segmentation codec;
3. reads a small slice of the official NYUv2 labeled HDF5/MAT file and sends a
   depth sample through the project's depth codec; and
4. loads the local SD2 tokenizer, text encoder, VAE, scheduler, and U-Net, then
   runs one forward pass through the project's UnifiedPerceptionDenoiser.

No network access is used. The default device is CPU so the check does not
interfere with shared GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_diffusion.codecs import (  # noqa: E402
    DepthCodec,
    SegmentationBinaryMaskCodec,
)
from perception_diffusion.models import build_unified_denoiser  # noqa: E402
from perception_diffusion.utils.config import load_config  # noqa: E402


EXPECTED_MODEL_FILES = (
    "download_manifest.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

COMPATIBLE_SD2_REPOS = {
    "stabilityai/stable-diffusion-2",
    "sd2-community/stable-diffusion-2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_CACHE", "models")) / "stable-diffusion-2",
        help="Local Diffusers-format stabilityai/stable-diffusion-2 directory.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data")),
        help="Root containing ADEChallengeData2016/ and nyuv2/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "multitask" / "stage1_shared_unet.yaml",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is the safe default on a shared server; CUDA uses the current GPU.",
    )
    parser.add_argument(
        "--skip-model-forward",
        action="store_true",
        help="Only validate manifests, configuration, and dataset readability.",
    )
    return parser.parse_args()


def require_file(path: Path, *, minimum_bytes: int = 1) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < minimum_bytes:
        raise ValueError(f"file is unexpectedly small ({size} bytes): {path}")
    return size


def validate_model_files(model_dir: Path) -> dict[str, Any]:
    sizes = {
        relative: require_file(model_dir / relative)
        for relative in EXPECTED_MODEL_FILES
    }
    manifest_path = model_dir / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_id = manifest.get("repo_id")
    if repo_id not in COMPATIBLE_SD2_REPOS:
        raise ValueError(
            f"unexpected model repo in manifest: {repo_id!r}; "
            f"expected one of {sorted(COMPATIBLE_SD2_REPOS)}"
        )
    if manifest.get("download_completed") is not True:
        raise ValueError("model download manifest is not marked completed")
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) < 7:
        raise ValueError("model manifest has no immutable resolved revision")
    return {
        "path": str(model_dir),
        "repo_id": repo_id,
        "resolved_revision": revision,
        "download_method": manifest.get("download_method", "unknown"),
        "revision_verification": manifest.get("revision_verification", "unknown"),
        "registered_local_file_count": manifest.get("local_file_count"),
        "selected_file_bytes": sum(sizes.values()),
        "weight_files": {
            name: size for name, size in sizes.items() if name.endswith(".safetensors")
        },
    }


def validate_config_paths(
    config_path: Path, model_dir: Path, data_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ["MODEL_CACHE"] = str(model_dir.parent)
    os.environ["DATA_ROOT"] = str(data_root)
    os.environ.setdefault("OUTPUT_ROOT", str(data_root.parent / "outputs"))
    config = load_config(config_path)

    configured_model = Path(config["model"]["backbone"]["pretrained_model_path"])
    configured_ade = Path(config["data"]["datasets"]["segmentation"]["root"])
    configured_nyu = Path(config["data"]["datasets"]["depth"]["root"])
    expected_ade = data_root / "ADEChallengeData2016"
    expected_nyu = data_root / "nyuv2"
    if configured_model != model_dir:
        raise ValueError(f"config model path {configured_model} != requested {model_dir}")
    if configured_ade != expected_ade:
        raise ValueError(f"config ADE20K path {configured_ade} != expected {expected_ade}")
    if configured_nyu != expected_nyu:
        raise ValueError(f"config NYUv2 path {configured_nyu} != expected {expected_nyu}")
    return config, {
        "config": str(config_path),
        "model": str(configured_model),
        "ade20k": str(configured_ade),
        "nyuv2": str(configured_nyu),
    }


def validate_ade20k(ade_root: Path) -> dict[str, Any]:
    image_root = ade_root / "images" / "training"
    mask_root = ade_root / "annotations" / "training"
    object_info = ade_root / "objectInfo150.txt"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(
            f"expected ADE20K images/training and annotations/training under {ade_root}"
        )
    require_file(object_info, minimum_bytes=100)

    image_path = next(iter(sorted(image_root.rglob("*.jpg"))), None)
    if image_path is None:
        raise FileNotFoundError(f"no training JPG found under {image_root}")
    relative = image_path.relative_to(image_root)
    mask_path = mask_root / relative.with_suffix(".png")
    require_file(mask_path, minimum_bytes=100)

    with Image.open(image_path) as image:
        image_size = image.size
        image_mode = image.mode
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)
        mask_size = mask_image.size
    if image_size != mask_size:
        raise ValueError(f"ADE20K image/mask size mismatch: {image_size} vs {mask_size}")
    if mask.ndim != 2:
        raise ValueError(f"ADE20K mask must be HxW, got {mask.shape}")
    if int(mask.min()) < 0 or int(mask.max()) > 150:
        raise ValueError(f"ADE20K mask values outside official 0..150: {mask.min()}..{mask.max()}")

    present = np.unique(mask[mask > 0])
    if present.size == 0:
        raise ValueError(f"ADE20K sample contains no labeled class: {mask_path}")
    selected_class = int(present[0])
    binary_mask = (mask == selected_class).astype(np.uint8)
    valid_mask = mask != 0
    codec = SegmentationBinaryMaskCodec(threshold=0.5)
    encoded = codec.encode(binary_mask, valid_mask)
    decoded = codec.decode(encoded.values, encoded.valid_mask)
    if not np.array_equal(decoded[valid_mask], binary_mask[valid_mask]):
        raise ValueError("project segmentation codec failed ADE20K sample round trip")

    metadata_rows = [
        line
        for line in object_info.read_text(encoding="utf-8-sig").splitlines()[1:]
        if line.strip()
    ]
    if len(metadata_rows) != 150:
        raise ValueError(f"objectInfo150.txt should contain 150 classes, got {len(metadata_rows)}")
    return {
        "root": str(ade_root),
        "sample_image": str(image_path),
        "sample_mask": str(mask_path),
        "image_mode": image_mode,
        "image_size": list(image_size),
        "mask_range": [int(mask.min()), int(mask.max())],
        "metadata_classes": len(metadata_rows),
        "codec": type(codec).__name__,
        "codec_round_trip": True,
    }


def _small_hdf5_slice(dataset: Any) -> np.ndarray:
    sample_axis = next(
        (index for index, size in enumerate(dataset.shape) if size == 1449), 0
    )
    selection = []
    for index, size in enumerate(dataset.shape):
        limit = 1 if index == sample_axis else min(int(size), 16)
        selection.append(slice(0, limit))
    return np.asarray(dataset[tuple(selection)])


def validate_nyuv2(nyu_root: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for NYUv2; reinstall the updated requirements.txt"
        ) from exc

    mat_path = nyu_root / "nyu_depth_v2_labeled.mat"
    require_file(mat_path, minimum_bytes=2_000_000_000)
    required_keys = {"images", "depths", "rawDepths", "labels"}
    with h5py.File(mat_path, "r") as handle:
        missing = sorted(required_keys - set(handle.keys()))
        if missing:
            raise ValueError(f"NYUv2 MAT file is missing datasets: {missing}")
        shapes = {key: list(handle[key].shape) for key in sorted(required_keys)}
        if not all(1449 in shape for shape in shapes.values()):
            raise ValueError(f"NYUv2 datasets should include 1449 labeled frames: {shapes}")
        depth_slice = _small_hdf5_slice(handle["depths"])
        image_slice = _small_hdf5_slice(handle["images"])
    if not np.all(np.isfinite(depth_slice)):
        raise ValueError("NYUv2 depth sample contains non-finite values")

    depth_2d = np.squeeze(depth_slice)
    if depth_2d.ndim != 2:
        raise ValueError(f"could not reduce NYUv2 depth sample to HxW: {depth_slice.shape}")
    codec = DepthCodec(min_depth=0.1, max_depth=10.0)
    encoded = codec.encode(depth_2d)
    decoded = codec.decode(encoded.values, encoded.valid_mask)
    if not np.all(np.isfinite(decoded[encoded.valid_mask])):
        raise ValueError("project depth codec produced non-finite valid values")
    return {
        "file": str(mat_path),
        "file_bytes": mat_path.stat().st_size,
        "datasets": shapes,
        "sample_image_shape": list(image_slice.shape),
        "sample_depth_shape": list(depth_slice.shape),
        "sample_depth_range": [float(depth_slice.min()), float(depth_slice.max())],
        "codec": type(codec).__name__,
        "codec_valid_pixels": int(encoded.valid_mask.sum()),
    }


def validate_model_forward(
    model_dir: Path, config: dict[str, Any], device_name: str
) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

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

    text_encoder.eval()
    vae.eval()
    unet.eval()
    tokens = tokenizer(
        "monocular depth estimation",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    pixels = torch.zeros(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        text_hidden_states = text_encoder(**tokens).last_hidden_state
        latent_distribution = vae.encode(pixels).latent_dist
        image_latent = latent_distribution.mode() * float(vae.config.scaling_factor)

    denoiser, trainability = build_unified_denoiser(unet, config)
    denoiser.to(device).eval()
    timestep = torch.tensor([1], device=device, dtype=torch.long)
    noise = torch.zeros_like(image_latent)
    noisy_target = scheduler.add_noise(image_latent, noise, timestep)
    with torch.inference_mode():
        output = denoiser(
            image_latent,
            noisy_target,
            timestep,
            "depth",
            text_hidden_states=text_hidden_states,
        )
        decoded = vae.decode(image_latent / float(vae.config.scaling_factor)).sample
    if not torch.isfinite(output.sample).all() or not torch.isfinite(decoded).all():
        raise ValueError("real-component forward produced non-finite tensors")
    return {
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "token_shape": list(tokens["input_ids"].shape),
        "text_hidden_shape": list(text_hidden_states.shape),
        "vae_latent_shape": list(image_latent.shape),
        "vae_decode_shape": list(decoded.shape),
        "unet_input_channels_after_project_wrap": denoiser.shared_unet.conv_in.in_channels,
        "denoiser_output_shape": list(output.sample.shape),
        "conditioning_shape": list(output.encoder_hidden_states.shape),
        "trainable_scope": trainability.scope,
        "trainable_parameters": trainability.trainable_parameters,
        "finite": True,
    }


def main() -> int:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_dir = args.model_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    results: dict[str, Any] = {
        "status": "RUNNING",
        "formal_experiment": False,
        "network_used": False,
    }
    try:
        results["model_files"] = validate_model_files(model_dir)
        config, config_result = validate_config_paths(
            args.config.expanduser().resolve(), model_dir, data_root
        )
        results["config_paths"] = config_result
        results["ade20k"] = validate_ade20k(data_root / "ADEChallengeData2016")
        results["nyuv2"] = validate_nyuv2(data_root / "nyuv2")
        if args.skip_model_forward:
            results["model_forward"] = {"skipped": True}
        else:
            results["model_forward"] = validate_model_forward(
                model_dir, config, args.device
            )
        results["status"] = "STAGE1_ASSET_VALIDATION_PASSED"
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        results["status"] = "STAGE1_ASSET_VALIDATION_FAILED"
        results["error_type"] = type(exc).__name__
        results["error"] = str(exc)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
