"""Offline loading boundary for the real Stable Diffusion 2 components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .builder import build_unified_denoiser
from .target_adapters import build_pre_vae_adapter
from .unet import TrainabilitySummary
from .unified_denoiser import UnifiedPerceptionDenoiser
from .visual_latent import VisualLatentPathway


def resolve_device(config: dict[str, Any], override: str | None = None) -> torch.device:
    requested = override or str(config.get("runtime", {}).get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported runtime device: {requested}")
    return torch.device(requested)


def resolve_precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    normalized = precision.casefold()
    if normalized in {"no", "fp32", "float32"}:
        return torch.float32
    if normalized in {"fp16", "float16"}:
        if device.type != "cuda":
            raise ValueError("fp16 execution requires CUDA")
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported mixed precision mode: {precision}")


@dataclass
class PretrainedPerceptionSystem:
    tokenizer: Any
    text_encoder: nn.Module
    visual_pathway: VisualLatentPathway
    denoiser: UnifiedPerceptionDenoiser
    training_scheduler: Any
    inference_scheduler: Any
    trainability: TrainabilitySummary
    device: torch.device
    autocast_dtype: torch.dtype
    model_path: Path
    model_revision: str

    @property
    def autocast_enabled(self) -> bool:
        return self.autocast_dtype != torch.float32

    @torch.no_grad()
    def encode_prompts(self, prompts: str | Sequence[str]) -> torch.Tensor:
        values = [prompts] if isinstance(prompts, str) else list(prompts)
        if not values or not all(isinstance(value, str) for value in values):
            raise ValueError("prompts must contain at least one string")
        tokens = self.tokenizer(
            values,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            output = self.text_encoder(**tokens)
        states = output.last_hidden_state
        if not torch.isfinite(states).all():
            raise FloatingPointError("text encoder produced non-finite states")
        return states


def load_pretrained_system(
    config: dict[str, Any],
    *,
    device_override: str | None = None,
    precision_override: str | None = None,
    for_training: bool = True,
) -> PretrainedPerceptionSystem:
    """Load revision-prepared local SD2 files without implicit network access."""

    try:
        from diffusers import (
            AutoencoderKL,
            DDIMScheduler,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:  # pragma: no cover - depends on server environment
        raise RuntimeError(
            "real model loading requires diffusers and transformers from requirements.txt"
        ) from exc

    distributed = config.get("runtime", {}).get("distributed", "disabled")
    if distributed not in {"disabled", "ddp"}:
        raise NotImplementedError(
            f"unsupported runtime.distributed={distributed!r}; expected disabled or ddp"
        )
    # For DDP the training runner initializes the process group and binds this
    # rank's device (torch.cuda.set_device) before calling in, so torch.device
    # ("cuda") below already resolves to the correct per-rank GPU.
    device = resolve_device(config, device_override)
    precision = precision_override or str(config["training"].get("mixed_precision", "no"))
    autocast_dtype = resolve_precision_dtype(precision, device)
    model_path = Path(config["model"]["backbone"]["pretrained_model_path"]).expanduser()
    if "$" in str(model_path):
        raise ValueError("pretrained_model_path contains an unresolved environment variable")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Stable Diffusion 2 directory does not exist: {model_path}")
    required = ("tokenizer", "text_encoder", "vae", "scheduler", "unet")
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"SD2 directory is missing components: {missing}")
    backbone = config["model"]["backbone"]
    manifest_path = model_path / "download_manifest.json"
    require_manifest = bool(backbone.get("require_download_manifest", True))
    if not manifest_path.is_file():
        if require_manifest:
            raise FileNotFoundError(f"SD2 download manifest does not exist: {manifest_path}")
        model_revision = "unverified"
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SD2 download manifest: {manifest_path}") from exc
        if manifest.get("download_completed") is not True:
            raise ValueError("SD2 download manifest is not marked completed")
        model_revision = manifest.get("resolved_revision")
        if model_revision != backbone.get("expected_revision"):
            raise ValueError(
                "SD2 resolved revision disagrees with configuration: "
                f"{model_revision!r} != {backbone.get('expected_revision')!r}"
            )
        if manifest.get("repo_id") != backbone.get("expected_repo_id"):
            raise ValueError(
                "SD2 repository ID disagrees with configuration: "
                f"{manifest.get('repo_id')!r} != {backbone.get('expected_repo_id')!r}"
            )

    tokenizer = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path / "text_encoder",
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_path / "vae", local_files_only=True, use_safetensors=True
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_path / "unet", local_files_only=True, use_safetensors=True
    ).to(device)
    expected_text_dim = int(config["model"]["conditioning"]["text_input_dim"])
    actual_text_dim = int(getattr(text_encoder.config, "hidden_size", -1))
    if actual_text_dim != expected_text_dim:
        raise ValueError(
            f"CLIP hidden size {actual_text_dim} != configured {expected_text_dim}"
        )
    expected_cross_dim = int(config["model"]["conditioning"]["cross_attention_dim"])
    actual_cross_dim = getattr(unet.config, "cross_attention_dim", None)
    cross_dims = (
        {int(value) for value in actual_cross_dim}
        if isinstance(actual_cross_dim, (list, tuple))
        else {int(actual_cross_dim)}
        if actual_cross_dim is not None
        else set()
    )
    if cross_dims != {expected_cross_dim}:
        raise ValueError(
            f"U-Net cross-attention dimensions {sorted(cross_dims)} != "
            f"configured {expected_cross_dim}"
        )
    actual_latent_channels = int(getattr(vae.config, "latent_channels", -1))
    expected_latent_channels = {
        int(config["model"]["backbone"]["image_latent_channels"]),
        int(config["model"]["backbone"]["target_latent_channels"]),
    }
    if expected_latent_channels != {actual_latent_channels}:
        raise ValueError(
            f"VAE latent channels {actual_latent_channels} disagree with configured "
            f"image/target channels {sorted(expected_latent_channels)}"
        )
    training_scheduler = DDPMScheduler.from_pretrained(
        model_path / "scheduler", local_files_only=True
    )
    prediction_type = str(config["model"].get("prediction_type", "epsilon"))
    scheduler_prediction_type = str(
        getattr(training_scheduler.config, "prediction_type", "epsilon")
    )
    if scheduler_prediction_type != prediction_type:
        raise ValueError(
            "model.prediction_type disagrees with the pretrained scheduler: "
            f"{prediction_type!r} != {scheduler_prediction_type!r}"
        )
    inference_name = str(config["inference"].get("scheduler", "ddim")).casefold()
    if inference_name == "ddim":
        inference_scheduler = DDIMScheduler.from_config(training_scheduler.config)
    elif inference_name == "ddpm":
        inference_scheduler = DDPMScheduler.from_config(training_scheduler.config)
    else:
        raise ValueError(f"unsupported inference scheduler: {inference_name}")

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    denoiser, trainability = build_unified_denoiser(unet, config)
    denoiser.to(device)
    denoiser.train(for_training)
    if for_training and bool(config["training"].get("gradient_checkpointing", False)):
        enable = getattr(denoiser.shared_unet, "enable_gradient_checkpointing", None)
        if not callable(enable):
            raise TypeError("configured gradient checkpointing is unsupported by this U-Net")
        enable()
    target_adapter = build_pre_vae_adapter(config).to(device)
    target_adapter.train(for_training)
    if not for_training:
        target_adapter.requires_grad_(False)
    visual_pathway = VisualLatentPathway(vae, target_adapter)
    return PretrainedPerceptionSystem(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        visual_pathway=visual_pathway,
        denoiser=denoiser,
        training_scheduler=training_scheduler,
        inference_scheduler=inference_scheduler,
        trainability=trainability,
        device=device,
        autocast_dtype=autocast_dtype,
        model_path=model_path.resolve(),
        model_revision=str(model_revision),
    )
