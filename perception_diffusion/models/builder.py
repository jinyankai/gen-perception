"""Configuration-driven construction of the unified latent denoiser."""

from __future__ import annotations

from typing import Any

from torch import nn

from .conditioning import TaskTokenConditioner
from .unet import TrainabilitySummary, configure_unet_trainability, expand_unet_conv_in
from .unified_denoiser import UnifiedPerceptionDenoiser


def build_unified_denoiser(
    shared_unet: nn.Module, config: dict[str, Any]
) -> tuple[UnifiedPerceptionDenoiser, TrainabilitySummary]:
    """Wrap a loaded Diffusers-compatible U-Net using the resolved config."""

    model_config = config["model"]
    backbone = model_config["backbone"]
    conditioning = model_config["conditioning"]
    condition_adapter = model_config["condition_adapter"]
    shared_unet_config = model_config["shared_unet"]

    image_channels = int(backbone["image_latent_channels"])
    target_channels = int(backbone["target_latent_channels"])
    expand_unet_conv_in(
        shared_unet,
        image_channels,
        target_channels,
        initialization=backbone["conv_in_initialization"],
    )
    trainability = configure_unet_trainability(
        shared_unet, shared_unet_config["trainable_scope"]
    )
    conditioner = TaskTokenConditioner(
        conditioning["task_names"],
        int(conditioning["cross_attention_dim"]),
        num_task_tokens=int(conditioning["num_task_tokens"]),
        adapter_bottleneck_dim=int(condition_adapter["bottleneck_dim"]),
        adapter_scale_init=float(condition_adapter["residual_scale_init"]),
        text_input_dim=int(conditioning["text_input_dim"]),
        dropout=float(conditioning["dropout"]),
        use_task_condition=bool(conditioning.get("use_task_condition", True)),
        use_text_condition=bool(conditioning.get("use_text_condition", True)),
    )
    model = UnifiedPerceptionDenoiser(
        shared_unet,
        conditioner,
        image_latent_channels=image_channels,
        target_latent_channels=target_channels,
    )
    return model, trainability
