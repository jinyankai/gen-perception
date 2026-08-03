"""Shared U-Net adaptation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


ConvInInit = Literal["repeat_half", "preserve_image_zero_target"]
TrainableScope = Literal["full", "cross_attention", "frozen"]


@dataclass(frozen=True)
class TrainabilitySummary:
    scope: str
    trainable_parameters: int
    total_parameters: int
    trainable_names: tuple[str, ...]


def _update_unet_in_channels(unet: nn.Module, in_channels: int) -> None:
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=in_channels)
        return
    config = getattr(unet, "config", None)
    if isinstance(config, dict):
        config["in_channels"] = in_channels
    elif config is not None and hasattr(config, "in_channels"):
        setattr(config, "in_channels", in_channels)


def expand_unet_conv_in(
    unet: nn.Module,
    image_latent_channels: int,
    target_latent_channels: int,
    *,
    initialization: ConvInInit = "repeat_half",
) -> bool:
    """Expand a pretrained image U-Net input convolution for image+target latents.

    Returns True when the layer was replaced and False when it already had the
    requested channel count.
    """

    conv_in = getattr(unet, "conv_in", None)
    if not isinstance(conv_in, nn.Conv2d):
        raise TypeError("shared U-Net must expose conv_in as torch.nn.Conv2d")
    requested_channels = image_latent_channels + target_latent_channels
    if conv_in.in_channels == requested_channels:
        return False
    if conv_in.in_channels != image_latent_channels:
        raise ValueError(
            f"cannot expand conv_in with {conv_in.in_channels} channels from "
            f"image_latent_channels={image_latent_channels}"
        )
    if initialization not in {"repeat_half", "preserve_image_zero_target"}:
        raise ValueError(f"unsupported conv_in initialization: {initialization}")
    if initialization == "repeat_half" and image_latent_channels != target_latent_channels:
        raise ValueError("repeat_half requires equal image and target latent channels")

    replacement = nn.Conv2d(
        requested_channels,
        conv_in.out_channels,
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding,
        dilation=conv_in.dilation,
        groups=conv_in.groups,
        bias=conv_in.bias is not None,
        padding_mode=conv_in.padding_mode,
        device=conv_in.weight.device,
        dtype=conv_in.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.zero_()
        if initialization == "repeat_half":
            replacement.weight[:, :image_latent_channels].copy_(conv_in.weight * 0.5)
            replacement.weight[:, image_latent_channels:].copy_(conv_in.weight * 0.5)
        else:
            replacement.weight[:, :image_latent_channels].copy_(conv_in.weight)
        if conv_in.bias is not None:
            replacement.bias.copy_(conv_in.bias)

    unet.conv_in = replacement
    _update_unet_in_channels(unet, requested_channels)
    return True


def configure_unet_trainability(
    unet: nn.Module, scope: TrainableScope
) -> TrainabilitySummary:
    """Apply the stage-one shared-U-Net trainability policy."""

    if scope not in {"full", "cross_attention", "frozen"}:
        raise ValueError(f"unsupported shared U-Net trainable scope: {scope}")

    for parameter in unet.parameters():
        parameter.requires_grad_(scope == "full")

    if scope == "cross_attention":
        for name, parameter in unet.named_parameters():
            if name.startswith("conv_in.") or ".attn2." in name or "cross_attn" in name:
                parameter.requires_grad_(True)

    named_parameters = tuple(unet.named_parameters())
    trainable_names = tuple(name for name, p in named_parameters if p.requires_grad)
    return TrainabilitySummary(
        scope=scope,
        trainable_parameters=sum(p.numel() for _, p in named_parameters if p.requires_grad),
        total_parameters=sum(p.numel() for _, p in named_parameters),
        trainable_names=trainable_names,
    )
