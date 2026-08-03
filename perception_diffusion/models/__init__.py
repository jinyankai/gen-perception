"""Shared model components for unified generative perception."""

from .adapters import ResidualConditionAdapter, TaskAdapterBank
from .builder import build_unified_denoiser
from .conditioning import TaskTokenConditioner
from .target_adapters import (
    IdentityPreVAEAdapter,
    ResidualPreVAEAdapter,
    TaskPreVAEAdapterBank,
    build_pre_vae_adapter,
)
from .pretrained import (
    PretrainedPerceptionSystem,
    load_pretrained_system,
    resolve_device,
    resolve_precision_dtype,
)
from .tasks import TASK_NAMES, validate_task_names
from .unet import (
    TrainabilitySummary,
    configure_unet_trainability,
    expand_unet_conv_in,
)
from .unified_denoiser import UnifiedDenoiserOutput, UnifiedPerceptionDenoiser
from .visual_latent import VisualLatentPair, VisualLatentPathway

__all__ = [
    "IdentityPreVAEAdapter",
    "PretrainedPerceptionSystem",
    "ResidualConditionAdapter",
    "ResidualPreVAEAdapter",
    "TASK_NAMES",
    "TaskAdapterBank",
    "TaskTokenConditioner",
    "TaskPreVAEAdapterBank",
    "TrainabilitySummary",
    "UnifiedDenoiserOutput",
    "UnifiedPerceptionDenoiser",
    "VisualLatentPair",
    "VisualLatentPathway",
    "build_unified_denoiser",
    "build_pre_vae_adapter",
    "configure_unet_trainability",
    "expand_unet_conv_in",
    "load_pretrained_system",
    "resolve_device",
    "resolve_precision_dtype",
    "validate_task_names",
]
