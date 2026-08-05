"""Shared training primitives for every perception task."""

from .noise import (
    ConfiguredNoiseSampler,
    annealed_noise_strength,
    multi_resolution_noise_like,
)
from .checkpoint import (
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from .distributed import (
    DistributedContext,
    resolve_distributed_context,
    single_process_context,
    wrap_ddp,
)
from .logging import TrainingLogger
from .optim import build_lr_scheduler, build_optimizer, gradient_norm
from .runner import RoundRobinBatchStream, run_training, seed_everything

from .unified_trainer import (
    DiffusionStepOutput,
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
)

__all__ = [
    "ConfiguredNoiseSampler",
    "DistributedContext",
    "DiffusionStepOutput",
    "UnifiedDiffusionTrainerCore",
    "UnifiedLatentBatch",
    "TrainingLogger",
    "RoundRobinBatchStream",
    "resolve_distributed_context",
    "single_process_context",
    "wrap_ddp",
    "annealed_noise_strength",
    "multi_resolution_noise_like",
    "build_lr_scheduler",
    "build_optimizer",
    "gradient_norm",
    "load_model_checkpoint",
    "load_training_checkpoint",
    "save_checkpoint",
    "run_training",
    "seed_everything",
]
