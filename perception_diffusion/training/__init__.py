"""Shared training primitives for every perception task."""

from .unified_trainer import (
    DiffusionStepOutput,
    UnifiedDiffusionTrainerCore,
    UnifiedLatentBatch,
)

__all__ = [
    "DiffusionStepOutput",
    "UnifiedDiffusionTrainerCore",
    "UnifiedLatentBatch",
]
