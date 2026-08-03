"""Task-specific adapters used to specialize shared cross-attention conditions."""

from __future__ import annotations

import torch
from torch import nn

from .tasks import validate_task_names


class ResidualConditionAdapter(nn.Module):
    """A small residual bottleneck operating on cross-attention tokens."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        *,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(self.norm(tokens))))
        return tokens + self.residual_scale * residual


class TaskAdapterBank(nn.Module):
    """Dispatch every sample to the adapter owned by its requested task."""

    def __init__(
        self,
        task_names: tuple[str, ...] | list[str],
        hidden_dim: int,
        bottleneck_dim: int,
        *,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.task_names = validate_task_names(task_names)
        self.adapters = nn.ModuleDict(
            {
                task_name: ResidualConditionAdapter(
                    hidden_dim,
                    bottleneck_dim,
                    residual_scale_init=residual_scale_init,
                )
                for task_name in self.task_names
            }
        )

    def forward(
        self, tokens: torch.Tensor, task_names: tuple[str, ...] | list[str]
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,L,D], got {tuple(tokens.shape)}")
        if len(task_names) != tokens.shape[0]:
            raise ValueError(
                f"received {len(task_names)} task names for batch size {tokens.shape[0]}"
            )
        unknown = sorted(set(task_names) - set(self.task_names))
        if unknown:
            raise ValueError(f"task adapter is not configured for: {unknown}")
        if len(set(task_names)) == 1:
            return self.adapters[task_names[0]](tokens)

        adapted = [
            self.adapters[task_name](sample_tokens.unsqueeze(0))
            for sample_tokens, task_name in zip(tokens, task_names, strict=True)
        ]
        return torch.cat(adapted, dim=0)
