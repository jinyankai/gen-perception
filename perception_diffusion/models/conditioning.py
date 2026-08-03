"""Task-token and optional text conditioning for shared U-Net cross-attention."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .adapters import TaskAdapterBank
from .tasks import validate_task_names


class TaskTokenConditioner(nn.Module):
    """Build cross-attention states from learned task tokens and optional text tokens.

    Batches are expected to be task-homogeneous during stage one, but the module
    also supports a different task name for every sample.
    """

    def __init__(
        self,
        task_names: Sequence[str],
        cross_attention_dim: int,
        *,
        num_task_tokens: int = 4,
        adapter_bottleneck_dim: int = 256,
        adapter_scale_init: float = 0.0,
        text_input_dim: int | None = None,
        dropout: float = 0.0,
        use_task_condition: bool = True,
        use_text_condition: bool = True,
    ) -> None:
        super().__init__()
        self.task_names = validate_task_names(task_names)
        if cross_attention_dim <= 0 or num_task_tokens <= 0:
            raise ValueError("conditioning dimensions and token count must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(use_task_condition, bool) or not isinstance(
            use_text_condition, bool
        ):
            raise TypeError("condition switches must be boolean")

        self.cross_attention_dim = cross_attention_dim
        self.num_task_tokens = num_task_tokens
        self.use_task_condition = use_task_condition
        self.use_text_condition = use_text_condition
        self.task_to_id = {name: index for index, name in enumerate(self.task_names)}
        self.task_embeddings = nn.Embedding(
            len(self.task_names) * num_task_tokens, cross_attention_dim
        )
        nn.init.normal_(self.task_embeddings.weight, mean=0.0, std=0.02)

        self.adapters = TaskAdapterBank(
            list(self.task_names),
            cross_attention_dim,
            adapter_bottleneck_dim,
            residual_scale_init=adapter_scale_init,
        )
        if text_input_dim is None or text_input_dim == cross_attention_dim:
            self.text_projection: nn.Module = nn.Identity()
        else:
            if text_input_dim <= 0:
                raise ValueError("text_input_dim must be positive")
            self.text_projection = nn.Linear(text_input_dim, cross_attention_dim)
        self.dropout = nn.Dropout(dropout)

    def _normalize_batch_tasks(
        self, task_names: str | Sequence[str], batch_size: int
    ) -> tuple[str, ...]:
        if isinstance(task_names, str):
            names = (task_names,) * batch_size
        else:
            names = tuple(task_names)
        if len(names) != batch_size:
            raise ValueError(f"received {len(names)} task names for batch size {batch_size}")
        unknown = sorted(set(names) - set(self.task_names))
        if unknown:
            raise ValueError(f"conditioner is not configured for tasks: {unknown}")
        return names

    def _task_tokens(
        self,
        task_names: tuple[str, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = torch.arange(self.num_task_tokens, device=device)
        rows = [self.task_to_id[name] * self.num_task_tokens + offsets for name in task_names]
        token_ids = torch.stack(rows, dim=0)
        tokens = self.task_embeddings(token_ids)
        return self.adapters(tokens, list(task_names))

    def forward(
        self,
        task_names: str | Sequence[str],
        *,
        batch_size: int,
        text_hidden_states: torch.Tensor | None = None,
        use_task_condition: bool | None = None,
        use_text_condition: bool | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if text_hidden_states is not None:
            if text_hidden_states.ndim != 3 or text_hidden_states.shape[0] != batch_size:
                raise ValueError(
                    "text_hidden_states must have shape [B,L,D] with the requested batch size"
                )
            device = text_hidden_states.device
        else:
            device = self.task_embeddings.weight.device

        names = self._normalize_batch_tasks(task_names, batch_size)
        task_tokens = self._task_tokens(names, device=device)
        task_enabled = (
            self.use_task_condition
            if use_task_condition is None
            else use_task_condition
        )
        text_enabled = (
            self.use_text_condition
            if use_text_condition is None
            else use_text_condition
        )
        if not isinstance(task_enabled, bool) or not isinstance(text_enabled, bool):
            raise TypeError("condition switch overrides must be boolean")
        if not task_enabled:
            task_tokens = torch.zeros_like(task_tokens)
        if text_hidden_states is None:
            return self.dropout(task_tokens)

        text_tokens = self.text_projection(text_hidden_states)
        if text_tokens.shape[-1] != self.cross_attention_dim:
            raise ValueError(
                "projected text dimension does not match the U-Net cross-attention dimension"
            )
        if not text_enabled:
            text_tokens = torch.zeros_like(text_tokens)
        return self.dropout(torch.cat([task_tokens, text_tokens], dim=1))
